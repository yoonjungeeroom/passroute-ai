import asyncio
import base64
import logging
import time
from collections import deque

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sqlalchemy import select
from app.services.face_analysis_service import analyze_frame, create_face_landmarker
from app.core.database import AsyncSessionLocal
from app.models.face_analysis import FaceAnalysis

logger = logging.getLogger(__name__)
router = APIRouter()

GAZE_WINDOW_SEC = 10
BLINK_IBI_COUNT = 5           # IBI 계산에 사용할 최근 깜빡임 수
BLINK_IBI_THRESHOLD = 2.0     # 평균 깜빡임 간격(초) 이하면 과다 판정
BLINK_DISPLAY_WINDOW_SEC = 10 # 화면 표시·리포트용 슬라이딩 윈도우
GAZE_RATIO_FEEDBACK_THRESHOLD = 60.0
FEEDBACK_COOLDOWN_SEC = 5.0
GAZE_RATIO_SAMPLE_INTERVAL = 30


@router.websocket("/ws/face/{session_id}/{question_id}")
async def face_websocket(websocket: WebSocket, session_id: str, question_id: str):
    await websocket.accept()
    ws_lock = asyncio.Lock()
    is_connected = True

    face_landmarker = await asyncio.to_thread(create_face_landmarker)

    gaze_window: deque = deque()
    blink_display: deque = deque()                    # 10초 표시용
    blink_ibi: deque = deque(maxlen=BLINK_IBI_COUNT)  # IBI 계산용 (최근 5개)
    prev_blink = False
    prev_gaze_on = True

    gaze_off_count = 0
    total_blink_count = 0
    gaze_ratio_values: list = []

    frame_count = 0
    last_feedback: dict = {}
    start_time = time.time()

    async def send_ws(data: dict):
        async with ws_lock:
            await websocket.send_json(data)

    def can_feedback(fb_type: str, cooldown: float = FEEDBACK_COOLDOWN_SEC) -> bool:
        now = time.time()
        if now - last_feedback.get(fb_type, 0) >= cooldown:
            last_feedback[fb_type] = now
            return True
        return False

    async def send_feedback(fb_type: str, message: str, cooldown: float = FEEDBACK_COOLDOWN_SEC, **extra):
        if can_feedback(fb_type, cooldown):
            await send_ws({"status": "feedback", "type": fb_type, "message": message, **extra})

    try:
        while True:
            data = await websocket.receive_text()
            now = time.time()
            frame_count += 1

            try:
                if "," in data:
                    data = data.split(",", 1)[1]
                img_bytes = base64.b64decode(data)
                np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            except Exception:
                continue

            result = await asyncio.to_thread(analyze_frame, face_landmarker, frame)
            face_detected = result["face_detected"]
            gaze_on = result["gaze_on"]
            blink = result["blink"]
            ear = result["ear"]

            # Gaze off event (transition: gaze_on → gaze_off)
            if prev_gaze_on and not (face_detected and gaze_on):
                gaze_off_count += 1
            prev_gaze_on = face_detected and gaze_on

            # Gaze sliding window for ratio
            gaze_window.append((now, face_detected and gaze_on))
            cutoff = now - GAZE_WINDOW_SEC
            while gaze_window and gaze_window[0][0] < cutoff:
                gaze_window.popleft()

            total_w = len(gaze_window)
            gazing_w = sum(1 for _, g in gaze_window if g)
            gaze_ratio = round(gazing_w / total_w * 100, 1) if total_w > 0 else 0.0

            # Blink detection (rising edge)
            if blink and not prev_blink:
                blink_display.append(now)
                blink_ibi.append(now)
                total_blink_count += 1
            prev_blink = blink

            while blink_display and blink_display[0] < now - BLINK_DISPLAY_WINDOW_SEC:
                blink_display.popleft()

            blink_in_window = len(blink_display)

            # Sample gaze ratio for report every N frames
            if frame_count % GAZE_RATIO_SAMPLE_INTERVAL == 0:
                gaze_ratio_values.append(gaze_ratio)

            if not is_connected:
                continue

            await send_ws({
                "status": "face_data",
                "face_detected": face_detected,
                "gaze_on": gaze_on,
                "gaze_ratio": gaze_ratio,
                "blink_in_window": blink_in_window,
                "ear": ear,
            })

            # Feedback
            if not face_detected:
                await send_feedback("gaze_off", "화면에서 얼굴이 감지되지 않습니다. 카메라를 확인해주세요.")
            elif not gaze_on:
                await send_feedback("gaze_off", "시선이 화면을 벗어나고 있습니다. 카메라를 바라봐 주세요.")
            elif gaze_ratio < GAZE_RATIO_FEEDBACK_THRESHOLD:
                await send_feedback("gaze_low", "시선 집중률이 낮습니다. 카메라를 바라봐 주세요.")

            # IBI 기반 깜빡임 과다 감지
            if len(blink_ibi) >= BLINK_IBI_COUNT:
                intervals = [blink_ibi[i + 1] - blink_ibi[i] for i in range(BLINK_IBI_COUNT - 1)]
                avg_ibi = sum(intervals) / len(intervals)
                if avg_ibi < BLINK_IBI_THRESHOLD:
                    if can_feedback("blink_high", cooldown=10.0):
                        await send_ws({"status": "feedback", "type": "blink_high", "message": "눈 깜빡임이 많습니다. 긴장을 풀고 편안하게 답변해 주세요."})

    except WebSocketDisconnect:
        is_connected = False
    except Exception as e:
        is_connected = False
        logger.error(f"Face WebSocket 루프 에러: {e}")
    finally:
        try:
            face_landmarker.close()
        except Exception as e:
            logger.error(f"face_landmarker 닫기 실패: {e}")

        duration_sec = round(time.time() - start_time, 2)
        try:
            avg_gaze_ratio = round(sum(gaze_ratio_values) / len(gaze_ratio_values), 2) if gaze_ratio_values else 0.0
            avg_blink_per_min = round(total_blink_count / (duration_sec / 60), 2) if duration_sec > 0 else 0.0
            async with AsyncSessionLocal() as db:
                db.add(FaceAnalysis(
                    session_id=session_id,
                    question_id=question_id,
                    gaze_off_count=gaze_off_count,
                    avg_gaze_ratio=avg_gaze_ratio,
                    avg_blink_per_min=avg_blink_per_min,
                ))
                await db.commit()
            logger.info(f"[{session_id}:{question_id}] 영상 분석 결과 MySQL 저장 완료")
        except Exception as e:
            logger.error(f"영상 분석 결과 MySQL 저장 에러: {e}")


@router.get("/face/summary/{session_id}/{question_id}")
async def get_face_analysis_summary(session_id: str, question_id: str):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(FaceAnalysis)
            .where(
                FaceAnalysis.session_id == session_id,
                FaceAnalysis.question_id == question_id,
            )
            .order_by(FaceAnalysis.created_at.desc())
            .limit(1)
        )
        row = result.scalars().first()
        if row is None:
            return {"gaze_off_count": 0, "avg_gaze_ratio": 0.0, "avg_blink_per_min": 0.0}
        return {
            "gaze_off_count": row.gaze_off_count,
            "avg_gaze_ratio": row.avg_gaze_ratio,
            "avg_blink_per_min": row.avg_blink_per_min,
        }
