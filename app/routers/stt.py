import asyncio
import logging
import time
from collections import deque

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from sqlalchemy import text
from app.services.stt_service import detect_voice, transcribe_audio, SAMPLE_RATE
from app.services.voice_analysis_service import count_filler_words
from app.core.database import AsyncSessionLocal
from app.models.voice_analysis import VoiceAnalysis

logger = logging.getLogger(__name__)
router = APIRouter()

WPM_SLOW = 100
WPM_FAST = 180
WPM_WINDOW_SEC = 15
SILENCE_ALERT_SEC = 5.0
FEEDBACK_COOLDOWN_SEC = 5.0
CONSECUTIVE_FILLER_THRESHOLD = 3
MIN_SILENCE_DURATION = 0.5


@router.websocket("/ws/stt/{session_id}/{question_id}")
async def stt_websocket(websocket: WebSocket, session_id: str, question_id: str):
    await websocket.accept()
    if not session_id.isdigit() or not question_id.isdigit():
        await websocket.close(code=1008)
        return
    audio_chunks = []
    ws_lock = asyncio.Lock()
    stt_queue: asyncio.Queue = asyncio.Queue()

    speech_segments: deque = deque()
    consecutive_filler_segments = 0

    silence_start: float | None = None
    last_feedback: dict = {}
    is_connected = True

    total_words = 0
    total_speech_sec = 0.0
    silence_values: list = []
    filler_count_total = 0
    stt_parts: list = []

    async def send_ws(data: dict):
        async with ws_lock:
            await websocket.send_json(data)

    def can_feedback(fb_type: str) -> bool:
        now = time.time()
        if now - last_feedback.get(fb_type, 0) >= FEEDBACK_COOLDOWN_SEC:
            last_feedback[fb_type] = now
            return True
        return False

    async def send_feedback(fb_type: str, message: str, **extra):
        if can_feedback(fb_type):
            await send_ws({"status": "feedback", "type": fb_type, "message": message, **extra})

    async def stt_worker():
        while True:
            item = await stt_queue.get()
            if item is None:
                stt_queue.task_done()
                break
            audio_buffer, segment_sec, ts = item
            await process_stt(audio_buffer, segment_sec, ts)
            stt_queue.task_done()

    worker_task = asyncio.create_task(stt_worker())

    async def process_stt(audio_buffer: np.ndarray, segment_sec: float, ts: float):
        nonlocal consecutive_filler_segments, total_words, total_speech_sec, filler_count_total
        try:
            text = await transcribe_audio(audio_buffer)
            if not text:
                return

            stt_parts.append(text)

            # WPM 슬라이딩 윈도우 (실시간 피드백용)
            words = len([w for w in text.split() if w])
            speech_segments.append((ts, words, segment_sec))
            cutoff = ts - WPM_WINDOW_SEC
            while speech_segments and speech_segments[0][0] < cutoff:
                speech_segments.popleft()

            window_words = sum(s[1] for s in speech_segments)
            window_sec = sum(s[2] for s in speech_segments)
            wpm = round(window_words / window_sec * 60, 1) if window_sec > 0 else 0.0

            # 리포트용 누적
            if words > 0:
                total_words += words
                total_speech_sec += segment_sec

            # 필러워드
            filler_count = count_filler_words(text)
            filler_count_total += filler_count
            if filler_count > 0:
                consecutive_filler_segments += 1
            else:
                consecutive_filler_segments = 0

            if not is_connected:
                return

            await send_ws({
                "status": "completed",
                "text": text,
                "session_id": session_id,
                "question_id": question_id,
                "wpm": wpm,
                "filler_count": filler_count,
            })

            if 0 < wpm < WPM_SLOW:
                await send_feedback("wpm_slow", "말이 느린 편입니다. 조금 더 빠른 속도로 답변해보세요.", wpm=wpm)
            elif wpm > WPM_FAST:
                await send_feedback("wpm_fast", "말이 빠릅니다. 조금만 천천히 말씀해 주세요.", wpm=wpm)

            if consecutive_filler_segments >= CONSECUTIVE_FILLER_THRESHOLD:
                await send_feedback("filler", "추임새가 많습니다. 의식적으로 줄여보세요.")

        except Exception as e:
            logger.error(f"STT 에러: {e}")
            if is_connected:
                try:
                    await send_ws({"status": "error", "message": str(e)})
                except Exception:
                    pass

    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                chunk = np.frombuffer(data, dtype=np.int16)
            except ValueError:
                logger.warning("잘못된 오디오 데이터 수신 (홀수 바이트)")
                continue
            now = time.time()

            if detect_voice(chunk):
                # 음성 감지 시 침묵 구간 종료 → 침묵 시간 리스트에 저장
                if silence_start is not None:
                    silence_sec = round(now - silence_start, 2)
                    if silence_sec > MIN_SILENCE_DURATION:
                        silence_values.append(silence_sec)
                    silence_start = None

                audio_chunks.append(chunk)

            else:
                if silence_start is None:
                    silence_start = now

                silence_sec = round(now - silence_start, 2)
                await send_ws({"status": "silence", "silence_sec": silence_sec})

                if silence_sec > SILENCE_ALERT_SEC:
                    await send_feedback("silence", "답변 중 침묵이 길어지고 있습니다.")

                if audio_chunks:
                    audio_buffer = np.concatenate(audio_chunks)
                    segment_sec = len(audio_buffer) / SAMPLE_RATE
                    await stt_queue.put((audio_buffer, segment_sec, now))
                    audio_chunks = []

    except WebSocketDisconnect:
        is_connected = False
    except Exception as e:
        is_connected = False
        logger.error(f"WebSocket 루프 에러: {e}")
    finally:
        try:
            # 마지막 침묵 구간 저장
            if silence_start is not None:
                silence_sec = round(time.time() - silence_start, 2)
                if silence_sec > MIN_SILENCE_DURATION:
                    silence_values.append(silence_sec)

            # 마지막 음성 청크 → STT 큐에 넣어 WPM/필러워드 통계 반영
            if audio_chunks:
                audio_buffer = np.concatenate(audio_chunks)
                segment_sec = len(audio_buffer) / SAMPLE_RATE
                await stt_queue.put((audio_buffer, segment_sec, time.time()))
        except Exception as e:
            logger.error(f"종료 전 메트릭 저장 에러: {e}")

        await stt_queue.put(None)
        try:
            await worker_task
        except Exception as e:
            logger.error(f"STT 워커 태스크 에러: {e}")

        try:
            avg_wpm = round(total_words / total_speech_sec * 60, 2) if total_speech_sec > 0 else 0.0
            avg_silence_duration = round(sum(silence_values) / len(silence_values), 2) if silence_values else 0.0
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    db.add(VoiceAnalysis(
                        session_id=session_id,
                        question_id=question_id,
                        avg_wpm=avg_wpm,
                        avg_silence_duration=avg_silence_duration,
                        filler_count=filler_count_total,
                    ))
                    if stt_parts:
                        full_text = " ".join(stt_parts)
                        await db.execute(
                            text("UPDATE interview_answers SET answer_text = :answer_text WHERE question_id = :question_id"),
                            {"answer_text": full_text, "question_id": int(question_id)},
                        )
            logger.info(f"[{session_id}:{question_id}] 음성 분석 결과 및 answer_text MySQL 저장 완료")
        except Exception as e:
            logger.error(f"음성 분석 결과 MySQL 저장 에러: {e}")


@router.websocket("/ws/stt/debate/{session_id}/{round_type}")
async def debate_stt_websocket(websocket: WebSocket, session_id: str, round_type: str):
    await websocket.accept()
    round_type = round_type.upper()
    ALLOWED_ROUNDS = {"OPENING", "REBUTTAL_1", "REBUTTAL_2", "CLOSING"}
    if not session_id.isdigit() or round_type not in ALLOWED_ROUNDS:
        await websocket.close(code=1008)
        return
    audio_chunks = []
    ws_lock = asyncio.Lock()
    stt_queue: asyncio.Queue = asyncio.Queue()

    speech_segments: deque = deque()
    consecutive_filler_segments = 0

    silence_start: float | None = None
    last_feedback: dict = {}
    is_connected = True

    total_words = 0
    total_speech_sec = 0.0
    silence_values: list = []
    filler_count_total = 0
    stt_parts: list = []

    async def send_ws(data: dict):
        async with ws_lock:
            await websocket.send_json(data)

    def can_feedback(fb_type: str) -> bool:
        now = time.time()
        if now - last_feedback.get(fb_type, 0) >= FEEDBACK_COOLDOWN_SEC:
            last_feedback[fb_type] = now
            return True
        return False

    async def send_feedback(fb_type: str, message: str, **extra):
        if can_feedback(fb_type):
            await send_ws({"status": "feedback", "type": fb_type, "message": message, **extra})

    async def stt_worker():
        while True:
            item = await stt_queue.get()
            if item is None:
                stt_queue.task_done()
                break
            audio_buffer, segment_sec, ts = item
            await process_stt(audio_buffer, segment_sec, ts)
            stt_queue.task_done()

    worker_task = asyncio.create_task(stt_worker())

    async def process_stt(audio_buffer: np.ndarray, segment_sec: float, ts: float):
        nonlocal consecutive_filler_segments, total_words, total_speech_sec, filler_count_total
        try:
            transcribed = await transcribe_audio(audio_buffer)
            if not transcribed:
                return

            stt_parts.append(transcribed)

            words = len([w for w in transcribed.split() if w])
            speech_segments.append((ts, words, segment_sec))
            cutoff = ts - WPM_WINDOW_SEC
            while speech_segments and speech_segments[0][0] < cutoff:
                speech_segments.popleft()

            window_words = sum(s[1] for s in speech_segments)
            window_sec = sum(s[2] for s in speech_segments)
            wpm = round(window_words / window_sec * 60, 1) if window_sec > 0 else 0.0

            if words > 0:
                total_words += words
                total_speech_sec += segment_sec

            filler_count = count_filler_words(transcribed)
            filler_count_total += filler_count
            if filler_count > 0:
                consecutive_filler_segments += 1
            else:
                consecutive_filler_segments = 0

            if not is_connected:
                return

            await send_ws({
                "status": "completed",
                "text": transcribed,
                "session_id": session_id,
                "round": round_type,
                "wpm": wpm,
                "filler_count": filler_count,
            })

            if 0 < wpm < WPM_SLOW:
                await send_feedback("wpm_slow", "말이 느린 편입니다. 조금 더 빠른 속도로 답변해보세요.", wpm=wpm)
            elif wpm > WPM_FAST:
                await send_feedback("wpm_fast", "말이 빠릅니다. 조금만 천천히 말씀해 주세요.", wpm=wpm)

            if consecutive_filler_segments >= CONSECUTIVE_FILLER_THRESHOLD:
                await send_feedback("filler", "추임새가 많습니다. 의식적으로 줄여보세요.")

        except Exception as e:
            logger.error(f"토론 STT 에러: {e}")
            if is_connected:
                try:
                    await send_ws({"status": "error", "message": str(e)})
                except Exception:
                    pass

    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                chunk = np.frombuffer(data, dtype=np.int16)
            except ValueError:
                logger.warning("잘못된 오디오 데이터 수신 (홀수 바이트)")
                continue
            now = time.time()

            if detect_voice(chunk):
                if silence_start is not None:
                    silence_sec = round(now - silence_start, 2)
                    if silence_sec > MIN_SILENCE_DURATION:
                        silence_values.append(silence_sec)
                    silence_start = None

                audio_chunks.append(chunk)

            else:
                if silence_start is None:
                    silence_start = now

                silence_sec = round(now - silence_start, 2)
                await send_ws({"status": "silence", "silence_sec": silence_sec})

                if silence_sec > SILENCE_ALERT_SEC:
                    await send_feedback("silence", "답변 중 침묵이 길어지고 있습니다.")

                if audio_chunks:
                    audio_buffer = np.concatenate(audio_chunks)
                    segment_sec = len(audio_buffer) / SAMPLE_RATE
                    await stt_queue.put((audio_buffer, segment_sec, now))
                    audio_chunks = []

    except WebSocketDisconnect:
        is_connected = False
    except Exception as e:
        is_connected = False
        logger.error(f"토론 WebSocket 루프 에러: {e}")
    finally:
        try:
            if silence_start is not None:
                silence_sec = round(time.time() - silence_start, 2)
                if silence_sec > MIN_SILENCE_DURATION:
                    silence_values.append(silence_sec)

            if audio_chunks:
                audio_buffer = np.concatenate(audio_chunks)
                segment_sec = len(audio_buffer) / SAMPLE_RATE
                await stt_queue.put((audio_buffer, segment_sec, time.time()))
        except Exception as e:
            logger.error(f"토론 종료 전 메트릭 저장 에러: {e}")

        await stt_queue.put(None)
        try:
            await worker_task
        except Exception as e:
            logger.error(f"토론 STT 워커 태스크 에러: {e}")

        try:
            avg_wpm = round(total_words / total_speech_sec * 60, 2) if total_speech_sec > 0 else 0.0
            avg_silence_duration = round(sum(silence_values) / len(silence_values), 2) if silence_values else 0.0
            async with AsyncSessionLocal() as db:
                async with db.begin():
                    db.add(VoiceAnalysis(
                        session_id=session_id,
                        question_id=round_type,
                        avg_wpm=avg_wpm,
                        avg_silence_duration=avg_silence_duration,
                        filler_count=filler_count_total,
                    ))
                    if stt_parts:
                        full_text = " ".join(stt_parts)
                        await db.execute(
                            text("UPDATE debate_sessions SET pending_stt = :pending_stt WHERE id = :session_id"),
                            {"pending_stt": full_text, "session_id": int(session_id)},
                        )
            logger.info(f"[debate {session_id}:{round_type}] 음성 분석 결과 및 pending_stt MySQL 저장 완료")
        except Exception as e:
            logger.error(f"토론 음성 분석 결과 MySQL 저장 에러: {e}")
