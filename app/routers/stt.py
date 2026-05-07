import logging
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.stt_service import detect_voice, transcribe_audio
from app.core.redis_client import append_stt_transcript

logger = logging.getLogger(__name__)
router = APIRouter()

@router.websocket("/ws/stt/{session_id}/{question_id}")
async def stt_websocket(websocket: WebSocket, session_id: str, question_id: str):
    await websocket.accept()
    audio_chunks = []

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16)

            if detect_voice(chunk):
                audio_chunks.append(chunk)
                await websocket.send_json({"status": "recording"})
            else:
                if audio_chunks:
                    try:
                        audio_buffer = np.concatenate(audio_chunks)
                        text = await transcribe_audio(audio_buffer)
                        if text:
                            await append_stt_transcript(session_id, question_id, text)
                            await websocket.send_json({
                                "status": "completed",
                                "text": text,
                                "session_id": session_id,
                                "question_id": question_id
                            })
                    except Exception as e:
                        logger.error(f"STT 에러: {e}")
                        await websocket.send_json({"status": "error", "message": str(e)})
                    audio_chunks = []
                else:
                    await websocket.send_json({"status": "silence"})

    except WebSocketDisconnect:
        if audio_chunks:
            try:
                text = await transcribe_audio(np.concatenate(audio_chunks))
                if text:
                    await append_stt_transcript(session_id, question_id, text)
                    logger.info(f"[{session_id}:{question_id}] 최종 STT: {text}")
            except Exception as e:
                logger.error(f"최종 STT 에러: {e}")