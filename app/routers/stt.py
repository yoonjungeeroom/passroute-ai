import io
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.stt_service import detect_voice, transcribe_audio

router = APIRouter()

@router.websocket("/ws/stt/{session_id}/{question_id}")
async def stt_websocket(websocket: WebSocket, session_id: str, question_id: str):
    await websocket.accept()
    audio_buffer = np.array([], dtype=np.int16)

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.int16)

            if detect_voice(chunk):
                audio_buffer = np.concatenate((audio_buffer, chunk))
                await websocket.send_json({"status": "recording"})
            else:
                if len(audio_buffer) > 0:
                    text = await transcribe_audio(audio_buffer)
                    if text:
                        await websocket.send_json({
                            "status": "completed",
                            "text": text,
                            "session_id": session_id,
                            "question_id": question_id
                        })
                    audio_buffer = np.array([], dtype=np.int16)
                else:
                    await websocket.send_json({"status": "silence"})

    except WebSocketDisconnect:
        if len(audio_buffer) > 0:
            text = await transcribe_audio(audio_buffer)
            if text:
                print(f"[{session_id}:{question_id}] 최종 STT: {text}")