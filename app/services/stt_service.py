import io
import json
import logging
import wave
import numpy as np
import httpx
import webrtcvad
from app.core.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_DURATION = 30

vad = webrtcvad.Vad(3)
_http_client = httpx.AsyncClient(timeout=30.0)


async def close_http_client():
    await _http_client.aclose()


def detect_voice(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
    audio_bytes = audio.tobytes()
    num_samples = int(sample_rate * FRAME_DURATION / 1000)
    frames = [audio_bytes[i:i + num_samples * 2] for i in range(0, len(audio_bytes), num_samples * 2)]

    count_speech = 0
    for frame in frames:
        if len(frame) < num_samples * 2:
            continue
        if vad.is_speech(frame, sample_rate):
            count_speech += 1
            if count_speech > 1:
                return True
    return False


def numpy_to_wav_bytes(audio: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buffer.getvalue()


async def transcribe_audio(audio: np.ndarray) -> str:
    if len(audio) < 500:
        return ""

    wav_bytes = numpy_to_wav_bytes(audio)
    params = json.dumps({
        "language": "ko-KR",
        "completion": "sync",
        "diarization": {"enable": False},
    })

    response = await _http_client.post(
        f"{settings.CLOVA_INVOKE_URL}/recognizer/upload",
        headers={"X-CLOVASPEECH-API-KEY": settings.CLOVA_SECRET_KEY},
        files={
            "media": ("audio.wav", wav_bytes, "audio/wav"),
            "params": (None, params, "application/json"),
        },
    )
    if response.status_code != 200:
        logger.error(f"Clova 에러 응답: {response.text}")
        response.raise_for_status()
    return response.json().get("text", "")
