import io
import wave
import numpy as np
import webrtcvad
from openai import AsyncOpenAI
from app.core.config import settings

SAMPLE_RATE = 16000
FRAME_DURATION = 30

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def detect_voice(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
    vad = webrtcvad.Vad(3)
    audio_bytes = audio.tobytes()
    num_samples = int(sample_rate * FRAME_DURATION / 1000)
    frames = [audio_bytes[i:i + num_samples * 2] for i in range(0, len(audio_bytes), num_samples * 2)]

    count_speech = 0
    for frame in frames:
        if len(frame) < num_samples * 2:
            continue
        if vad.is_speech(frame, sample_rate):
            count_speech += 1
            if count_speech > 6:
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
    audio_bytes = numpy_to_wav_bytes(audio)
    data = ("audio.wav", audio_bytes, "audio/wav")
    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=data,
        language="ko",
        response_format="text"
    )
    return response