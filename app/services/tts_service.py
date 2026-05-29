import asyncio
import json
import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import boto3
from google.cloud import texttospeech
from google.oauth2 import service_account

from app.core.config import settings

logger = logging.getLogger(__name__)

_PERSONA_SPEAKERS: dict[str, str] = dict(
    pair.split(":")
    for pair in settings.DEBATE_TTS_PERSONA_SPEAKERS.split(",")
    if ":" in pair
)


class TtsService(ABC):
    @abstractmethod
    async def synthesize(self, text: str, speaker: str, file_key: str) -> str | None:
        """음성 합성 후 S3 URL 반환. 실패 시 None."""


class NoopTtsService(TtsService):
    async def synthesize(self, text: str, speaker: str, file_key: str) -> str | None:
        return None


class GoogleTtsService(TtsService):
    def __init__(self) -> None:
        credentials = service_account.Credentials.from_service_account_info(
            json.loads(settings.GOOGLE_TTS_CREDENTIALS_JSON),
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._tts = texttospeech.TextToSpeechAsyncClient(credentials=credentials)
        self._s3 = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    async def synthesize(self, text: str, speaker: str, file_key: str) -> str | None:
        audio = await self._call_google(text, speaker)
        if audio is None:
            return None
        return await self._upload_s3(audio, file_key)

    async def _call_google(self, text: str, speaker: str) -> bytes | None:
        try:
            response = await self._tts.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=texttospeech.VoiceSelectionParams(
                    language_code="ko-KR",
                    name=speaker,
                ),
                audio_config=texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                ),
            )
            return response.audio_content
        except Exception as e:
            logger.error("Google TTS 호출 실패: %s", e)
            return None

    async def _upload_s3(self, audio: bytes, file_key: str) -> str | None:
        bucket = settings.AWS_S3_BUCKET_NAME
        key = f"{settings.TTS_S3_PREFIX}/{file_key}.mp3"
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=audio,
                    ContentType="audio/mpeg",
                ),
            )
            return f"https://{bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"
        except Exception as e:
            logger.error("S3 업로드 실패: %s", e)
            return None


@lru_cache(maxsize=1)
def get_tts_service() -> TtsService:
    return GoogleTtsService() if settings.TTS_ENABLED else NoopTtsService()


def get_speaker_for_interviewer() -> str:
    return settings.DEBATE_TTS_INTERVIEWER_SPEAKER


def get_speaker_for_persona(persona_id: str) -> str:
    return _PERSONA_SPEAKERS.get(persona_id, "ko-KR-Neural2-A")
