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

_PERSONA_SPEAKERS: dict[str, str] = {}
for _pair in settings.DEBATE_TTS_PERSONA_SPEAKERS.split(","):
    if ":" in _pair:
        _k, _v = _pair.split(":", 1)
        _PERSONA_SPEAKERS[_k.strip()] = _v.strip()


class TtsService(ABC):
    @abstractmethod
    async def synthesize(
        self, text: str, speaker: str, file_key: str, cache: bool = False
    ) -> str | None:
        """음성 합성 후 S3 URL 반환. 실패 시 None.

        cache=True면 같은 file_key의 오디오가 이미 S3에 있을 때 재생성하지 않고
        기존 URL을 반환한다. (고정 문구인 진행 멘트용)
        """


class NoopTtsService(TtsService):
    async def synthesize(
        self, text: str, speaker: str, file_key: str, cache: bool = False
    ) -> str | None:
        return None


class GoogleTtsService(TtsService):
    def __init__(self) -> None:
        if not settings.GOOGLE_TTS_CREDENTIALS_JSON:
            raise ValueError("GOOGLE_TTS_CREDENTIALS_JSON이 비어 있습니다. TTS_ENABLED=true일 때는 서비스 계정 JSON이 필요합니다.")
        try:
            credentials_info = json.loads(settings.GOOGLE_TTS_CREDENTIALS_JSON)
        except json.JSONDecodeError as e:
            raise ValueError(f"GOOGLE_TTS_CREDENTIALS_JSON 파싱 실패: {e}")
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._tts = texttospeech.TextToSpeechAsyncClient(credentials=credentials)
        self._s3 = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    async def synthesize(
        self, text: str, speaker: str, file_key: str, cache: bool = False
    ) -> str | None:
        if cache:
            loop = asyncio.get_running_loop()
            if await loop.run_in_executor(None, self._s3_object_exists, file_key):
                return self._object_url(file_key)
        audio = await self._call_google(text, speaker)
        if audio is None:
            return None
        return await self._upload_s3(audio, file_key)

    def _object_url(self, file_key: str) -> str:
        bucket = settings.AWS_S3_BUCKET_NAME
        key = f"{settings.TTS_S3_PREFIX}/{file_key}.mp3"
        return f"https://{bucket}.s3.{settings.AWS_REGION}.amazonaws.com/{key}"

    def _s3_object_exists(self, file_key: str) -> bool:
        key = f"{settings.TTS_S3_PREFIX}/{file_key}.mp3"
        try:
            self._s3.head_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=key)
            return True
        except Exception:
            return False

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
        loop = asyncio.get_running_loop()
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
            return self._object_url(file_key)
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
