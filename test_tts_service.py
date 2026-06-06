import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import pytest
from moto import mock_aws

os.environ.setdefault("DATABASE_URL", "mysql+aiomysql://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("TTS_ENABLED", "true")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("AWS_S3_BUCKET_NAME", "test-bucket")
os.environ.setdefault(
    "GOOGLE_TTS_CREDENTIALS_JSON",
    json.dumps({
        "type": "service_account",
        "project_id": "test-project",
        "private_key_id": "key-id",
        "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA2a2rwplBQLF29amygykEMmYz0+Kcj3bKBp29D2rFDcMslMk\ndFRqKgMFAFrS9zxEkBUfQBZ2UJrRCDuDNnb+YJMiGgkE5M4LcZHB0Q==\n-----END RSA PRIVATE KEY-----\n",
        "client_email": "test@test-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }),
)


@pytest.fixture
def s3_bucket():
    with mock_aws():
        client = boto3.client("s3", region_name="ap-northeast-2")
        client.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ap-northeast-2"},
        )
        yield client


def _make_google_tts_service():
    """GoogleTtsService를 Google 인증 없이 생성."""
    from app.services.tts_service import GoogleTtsService
    mock_tts_client = AsyncMock()
    mock_s3 = MagicMock()
    svc = object.__new__(GoogleTtsService)
    svc._tts = mock_tts_client
    svc._s3 = mock_s3
    return svc


@pytest.mark.asyncio
async def test_noop_returns_none():
    from app.services.tts_service import NoopTtsService
    svc = NoopTtsService()
    result = await svc.synthesize("안녕하세요", "ko-KR-Neural2-A", "test_key")
    assert result is None


@pytest.mark.asyncio
async def test_google_tts_success_returns_s3_url(s3_bucket, monkeypatch):
    from google.cloud.texttospeech_v1.types import SynthesizeSpeechResponse
    from app.services.tts_service import GoogleTtsService
    from app.core.config import settings
    monkeypatch.setattr(settings, "AWS_S3_BUCKET_NAME", "test-bucket")

    mock_response = MagicMock(spec=SynthesizeSpeechResponse)
    mock_response.audio_content = b"fake-mp3-data"

    svc = object.__new__(GoogleTtsService)
    svc._tts = AsyncMock()
    svc._tts.synthesize_speech = AsyncMock(return_value=mock_response)
    svc._s3 = boto3.client("s3", region_name="ap-northeast-2")

    url = await svc.synthesize("안녕하세요", "ko-KR-Neural2-A", "interviewer_opening_abc123")

    assert url is not None
    assert "test-bucket" in url
    assert "interviewer_opening_abc123.mp3" in url


@pytest.mark.asyncio
async def test_google_tts_api_failure_returns_none():
    from app.services.tts_service import GoogleTtsService

    svc = object.__new__(GoogleTtsService)
    svc._tts = AsyncMock()
    svc._tts.synthesize_speech = AsyncMock(side_effect=Exception("API 오류"))
    svc._s3 = MagicMock()

    url = await svc.synthesize("테스트", "ko-KR-Neural2-A", "fail_key")
    assert url is None


@pytest.mark.asyncio
async def test_google_tts_s3_failure_returns_none():
    from google.cloud.texttospeech_v1.types import SynthesizeSpeechResponse
    from app.services.tts_service import GoogleTtsService

    mock_response = MagicMock(spec=SynthesizeSpeechResponse)
    mock_response.audio_content = b"audio"

    svc = object.__new__(GoogleTtsService)
    svc._tts = AsyncMock()
    svc._tts.synthesize_speech = AsyncMock(return_value=mock_response)
    svc._s3 = MagicMock()
    svc._s3.put_object = MagicMock(side_effect=Exception("S3 오류"))

    url = await svc.synthesize("테스트", "ko-KR-Neural2-A", "s3_fail_key")
    assert url is None


def test_speaker_for_interviewer():
    from app.services.tts_service import get_speaker_for_interviewer
    assert get_speaker_for_interviewer() == "ko-KR-Neural2-A"


def test_speaker_for_known_persona():
    from app.services.tts_service import get_speaker_for_persona
    assert get_speaker_for_persona("persona_01_stable") == "ko-KR-Neural2-B"
    assert get_speaker_for_persona("persona_02_aggressive") == "ko-KR-Neural2-C"
    assert get_speaker_for_persona("persona_03_creative") == "ko-KR-Neural2-D"
    assert get_speaker_for_persona("persona_04_veteran") == "ko-KR-Wavenet-B"
    assert get_speaker_for_persona("persona_05_nondev") == "ko-KR-Wavenet-A"


def test_speaker_for_unknown_persona_fallback():
    from app.services.tts_service import get_speaker_for_persona
    assert get_speaker_for_persona("persona_unknown") == "ko-KR-Neural2-A"


def test_speaker_for_interview_persona_known():
    from app.services.tts_service import get_speaker_for_interview_persona
    assert get_speaker_for_interview_persona("HR_MANAGER") == "ko-KR-Neural2-A"
    assert get_speaker_for_interview_persona("TEAM_LEAD") == "ko-KR-Neural2-C"
    assert get_speaker_for_interview_persona("EXECUTIVE") == "ko-KR-Wavenet-D"
    assert get_speaker_for_interview_persona("TECH_INTERVIEWER") == "ko-KR-Neural2-D"


def test_speaker_for_interview_persona_none_and_unknown_fallback():
    from app.services.tts_service import get_speaker_for_interview_persona
    assert get_speaker_for_interview_persona(None) == "ko-KR-Neural2-A"
    assert get_speaker_for_interview_persona("UNKNOWN") == "ko-KR-Neural2-A"


@pytest.mark.asyncio
async def test_google_tts_custom_prefix_in_url(s3_bucket, monkeypatch):
    from google.cloud.texttospeech_v1.types import SynthesizeSpeechResponse
    from app.services.tts_service import GoogleTtsService
    from app.core.config import settings
    monkeypatch.setattr(settings, "AWS_S3_BUCKET_NAME", "test-bucket")

    mock_response = MagicMock(spec=SynthesizeSpeechResponse)
    mock_response.audio_content = b"fake-mp3-data"

    svc = object.__new__(GoogleTtsService)
    svc._tts = AsyncMock()
    svc._tts.synthesize_speech = AsyncMock(return_value=mock_response)
    svc._s3 = boto3.client("s3", region_name="ap-northeast-2")

    url = await svc.synthesize(
        "면접 질문입니다",
        "ko-KR-Neural2-A",
        "interview_question_abc123",
        prefix="tts/interview",
    )

    assert url is not None
    assert "tts/interview/interview_question_abc123.mp3" in url
