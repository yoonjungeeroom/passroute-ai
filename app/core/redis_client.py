import redis.asyncio as aioredis
from app.core.config import settings

_redis: aioredis.Redis | None = None

STT_TRANSCRIPT_TTL = 3600  # 1시간


async def init_redis():
    global _redis
    _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)


async def close_redis():
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialized")
    return _redis


async def append_stt_transcript(session_id: str, question_id: str, text: str) -> None:
    r = get_redis()
    key = f"stt:{session_id}:{question_id}"
    await r.rpush(key, text)
    await r.expire(key, STT_TRANSCRIPT_TTL)
