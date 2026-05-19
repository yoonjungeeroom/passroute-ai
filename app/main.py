from contextlib import asynccontextmanager
from fastapi import FastAPI
import chromadb
from app.core.config import settings
from app.routers.follow_up import router as follow_up_router
from app.services.embedder import OnnxEmbedder
from app.core.redis_client import init_redis, close_redis
from app.routers import stt
from app.routers.resume import router as resume_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    app.state.chroma = chromadb.HttpClient(
        host=settings.CHROMADB_HOST,
        port=settings.CHROMADB_PORT,
    )
    app.state.embedder = OnnxEmbedder()
    yield
    await close_redis()


app = FastAPI(
    title="PassRoute AI server",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(stt.router)

app.include_router(follow_up_router)

app.include_router(resume_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/test")
async def test_page():
    from fastapi.responses import FileResponse
    return FileResponse("test.html")