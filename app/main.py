from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import FileResponse
#import chromadb
#from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.routers import stt

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    #app.state.chroma = chromadb.HttpClient(
    #    host=settings.CHROMADB_HOST,
    #    port=settings.CHROMADB_PORT,
    #)
    #app.state.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    yield
    # shutdown


app = FastAPI(
    title="PassRoute AI server",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(stt.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/test")
async def test_page():
    return FileResponse("test.html")