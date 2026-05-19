from chromadb import HttpClient
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.embedder import OnnxEmbedder


class OnnxEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self._embedder = OnnxEmbedder()

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embedder.embed(text) for text in input]


_client = None
_embedding_fn = None


def _get_client():
    global _client
    if _client is None:
        _client = HttpClient(
            host=settings.CHROMADB_HOST,
            port=settings.CHROMADB_PORT,
        )
    return _client


def _get_embedding_fn():
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = OnnxEmbeddingFunction()
    return _embedding_fn


def _get_collection():
    return _get_client().get_or_create_collection(
        name="resumes",
        embedding_function=_get_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


def store_resume(user_id: str, raw_text: str, structured: dict) -> None:
    collection = _get_collection()

    existing = collection.get(where={"user_id": user_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    collection.add(
        documents=chunks,
        metadatas=[{"user_id": user_id, "type": "chunk"} for _ in chunks],
        ids=[f"{user_id}_chunk_{i}" for i in range(len(chunks))],
    )

    if structured.get("summary"):
        collection.add(
            documents=[structured["summary"]],
            metadatas=[{
                "user_id": user_id,
                "type": "summary",
                "skills": ",".join(structured.get("skills", [])),
            }],
            ids=[f"{user_id}_summary"],
        )


def search_candidates(query: str, top_k: int = 5) -> list[dict]:
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=top_k)

    seen = set()
    candidates = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        uid = meta.get("user_id")
        if uid not in seen:
            seen.add(uid)
            candidates.append({
                "user_id": uid,
                "score": round(1 - distance, 4),
                "skills": meta.get("skills", "").split(","),
                "matched_text": doc[:200],
            })

    return candidates