from chromadb import HttpClient
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.embedder import get_embedder


class OnnxEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self._embedder = get_embedder()

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


def store_resume(user_id: str, raw_text: str) -> None:  # structured 제거
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


_crawled_collection = None


def _get_crawled_collection():
    """job_descriptions 컬렉션을 캐싱하여 반환한다."""
    global _crawled_collection
    if _crawled_collection is None:
        _crawled_collection = _get_client().get_or_create_collection(
            name="job_descriptions",
            embedding_function=_get_embedding_fn(),
            metadata={"hnsw:space": "cosine"},
        )
    return _crawled_collection


def query_crawled_data(search_query: str, sources: list[str], n_results: int = 3) -> str:
    """ChromaDB job_descriptions 컬렉션에서 크롤링 데이터를 검색한다."""
    if not search_query or not search_query.strip():
        return ""

    collection = _get_crawled_collection()

    where_filter = (
        {"source": {"$in": sources}} if len(sources) > 1 else {"source": sources[0]}
    )

    results = collection.query(
        query_texts=[search_query],
        n_results=n_results,
        where=where_filter,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return ""

    source_labels = {
        "tech_blog": "기술블로그",
        "jobkorea": "채용공고",
        "naver_news": "뉴스",
    }
    parts = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "")
        title = meta.get("title", "")
        label = source_labels.get(source, source)
        parts.append(f"- [{label}] {title}: {doc[:300]}")

    return "\n".join(parts)


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
                "matched_text": doc[:200],
            })

    return candidates