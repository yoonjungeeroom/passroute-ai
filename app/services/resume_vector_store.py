from chromadb import PersistentClient
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.services.embedder import OnnxEmbedder


class OnnxEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB는 자체 EmbeddingFunction 인터페이스를 요구함.
    기존 OnnxEmbedder를 그 인터페이스에 맞게 래핑.
    """
    def __init__(self):
        self._embedder = OnnxEmbedder()

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embedder.embed(text) for text in input]


_client = None
_embedding_fn = None


def _get_client():
    global _client
    if _client is None:
        _client = PersistentClient(path="./chroma_db")
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
        # cosine 쓰는 이유: OnnxEmbedder가 L2 정규화를 하기 때문에
        # cosine similarity가 가장 정확한 유사도 측정 방식임
    )


def store_resume(user_id: str, raw_text: str, structured: dict) -> None:
    collection = _get_collection()

    # 재업로드 시 기존 데이터 삭제
    existing = collection.get(where={"user_id": user_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # 1) 원문 청크 저장 (의미 검색용)
    # ex) "React 경험 있는 사람 찾아줘" 같은 쿼리 대응
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(raw_text)

    collection.add(
        documents=chunks,
        metadatas=[{"user_id": user_id, "type": "chunk"} for _ in chunks],
        ids=[f"{user_id}_chunk_{i}" for i in range(len(chunks))],
    )

    # 2) summary 저장 (전체 프로필 매칭용)
    # ex) 공고 전체랑 후보자 전체 프로필 매칭할 때
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

    # 같은 user_id 중복 제거 (청크가 여러 개라 같은 사람이 여러 번 나올 수 있음)
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
                "score": round(1 - distance, 4),  # 거리 → 유사도 변환
                "skills": meta.get("skills", "").split(","),
                "matched_text": doc[:200],
            })

    return candidates