"""토론 주제 생성을 위한 크롤링 뉴스 조회.

크롤링 Consumer는 job_descriptions 컬렉션을 '사전 계산된 임베딩 + default EF'로 적재한다.
get_or_create_collection에 다른 EF(OnnxEmbeddingFunction)를 넘기면 최신 ChromaDB가
EF 충돌(ValueError)을 던지고, EF를 빼고 query_texts로 조회하면 컬렉션의 default EF
(다른 모델/차원)로 쿼리가 임베딩돼 KR-SBERT 768차원 저장 벡터와 불일치한다.
→ 컬렉션엔 EF를 붙이지 않고 get_collection으로 가져오고, 쿼리 임베딩만 동일한
   KR-SBERT 임베더로 직접 계산해 query_embeddings로 전달한다.
"""
import logging

import chromadb

from app.core.config import settings
from app.services.embedder import get_embedder

logger = logging.getLogger(__name__)

NEWS_SOURCE = "naver_news"
NEWS_COLLECTION = "job_descriptions"

_client = None


def _get_client():
    """ChromaDB HttpClient를 1회 생성해 재사용한다."""
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.CHROMADB_HOST,
            port=settings.CHROMADB_PORT,
        )
    return _client


def query_news(
    query_text: str,
    company_name: str | None = None,
    n_results: int = 8,
) -> list[dict]:
    """job_descriptions 컬렉션에서 naver_news 데이터만 조회한다.

    company_name이 주어지면 해당 기업 뉴스로 필터링하고(현재 토론 흐름에선 미사용),
    없으면 query_text 유사도로 전체 뉴스에서 검색한다.
    각 결과는 {"title", "company_name", "url", "content"} 형태로 반환된다.
    컬렉션이 없거나 조회/쿼리 실패 시 빈 리스트를 반환한다(폴백).
    """
    seed = (query_text or "").strip() or company_name or "최신 기술 동향"

    conditions: list[dict] = [{"source": {"$eq": NEWS_SOURCE}}]
    if company_name:
        conditions.append({"company_name": {"$eq": company_name}})
    where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    try:
        collection = _get_client().get_collection(name=NEWS_COLLECTION)
        # 쿼리 임베딩을 KR-SBERT로 직접 계산 (컬렉션 EF에 의존하지 않음)
        query_embedding = get_embedder().embed(seed)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
        )
    except Exception as e:
        logger.warning("크롤링 뉴스 컬렉션(%s) 조회 실패: %s", NEWS_COLLECTION, e)
        return []

    raw_documents = results.get("documents")
    documents = raw_documents[0] if raw_documents else []
    raw_metadatas = results.get("metadatas")
    metadatas = raw_metadatas[0] if raw_metadatas else []

    items: list[dict] = []
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        items.append({
            "title": meta.get("title", ""),
            "company_name": meta.get("company_name", ""),
            "url": meta.get("url", ""),
            "content": doc,
        })
    return items
