"""토론 주제 생성을 위한 크롤링 뉴스 조회.
"""
import logging

from app.services.embedder import get_embedder
from app.services.resume_vector_store import _get_client

logger = logging.getLogger(__name__)

NEWS_SOURCE = "naver_news"
NEWS_COLLECTION = "job_descriptions"


def query_news(
    query_text: str,
    company_name: str | None = None,
    n_results: int = 8,
) -> list[dict]:
    """job_descriptions 컬렉션에서 naver_news 데이터만 조회한다.

    company_name이 주어지면 해당 기업 뉴스로 필터링하고(현재 토론 흐름에선 미사용),
    없으면 query_text 유사도로 전체 뉴스에서 검색한다.
    각 결과는 {"title", "company_name", "url", "content"} 형태로 반환된다.
    컬렉션이 없거나 조회 실패 시 빈 리스트를 반환한다(폴백).
    """
    seed = (query_text or "").strip() or company_name or "최신 기술 동향"

    try:
        collection = _get_client().get_collection(name=NEWS_COLLECTION)
    except Exception as e:
        logger.warning("크롤링 뉴스 컬렉션(%s) 조회 실패: %s", NEWS_COLLECTION, e)
        return []

    conditions: list[dict] = [{"source": {"$eq": NEWS_SOURCE}}]
    if company_name:
        conditions.append({"company_name": {"$eq": company_name}})
    where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    # 쿼리 임베딩을 KR-SBERT로 직접 계산 (컬렉션 EF에 의존하지 않음)
    query_embedding = get_embedder().embed(seed)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )

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
