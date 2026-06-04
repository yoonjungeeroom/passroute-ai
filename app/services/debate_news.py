"""토론 주제 생성을 위한 크롤링 뉴스 조회.
"""
from app.services.resume_vector_store import _get_crawled_collection

NEWS_SOURCE = "naver_news"


def query_news( 
    query_text: str,
    company_name: str | None = None,
    n_results: int = 8,
) -> list[dict]:
    """job_descriptions 컬렉션에서 naver_news 데이터만 조회한다.

    company_name이 주어지면 해당 기업 뉴스로 필터링하고(현재 토론 흐름에선 미사용),
    없으면 query_text 유사도로 전체 뉴스에서 검색한다.
    각 결과는 {"title", "company_name", "url", "content"} 형태로 반환된다.
    """
    seed = (query_text or "").strip() or company_name or "최신 기술 동향"

    conditions: list[dict] = [{"source": {"$eq": NEWS_SOURCE}}]
    if company_name:
        conditions.append({"company_name": {"$eq": company_name}})
    where_filter = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    collection = _get_crawled_collection()
    results = collection.query(
        query_texts=[seed],
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
