import asyncio
from app.services.resume_parser import parse_resume
from app.services.resume_vector_store import store_resume, search_candidates, _get_collection


async def test():
    # 1단계: 파싱
    print("=== 1. 파싱 테스트 ===")
    with open("test_resume.pdf", "rb") as f:
        raw_text = parse_resume(f.read(), "test_resume.pdf")
    print(raw_text[:500])

    # 2단계: 벡터 저장
    print("\n=== 2. 벡터 저장 테스트 ===")
    store_resume("test_user_1", raw_text)
    print("저장 완료!")

    # 3단계: 실제로 들어갔는지 확인
    print("\n=== 3. 저장 확인 ===")
    collection = _get_collection()
    result = collection.get(where={"user_id": "test_user_1"})
    print(f"저장된 청크 수: {len(result['ids'])}")
    print(f"저장된 ID들: {result['ids']}")

    # 4단계: 검색
    print("\n=== 4. 검색 테스트 ===")
    results = search_candidates("백엔드 개발자")
    print(results)


asyncio.run(test())