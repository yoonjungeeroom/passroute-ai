"""꼬리 질문 생성 테스트 스크립트."""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from app.schemas.follow_up import FollowUpRequest, QATurn
from app.services.follow_up import generate_follow_up


async def main():
    # 테스트 케이스 1: 피상적인 답변 (꼬리 질문 생성 예상)
    print("=" * 60)
    print("[테스트 1] 피상적인 답변 - difficulty: middle")
    print("=" * 60)
    request1 = FollowUpRequest(
        interview_type="technical",
        difficulty="middle",
        conversation=[
            QATurn(
                question="RESTful API 설계 원칙에 대해 설명해주세요.",
                answer="HTTP 메서드를 사용하는 것입니다.",
            )
        ],
    )
    result1 = await generate_follow_up(request1)
    print(f"has_follow_up: {result1.has_follow_up}")
    print(f"follow_up_question: {result1.follow_up_question}")
    print(f"reason: {result1.reason}")

    # 테스트 케이스 2: 구체적인 답변 (꼬리 질문 미생성 예상)
    print("\n" + "=" * 60)
    print("[테스트 2] 구체적인 답변 - difficulty: middle")
    print("=" * 60)
    request2 = FollowUpRequest(
        interview_type="technical",
        difficulty="middle",
        conversation=[
            QATurn(
                question="RESTful API 설계 원칙에 대해 설명해주세요.",
                answer="RESTful API는 리소스를 URI로 표현하고 HTTP 메서드로 행위를 구분합니다. "
                "GET은 조회, POST는 생성, PUT은 전체 수정, PATCH는 부분 수정, DELETE는 삭제에 사용합니다. "
                "실제 프로젝트에서 /api/users/{id}/orders 형태로 리소스 간 관계를 표현했고, "
                "상태 코드도 200, 201, 404, 409 등을 구분하여 클라이언트가 응답을 명확히 처리할 수 있도록 했습니다.",
            )
        ],
    )
    result2 = await generate_follow_up(request2)
    print(f"has_follow_up: {result2.has_follow_up}")
    print(f"follow_up_question: {result2.follow_up_question}")
    print(f"reason: {result2.reason}")

    # 테스트 케이스 3: 높은 난이도 + 표면적 답변 (꼬리 질문 생성 예상)
    print("\n" + "=" * 60)
    print("[테스트 3] 표면적 답변 - difficulty: high")
    print("=" * 60)
    request3 = FollowUpRequest(
        interview_type="technical",
        difficulty="high",
        conversation=[
            QATurn(
                question="데이터베이스 인덱스의 동작 원리와 트레이드오프에 대해 설명해주세요.",
                answer="인덱스를 걸면 조회가 빨라집니다. B-Tree 구조를 사용합니다.",
            )
        ],
    )
    result3 = await generate_follow_up(request3)
    print(f"has_follow_up: {result3.has_follow_up}")
    print(f"follow_up_question: {result3.follow_up_question}")
    print(f"reason: {result3.reason}")


asyncio.run(main())
