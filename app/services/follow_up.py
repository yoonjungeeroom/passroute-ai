import json
import logging

from openai import APIError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.schemas.follow_up import FollowUpRequest, FollowUpResponse

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = """\
당신은 개발자 채용 면접관입니다.
지원자의 답변을 분석하여 꼬리 질문이 필요한지 판단하고, 필요하다면 꼬리 질문을 생성하세요.

## 꼬리 질문을 생성해야 하는 경우
- 답변이 모호하거나 피상적이어서 구체적인 확인이 필요한 경우
- 기술적 깊이를 더 확인해야 하는 경우 (예: 원리, 트레이드오프, 대안)
- 실제 경험을 검증해야 하는 경우 (예: 구체적 사례, 수치, 결과)
- 답변에 논리적 허점이나 모순이 있는 경우

## 꼬리 질문을 생성하지 않아야 하는 경우
- 답변이 이미 충분히 구체적이고 깊이가 있는 경우
- 원래 질문이 단순 사실 확인(예/아니오)인 경우
- 추가 질문이 면접 흐름에 도움이 되지 않는 경우
- 답변과 무관한 방향으로 흘러갈 위험이 있는 경우

## 난이도별 기준
- low: 기본 개념 확인 수준. 답변이 핵심만 담고 있으면 꼬리 질문 불필요
- middle: 개념 + 적용 경험 확인. 경험이나 구체적 사례가 빠지면 꼬리 질문 생성
- high: 깊은 이해 + 트레이드오프 + 대안 제시까지 기대. 표면적 답변이면 반드시 꼬리 질문 생성

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트를 포함하지 마세요.
{
  "has_follow_up": true 또는 false,
  "follow_up_question": "꼬리 질문 텍스트" 또는 null,
  "reason": "판단 근거"
}\
"""

DIFFICULTY_LABELS = {
    "low": "하",
    "middle": "중",
    "high": "상",
}

INTERVIEW_TYPE_LABELS = {
    "technical": "기술 면접",
    "personality": "인성 면접",
}


def _build_user_message(request: FollowUpRequest) -> str:
    """OpenAI에 전달할 사용자 메시지를 구성한다."""
    interview_type = INTERVIEW_TYPE_LABELS[request.interview_type]
    difficulty = DIFFICULTY_LABELS[request.difficulty]

    parts = [
        f"[면접 정보] 유형: {interview_type} | 난이도: {difficulty}",
        "\n[대화 이력]",
    ]

    for i, turn in enumerate(request.conversation):
        label = "원래 질문" if i == 0 else f"꼬리 질문 {i}"
        parts.append(f"{label}: {turn.question}")
        parts.append(f"답변: {turn.answer}")

    return "\n".join(parts)


async def generate_follow_up(
    request: FollowUpRequest,
) -> FollowUpResponse:
    """꼬리 질문을 생성한다.

    OpenAI가 답변의 적합성을 판단하여 꼬리 질문 생성 여부를 결정한다.
    스킵 판단과 턴 제한은 Spring 서버에서 사전 처리한다.
    """
    user_message = _build_user_message(request)

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=512,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            timeout=30.0,
        )
    except APITimeoutError:
        logger.error("OpenAI API 타임아웃")
        return FollowUpResponse(
            has_follow_up=False,
            reason="AI 서비스 응답 시간 초과",
        )
    except APIError as e:
        logger.error("OpenAI API 오류: %s", e)
        return FollowUpResponse(
            has_follow_up=False,
            reason="AI 서비스 호출 실패",
        )

    raw_text = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw_text)
        return FollowUpResponse(
            has_follow_up=parsed["has_follow_up"],
            follow_up_question=parsed.get("follow_up_question"),
            reason=parsed.get("reason"),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.error("OpenAI 응답 파싱 실패: %s", raw_text)
        return FollowUpResponse(
            has_follow_up=False,
            reason="AI 응답 파싱 실패",
        )
