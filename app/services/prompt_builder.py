import asyncio
import json
import logging
from typing import Optional

from openai import APIError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.schemas.prompt_builder import (
    GeneratedQuestion,
    QuestionGenerateRequest,
    QuestionGenerateResponse,
)
from app.services.resume_vector_store import query_crawled_data

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# ──────────────────────────────────────────────
# 페르소나 기본 템플릿
# ──────────────────────────────────────────────

_PERSONA_TEMPLATES: dict[str, str] = {
    "HR_MANAGER": """\
당신은 HR 담당자로서 면접을 진행합니다.
집중 영역: 가치관, 문화핏, 커뮤니케이션 능력
말투: 친근하고 따뜻하게
질문 스타일: "~하신 경험이 있으신가요?", "어떻게 생각하세요?" 형식을 자주 활용하세요.
주요 확인 사항:
- 지원 동기와 회사에 대한 이해도
- 팀 내 갈등 상황 대처 경험
- 본인의 강점/약점 인식
- 조직 문화 적합성""",

    "TEAM_LEAD": """\
당신은 현업 팀장으로서 면접을 진행합니다.
집중 영역: 협업 방식, 문제 해결 태도, 성장 가능성
말투: 실무적이고 직접적으로
질문 스타일: "그때 어떻게 해결하셨나요?", "팀원들과 어떻게 조율하셨나요?" 형식을 자주 활용하세요.
주요 확인 사항:
- 프로젝트에서 맡은 구체적인 역할
- 팀원과의 의견 충돌 경험과 해결 방식
- 실패 경험과 극복 방법
- 업무 우선순위 설정 방식""",

    "EXECUTIVE": """\
당신은 임원으로서 면접을 진행합니다.
집중 영역: 비전, 큰 그림, 조직 기여도
말투: 격식 있고 날카롭게
질문 스타일: "5년 후 어떤 모습이 되고 싶나요?", "우리 회사에 어떤 기여를 할 수 있나요?" 형식을 활용하세요.
주요 확인 사항:
- 커리어 목표와 비전
- 업계 트렌드에 대한 견해
- 회사 성장에 기여할 수 있는 부분
- 리더십 경험""",

    "TECH_INTERVIEWER": """\
당신은 기술 면접관으로서 면접을 진행합니다.
집중 영역: 기술 스택, 설계 판단력, 트러블슈팅
말투: 논리적이고 냉정하게
질문 스타일: "왜 이 기술을 선택했나요?", "다른 방법은 없었나요?" 형식을 자주 활용하세요.
주요 확인 사항:
- 사용한 기술 스택 선택 이유와 트레이드오프
- 시스템 설계 경험
- 트러블슈팅 경험과 문제 해결 과정
- 코드 품질, 성능 최적화 경험""",
}


# ──────────────────────────────────────────────
# 슬라이더 → 지침 변환
# ──────────────────────────────────────────────

def _pressure_guideline(level: int) -> str:
    if level <= 3:
        return (
            "[압박 강도: 낮음]\n"
            "지원자가 편안하게 답변할 수 있도록 유도하세요.\n"
            "답변이 부족해도 부드럽게 재질문하세요.\n"
            "긍정적인 리액션을 가끔 섞으세요."
        )
    if level <= 6:
        return (
            "[압박 강도: 중간]\n"
            "답변의 핵심을 파고드는 질문을 하세요.\n"
            "논리적 허점이 있으면 재질문하세요.\n"
            "감정적 표현 없이 중립적 톤을 유지하세요."
        )
    return (
        "[압박 강도: 높음]\n"
        "답변의 논리적 허점을 즉시 지적하세요.\n"
        "반박하거나 재질문을 빠르게 던지세요.\n"
        "침묵으로 압박하거나 예상치 못한 질문을 던지세요."
    )


def _difficulty_guideline(difficulty: str) -> str:
    mapping = {
        "EASY": (
            "[질문 난이도: 기초]\n"
            "기본적인 경험과 생각을 묻는 질문을 생성하세요.\n"
            "구체적인 기술 깊이보다 전반적인 이해도를 확인하세요."
        ),
        "NORMAL": (
            "[질문 난이도: 중간]\n"
            "경험의 구체적인 내용과 이유를 묻는 질문을 생성하세요.\n"
            "답변의 논리적 흐름을 확인하세요."
        ),
        "HARD": (
            "[질문 난이도: 심화]\n"
            "깊이 있는 전문성을 요구하는 질문을 생성하세요.\n"
            "트레이드오프, 설계 판단, 한계점을 파고드세요."
        ),
    }
    return mapping[difficulty]


def _followup_guideline(count: int) -> str:
    if count == 0:
        return "[꼬리질문] 꼬리질문 없이 기본 질문만 생성하세요."
    if count <= 2:
        return f"[꼬리질문] 각 질문에 꼬리질문 {count}개를 함께 생성하세요."
    return (
        f"[꼬리질문] 각 질문에 꼬리질문 {count}개를 함께 생성하세요.\n"
        "꼬리질문은 답변의 빈틈을 깊이 파고드는 방향으로 생성하세요."
    )


def _interview_type_guideline(interview_type: str) -> str:
    if interview_type == "PERSONALITY":
        return (
            "[면접 유형] 인성 면접입니다.\n"
            "가치관, 태도, 협업 경험, 성격 등을 묻는 질문을 생성하세요."
        )
    return (
        "[면접 유형] 기술 면접입니다.\n"
        "기술 스택, 설계 경험, 문제 해결 능력을 검증하는 질문을 생성하세요."
    )


def _format_guideline(interview_format: str) -> str:
    if interview_format == "ONE_ON_ONE":
        return (
            "[면접 방식] 1:1 면접입니다.\n"
            "지원자 개인의 경험과 역량을 직접 질문하는 방식으로 진행하세요."
        )
    return (
        "[면접 방식] 토론 면접입니다.\n"
        "주어진 주제에 대해 찬성/반대 입장을 지정하고, "
        "각 입장에서 논리적 근거를 제시하도록 유도하는 질문을 생성하세요."
    )


_SOURCE_MAP: dict[str, dict[str, list[str]]] = {
    "DEBATE": {"sources": ["naver_news"], "label": "참고 뉴스 데이터"},
    "TECHNICAL": {"sources": ["tech_blog", "jobkorea"], "label": "참고 업계 자료 (기술블로그/채용공고)"},
    "PERSONALITY": {"sources": ["jobkorea"], "label": "참고 채용공고 데이터"},
}


def _resolve_crawled_sources(interview_type: str, interview_format: str) -> tuple[list[str], str]:
    """면접 유형/형식에 따라 검색할 크롤링 소스와 라벨을 반환한다."""
    if interview_format == "DEBATE":
        config = _SOURCE_MAP["DEBATE"]
    else:
        config = _SOURCE_MAP.get(interview_type, _SOURCE_MAP["TECHNICAL"])
    return config["sources"], config["label"]


# ──────────────────────────────────────────────
# 핵심 함수: 프롬프트 조합
# ──────────────────────────────────────────────

def build_prompt(
    persona: str,
    pressure_level: int,
    difficulty: str,
    followup_count: int,
    interview_type: str,
    interview_format: str,
    cover_letter: str,
    resume: Optional[str] = None,
    portfolio: Optional[str] = None,
    crawled_context: Optional[str] = None,
    crawled_label: str = "참고 업계 자료",
    question_count: int = 5,
) -> str:
    followup_field = (
        f'"followup_questions": ["꼬리질문1", ..., "꼬리질문{followup_count}"]'
        if followup_count > 0
        else '"followup_questions": []'
    )

    generation_instruction = (
        f"[생성 지시]\n"
        f"위 정보를 바탕으로 면접 질문 {question_count}개를 아래 JSON 형식으로 생성하세요.\n"
        f"다른 텍스트 없이 순수 JSON만 응답하세요.\n\n"
        f'{{\n'
        f'  "questions": [\n'
        f'    {{\n'
        f'      "question": "질문 내용",\n'
        f'      {followup_field},\n'
        f'      "intent": "이 질문의 의도"\n'
        f'    }}\n'
        f'  ]\n'
        f'}}'
    )

    sections = [
        _PERSONA_TEMPLATES[persona],
        _pressure_guideline(pressure_level),
        _difficulty_guideline(difficulty),
        _followup_guideline(followup_count),
        _interview_type_guideline(interview_type),
        _format_guideline(interview_format),
        f"[자기소개서]\n{cover_letter}",
    ]

    if resume:
        sections.append(f"[이력서]\n{resume}")

    if portfolio:
        sections.append(f"[포트폴리오]\n{portfolio}")

    if crawled_context:
        sections.append(f"[{crawled_label}]\n{crawled_context}")

    sections.append(generation_instruction)

    return "\n\n".join(sections)


# ──────────────────────────────────────────────
# OpenAI 호출 + 응답 파싱
# ──────────────────────────────────────────────

async def generate_questions(
    request: QuestionGenerateRequest,
) -> QuestionGenerateResponse:
    sources, crawled_label = _resolve_crawled_sources(
        request.interview_type, request.interview_format,
    )

    search_query = request.cover_letter[:500].strip()
    if request.resume:
        search_query = (search_query + " " + request.resume[:300]).strip()

    try:
        crawled_context = await asyncio.to_thread(
            query_crawled_data, search_query, sources,
        )
    except Exception as e:
        logger.warning("ChromaDB 크롤링 데이터 검색 실패: %s", e)
        crawled_context = ""

    system_prompt = build_prompt(
        persona=request.persona,
        pressure_level=request.pressure_level,
        difficulty=request.difficulty,
        followup_count=request.followup_count,
        interview_type=request.interview_type,
        interview_format=request.interview_format,
        cover_letter=request.cover_letter,
        resume=request.resume,
        portfolio=request.portfolio,
        crawled_context=crawled_context,
        crawled_label=crawled_label,
        question_count=request.question_count,
    )

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=4096,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "면접 질문을 생성해 주세요."},
            ],
            timeout=60.0,
        )
    except APITimeoutError:
        logger.error("OpenAI API 타임아웃 (질문 생성)")
        raise
    except APIError as e:
        logger.error("OpenAI API 오류 (질문 생성): %s", e)
        raise

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
        questions = [
            GeneratedQuestion(
                question=q["question"],
                followup_questions=q.get("followup_questions", []),
                intent=q.get("intent"),
            )
            for q in parsed["questions"]
        ]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error("OpenAI 응답 파싱 실패: %s | raw=%s", e, raw[:300])
        raise ValueError(f"AI 응답 파싱 실패: {e}") from e

    return QuestionGenerateResponse(persona=request.persona, questions=questions)
