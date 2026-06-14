import asyncio
import json
import logging
import uuid
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import APIError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.schemas.follow_up import FollowUpRequest, FollowUpResponse
from app.services.resume_vector_store import _get_collection, query_crawled_data
from app.services.tts_service import (
    get_speaker_for_interview_persona,
    get_tts_service,
)

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

# ──────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────

ANALYSIS_PROMPT = """\
당신은 개발자 면접 답변 분석 전문가입니다.
지원자의 최신 답변(대화 이력의 마지막 답변)을 중점 분석하되, 이전 대화 맥락도 함께 고려하세요.

## 분석 기준

### 1. 답변 품질 평가
- 구체성: 추상적 설명인가, 실제 경험·수치·사례가 포함되었는가
- 기술적 깊이: 표면적 언급인가, 동작 원리·내부 구조까지 이해하고 있는가
- 논리적 완성도: 주장에 대한 근거가 있는가, 논리적 허점이 없는가

### 2. 키워드 추출 (중요)
답변에서 언급된 기술·도구·개념을 두 분류로 나누어 추출하세요:
- **tech_keywords**: 답변에서 직접 언급한 구체적 기술/도구/프레임워크 (예: Redis, Kafka, Docker, JPA, Spring Security)
- **concept_keywords**: 답변에서 언급한 개념/패턴/방법론 (예: 캐싱 전략, 이벤트 드리븐, CI/CD, TDD)

### 3. 심화 가능 포인트 식별
답변에서 언급했지만 깊이가 부족한 부분을 구체적으로 식별하세요:
- 기술을 언급했지만 왜 그 기술을 선택했는지 설명하지 않은 경우
- 결과를 말했지만 과정이나 트레이드오프를 설명하지 않은 경우
- 개념을 아는 것처럼 말했지만 실제 적용 경험이 드러나지 않은 경우

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요.
{
  "quality": "sufficient" 또는 "insufficient" 또는 "partial",
  "tech_keywords": ["Redis", "Docker", ...],
  "concept_keywords": ["캐싱 전략", "이벤트 드리븐", ...],
  "deep_dive_targets": [
    {"keyword": "언급된 기술/개념", "gap": "부족한 부분 설명"}
  ],
  "weak_points": ["부족한 부분1", "부족한 부분2", ...],
  "summary": "답변 분석 요약 (1~2문장)"
}\
"""

_BASE_SYSTEM_PROMPT = """\
당신은 실무 경험이 풍부한 시니어 개발자 면접관입니다.
지원자의 답변에서 언급된 구체적인 기술·개념을 포착하여, 실제 이해도와 실무 역량을 검증하는 날카로운 꼬리 질문을 생성하세요.

## 꼬리 질문 전략 (핵심)
답변에서 포착한 기술 키워드를 기반으로, 아래 전략 중 가장 적합한 것을 선택하세요:

1. **동작 원리 검증**: 기술을 언급했다면, 그 기술의 내부 동작을 이해하는지 확인
   - 예: "Redis를 캐시로 사용했다" → "Redis의 eviction 정책은 어떤 걸 사용하셨고, 그 이유는?"
   - 예: "JPA를 사용했다" → "N+1 문제가 발생한 적 있나요? 어떻게 해결하셨나요?"

2. **선택 근거 확인**: 특정 기술을 선택한 이유와 대안 비교를 확인
   - 예: "메시지 큐로 Kafka를 도입했다" → "RabbitMQ 대신 Kafka를 선택한 이유는? 처리량 요구사항이 어느 정도였나요?"

3. **장애·한계 경험 확인**: 해당 기술 사용 시 겪은 문제와 해결 과정을 확인
   - 예: "Docker로 배포했다" → "컨테이너 환경에서 겪은 가장 까다로운 문제는? 어떻게 디버깅하셨나요?"

4. **설계 의사결정 검증**: 아키텍처나 설계 선택의 트레이드오프를 확인
   - 예: "MSA로 전환했다" → "서비스 간 통신은 동기/비동기 중 어떤 방식을 선택했고, 데이터 정합성은 어떻게 보장했나요?"

5. **수치·성과 검증**: 정량적 결과를 주장했다면, 측정 방법과 기준을 확인
   - 예: "성능을 50% 개선했다" → "그 수치는 어떤 지표로 측정한 건가요? 개선 전후 구체적 수치를 말씀해 주세요"

## 꼬리 질문을 생성해야 하는 경우
- 답변에서 기술·도구를 언급했지만 표면적 수준에 머문 경우
- 결과만 말하고 과정·근거·트레이드오프 설명이 빠진 경우
- 이력서에 적힌 경험과 답변 사이에 괴리가 있는 경우
- 답변에 논리적 허점이나 모순이 있는 경우

## 꼬리 질문을 생성하지 않아야 하는 경우
- 답변이 이미 원리·근거·경험까지 충분히 다룬 경우
- 원래 질문이 단순 사실 확인(예/아니오)인 경우
- 추가 질문이 면접 흐름에 도움이 되지 않는 경우

## 중복 방지 지침 (필수)
- 대화 이력에 이미 등장한 질문과 동일하거나 유사한 질문을 절대 반복하지 마세요
- "구체적으로 설명해 주세요", "자세히 말씀해 주세요" 같은 포괄적 요청 대신, 특정 기술·상황을 지목하여 질문하세요
- 이전 답변에서 이미 충분히 다룬 내용을 다시 묻지 마세요
- 꼬리 질문은 반드시 최신 답변에서 새롭게 드러난 기술·경험·키워드를 타겟으로 해야 합니다

## 난이도별 판단 기준
{difficulty_criteria}

## 이력서 활용 지침
- 이력서 정보가 제공되면, 지원자가 실제로 경험했다고 주장하는 내용과 답변을 대조하세요
- 이력서에 관련 경험이 있는데 답변에서 언급하지 않았다면, 해당 경험을 끌어내는 꼬리 질문을 생성하세요
- 이력서 정보가 없더라도 답변 자체의 품질만으로 판단하세요

## 업계 참고 자료 활용 지침
- 채용공고, 기술블로그, 뉴스 등의 참고 자료가 제공되면, 업계에서 실제로 요구하는 역량과 답변을 대조하세요
- 채용공고에서 요구하는 기술을 답변에서 언급했지만 깊이가 부족하면, 해당 기술의 실무 적용 경험을 묻는 꼬리 질문을 생성하세요
- 기술블로그의 트렌드와 관련된 답변이라면, 최신 동향에 대한 이해도를 확인하는 꼬리 질문을 생성하세요
- 참고 자료가 없더라도 답변 자체의 품질만으로 판단하세요

## 응답 형식
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트를 포함하지 마세요.
{{
  "has_follow_up": true 또는 false,
  "follow_up_question": "꼬리 질문 텍스트" 또는 null,
  "reason": "판단 근거"
}}\
"""

DIFFICULTY_CRITERIA = {
    "low": (
        "현재 난이도: 하 (관대한 평가)\n"
        "답변이 질문의 핵심 개념을 포함하고 있다면 추가 질문이 필요하지 않습니다.\n"
        "기본적인 이해를 확인하는 수준으로 판단하세요.\n"
        "구체적 사례나 경험이 없어도, 핵심 개념만 언급했다면 충분합니다."
    ),
    "middle": (
        "현재 난이도: 중 (보통 평가)\n"
        "답변에 개념 설명과 함께 구체적인 경험이나 사례가 포함되어야 합니다.\n"
        "경험이나 사례가 빠져 있다면 이를 확인하는 꼬리질문을 생성하세요.\n"
        "개념만 나열한 답변은 부족하며, 적용 경험까지 확인해야 합니다."
    ),
    "high": (
        "현재 난이도: 상 (엄격한 평가)\n"
        "답변에 깊은 이해, 트레이드오프 분석, 대안 제시가 포함되어야 합니다.\n"
        "표면적이거나 암기식 답변이라면 반드시 꼬리질문을 생성하세요.\n"
        "단순 개념 설명이나 경험 나열만으로는 부족하며, "
        "왜 그 선택을 했는지, 다른 대안은 무엇이었는지까지 확인하세요."
    ),
}


def _build_system_prompt(difficulty: str) -> str:
    """난이도에 맞는 시스템 프롬프트를 구성한다."""
    criteria = DIFFICULTY_CRITERIA[difficulty]
    return _BASE_SYSTEM_PROMPT.format(difficulty_criteria=criteria)

DIFFICULTY_LABELS = {
    "low": "하",
    "middle": "중",
    "high": "상",
}

INTERVIEW_TYPE_LABELS = {
    "technical": "기술 면접",
    "personality": "인성 면접",
}


# ──────────────────────────────────────────────
# LangGraph 상태 정의
# ──────────────────────────────────────────────

class FollowUpState(TypedDict):
    request: FollowUpRequest
    user_message: str
    analysis: dict
    resume_context: str
    crawled_context: str
    response: FollowUpResponse


# ──────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────

async def _noop() -> str:
    return ""


def _build_conversation_text(request: FollowUpRequest) -> str:
    """대화 이력을 텍스트로 구성한다."""
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

    if len(request.conversation) > 1:
        asked = [turn.question for turn in request.conversation]
        parts.append("\n[이미 질문한 내용 - 절대 반복 금지]")
        for q in asked:
            parts.append(f"- {q}")

    return "\n".join(parts)


# ──────────────────────────────────────────────
# 노드 1: 답변 분석 (LLM 호출)
# ──────────────────────────────────────────────

async def analyze_answer(state: FollowUpState) -> dict:
    """답변 품질을 분석하고 핵심 키워드를 추출한다."""
    user_message = state["user_message"]

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=512,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ANALYSIS_PROMPT},
                {"role": "user", "content": user_message},
            ],
            timeout=15.0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content")
        analysis = json.loads(content.strip())
        if not isinstance(analysis, dict):
            raise ValueError("Response is not a JSON object")
    except (APITimeoutError, APIError, json.JSONDecodeError, KeyError, IndexError, AttributeError, ValueError) as e:
        logger.warning("답변 분석 실패, 기본값 사용: %s", e)
        analysis = {
            "quality": "partial",
            "keywords": [],
            "weak_points": [],
            "summary": "분석 불가",
        }

    return {"analysis": analysis}


# ──────────────────────────────────────────────
# 노드 2: ChromaDB 검색 (이력서 + 크롤링 데이터)
# ──────────────────────────────────────────────

# 면접 유형별 검색할 크롤링 데이터 source
_SOURCE_FILTER: dict[str, list[str]] = {
    "technical": ["tech_blog", "jobkorea"],
    "personality": ["jobkorea"],
}

def _query_resume(user_id: str, search_query: str) -> str:
    """ChromaDB resumes 컬렉션에서 이력서 청크를 검색한다."""
    collection = _get_collection()
    results = collection.query(
        query_texts=[search_query],
        n_results=3,
        where={"user_id": user_id},
    )

    documents = results.get("documents", [[]])[0]
    if not documents:
        return ""

    return "\n".join(f"- {doc[:300]}" for doc in documents)


async def search_context(state: FollowUpState) -> dict:
    """이력서와 크롤링 데이터를 ChromaDB에서 병렬 검색한다."""
    request = state["request"]
    analysis = state["analysis"]

    keywords = (
        analysis.get("tech_keywords", [])
        + analysis.get("concept_keywords", [])
    )
    if not keywords:
        return {"resume_context": "", "crawled_context": ""}

    search_query = " ".join(keywords)
    sources = _SOURCE_FILTER.get(request.interview_type, ["jobkorea"])

    # 이력서 + 크롤링 데이터 병렬 검색
    resume_task = (
        asyncio.to_thread(_query_resume, request.user_id, search_query)
        if request.user_id
        else _noop()
    )
    crawled_task = asyncio.to_thread(query_crawled_data, search_query, sources)

    results = await asyncio.gather(resume_task, crawled_task, return_exceptions=True)

    resume_context = results[0] if not isinstance(results[0], Exception) else ""
    crawled_context = results[1] if not isinstance(results[1], Exception) else ""

    if isinstance(results[0], Exception):
        logger.warning("이력서 검색 실패: %s", results[0])
    if isinstance(results[1], Exception):
        logger.warning("크롤링 데이터 검색 실패: %s", results[1])

    return {"resume_context": resume_context, "crawled_context": crawled_context}


# ──────────────────────────────────────────────
# 노드 3: 꼬리 질문 생성 (LLM 호출)
# ──────────────────────────────────────────────

async def generate_question(state: FollowUpState) -> dict:
    """답변 분석 결과, 이력서, 크롤링 데이터를 결합하여 꼬리 질문을 생성한다."""
    user_message = state["user_message"]
    analysis = state["analysis"]
    resume_context = state["resume_context"]
    crawled_context = state["crawled_context"]

    enriched_parts = [user_message]

    tech_kw = ", ".join(analysis.get("tech_keywords", [])) or "없음"
    concept_kw = ", ".join(analysis.get("concept_keywords", [])) or "없음"

    deep_dive_lines = []
    for target in analysis.get("deep_dive_targets", []):
        deep_dive_lines.append(
            f"  - {target.get('keyword', '?')}: {target.get('gap', '?')}"
        )
    deep_dive_text = "\n".join(deep_dive_lines) if deep_dive_lines else "  없음"

    enriched_parts.append(
        f"\n[답변 분석 결과]\n"
        f"품질: {analysis.get('quality', '알 수 없음')}\n"
        f"언급된 기술/도구: {tech_kw}\n"
        f"언급된 개념/패턴: {concept_kw}\n"
        f"심화 가능 포인트:\n{deep_dive_text}\n"
        f"부족한 부분: {', '.join(analysis.get('weak_points', []))}\n"
        f"요약: {analysis.get('summary', '')}"
    )

    if resume_context:
        enriched_parts.append(
            f"\n[지원자 이력서 관련 정보]\n{resume_context}"
        )

    if crawled_context:
        enriched_parts.append(
            f"\n[업계 참고 자료 (채용공고/기술블로그/뉴스)]\n{crawled_context}"
        )

    enriched_message = "\n".join(enriched_parts)

    system_prompt = _build_system_prompt(state["request"].difficulty)

    try:
        response = await _client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            max_tokens=512,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enriched_message},
            ],
            timeout=15.0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content")
        parsed = json.loads(content.strip())
        result = FollowUpResponse(
            has_follow_up=parsed["has_follow_up"],
            follow_up_question=parsed.get("follow_up_question"),
            reason=parsed.get("reason"),
        )
    except APITimeoutError:
        logger.error("꼬리 질문 생성 타임아웃")
        result = FollowUpResponse(
            has_follow_up=False, reason="AI 서비스 응답 시간 초과"
        )
    except APIError as e:
        logger.error("꼬리 질문 생성 API 오류: %s", e)
        result = FollowUpResponse(
            has_follow_up=False, reason="AI 서비스 호출 실패"
        )
    except (json.JSONDecodeError, KeyError, TypeError, IndexError, AttributeError, ValueError) as e:
        logger.error("꼬리 질문 응답 파싱 실패: %s", e)
        result = FollowUpResponse(
            has_follow_up=False, reason="AI 응답 파싱 실패"
        )

    # 꼬리 질문이 생성된 경우 면접관 페르소나 화자로 음성 합성.
    # TTS는 부가 기능이므로 실패해도 꼬리 질문 생성 자체는 정상 반환한다.
    if result.has_follow_up and result.follow_up_question:
        try:
            tts = get_tts_service()
            speaker = get_speaker_for_interview_persona(state["request"].persona)
            result.audio_url = await tts.synthesize(
                result.follow_up_question,
                speaker,
                f"interview_followup_{uuid.uuid4().hex}",
                prefix=settings.INTERVIEW_TTS_S3_PREFIX,
            )
        except Exception as e:
            logger.warning("꼬리 질문 TTS 합성 실패: %s", e)

    return {"response": result}


# ──────────────────────────────────────────────
# LangGraph 워크플로우 구성
# ──────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(FollowUpState)

    graph.add_node("analyze", analyze_answer)
    graph.add_node("search", search_context)
    graph.add_node("generate", generate_question)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "search")
    graph.add_edge("search", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


_workflow = _build_graph()


# ──────────────────────────────────────────────
# 외부 인터페이스
# ──────────────────────────────────────────────

async def generate_follow_up(request: FollowUpRequest) -> FollowUpResponse:
    """LangGraph 워크플로우를 실행하여 꼬리 질문을 생성한다.

    1단계: 답변 품질 분석 + 키워드 추출 (LLM)
    2단계: ChromaDB에서 이력서 + 크롤링 데이터(채용공고/기술블로그) 병렬 검색
    3단계: 분석 결과 + 이력서 + 크롤링 데이터를 결합하여 꼬리 질문 생성 (LLM)
    """
    initial_state: FollowUpState = {
        "request": request,
        "user_message": _build_conversation_text(request),
        "analysis": {},
        "resume_context": "",
        "crawled_context": "",
        "response": FollowUpResponse(has_follow_up=False),
    }

    result = await _workflow.ainvoke(initial_state)
    return result["response"]
