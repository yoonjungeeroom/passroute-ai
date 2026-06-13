import json
import logging
import re
from openai import AsyncOpenAI, OpenAIError
from fastapi import HTTPException
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.evaluation import (
    EvaluationSummary,
    FactCheck,
    IncorrectClaim,
    LLMScores,
    LLMScoresWithWeight,
    ScoreItemWithWeight,
    QuestionEvaluationRequest,
    QuestionEvaluationResponse,
    StarEvaluationRequest,
    StarEvaluationResponse,
    StarEvaluationDetail,
    StarBreakdown,
    SessionSummaryRequest,
    SessionSummaryResponse,
    SessionSummaryOutput,
    QuestionHighlight,
    ReportGenerationRequest,
    ReportGenerationResponse,
    WeaknessItem,
    QuestionDetailedFeedback,
    QuestionFeedback,
    SelfIntroReportRequest,
    SelfIntroReportResponse,
    SelfIntroSummaryOutput,
)

# score-policy.md 가중치 테이블
_WEIGHTS: dict[str, dict[str, float | None]] = {
    "technical": {
        "relevance": 0.15, "logic": 0.15, "specificity": 0.15,
        "conciseness": 0.10, "clarity": 0.10,
        "accuracy": 0.15, "depth": 0.15, "job_relevance": 0.05,
        "authenticity": None, "growth": None,
    },
    "personality": {
        "relevance": 0.15, "logic": 0.20, "specificity": 0.15,
        "conciseness": 0.15, "clarity": 0.15,
        "accuracy": None, "depth": None,
        "job_relevance": 0.05, "authenticity": 0.10, "growth": 0.05,
    },
}


logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=60.0)

# 리포트는 출력이 커서(문항별 구조화 피드백·fact_check 포함, 최대 9000토큰)
# 기본 60초로는 빠듯하므로 별도 타임아웃을 둔다.
_REPORT_TIMEOUT = 150.0

def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```[^\n]*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    return json.loads(text.strip())

async def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int,
    timeout: float | None = None,
    model: str | None = None,
) -> dict:
    try:
        kwargs: dict = dict(
            model=model or settings.OPENAI_MODEL,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=max_output_tokens,
            text={"format": {"type": "json_object"}},
        )
        if timeout is not None:
            kwargs["timeout"] = timeout
        response = await _client.responses.create(**kwargs)
        return _parse_json(response.output_text)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM 응답 파싱 실패: {e}")
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI 호출 실패: {e}")


def _merge_weights(llm_scores: LLMScores, question_type: str) -> LLMScoresWithWeight:
    weights = _WEIGHTS[question_type]
    data: dict = {}
    for field in LLMScores.model_fields:
        item = getattr(llm_scores, field)
        w = weights[field]
        if item is not None and w is not None:
            data[field] = ScoreItemWithWeight(score=item.score, weight=w, feedback=item.feedback)
        else:
            data[field] = None
    return LLMScoresWithWeight(**data)


# ── 신규 구조 필드 방어적 파싱 ───────────────────────────────────────────────────
# LLM이 새 필드(detailed_feedback / fact_check)를 누락하거나 일부만 깨뜨려 보내더라도
# 해당 필드만 None으로 떨어지고 리포트/평가 전체는 정상 생성되도록 한다.

def _build_fact_check(raw) -> FactCheck | None:
    """fact_check dict → FactCheck. 없거나 형식이 어긋나면 None."""
    if not isinstance(raw, dict):
        return None

    # 항목 생성·타입 변환 전부를 try 안에 둔다. LLM이 비정상 타입을 보내도
    # 예외가 helper 밖으로 새지 않고 fact_check만 None으로 격하되도록 한다.
    try:
        claims: list[IncorrectClaim] = []
        for c in raw.get("incorrect_claims") or []:
            if not isinstance(c, dict):
                continue
            # snake_case 우선, camelCase도 방어적으로 수용한다.
            claims.append(IncorrectClaim(
                user_claim=c.get("user_claim") or c.get("userClaim") or "",
                issue=c.get("issue") or "",
                correct_explanation=c.get("correct_explanation") or c.get("correctExplanation") or "",
                suggested_fix=c.get("suggested_fix") or c.get("suggestedFix") or "",
            ))

        # 리스트가 아니면(문자열 등) 빈 리스트로 처리해 문자 단위 순회 버그를 막는다.
        unsupported_raw = raw.get("unsupported_claims")
        unsupported = (
            [str(u) for u in unsupported_raw if u]
            if isinstance(unsupported_raw, list)
            else []
        )

        return FactCheck(
            is_fact_check_applicable=bool(
                raw.get("is_fact_check_applicable", raw.get("isFactCheckApplicable", False))
            ),
            incorrect_claims=claims,
            unsupported_claims=unsupported,
            correct_explanation=raw.get("correct_explanation") or None,
            suggested_fix=raw.get("suggested_fix") or None,
        )
    except (TypeError, ValidationError) as e:
        logger.warning("fact_check 파싱 스킵: %s | %s", e, raw)
        return None


def _build_detailed_feedback(raw) -> QuestionDetailedFeedback | None:
    """detailed_feedback dict → QuestionDetailedFeedback. 없거나 형식이 어긋나면 None."""
    if not isinstance(raw, dict):
        return None
    try:
        # 리스트가 아니면(문자열 등) 빈 리스트로 처리해 문자 단위 순회 버그를 막는다.
        missing_info_raw = raw.get("missing_info")
        missing_info = (
            [str(m) for m in missing_info_raw if m]
            if isinstance(missing_info_raw, list)
            else []
        )
        return QuestionDetailedFeedback(
            strength=raw.get("strength") or "",
            weakness=raw.get("weakness") or "",
            missing_info=missing_info,
            improvement_example=raw.get("improvement_example") or None,
            suggested_answer=raw.get("suggested_answer") or None,
            retry_strategy=raw.get("retry_strategy") or None,
        )
    except (TypeError, ValidationError) as e:
        logger.warning("detailed_feedback 파싱 스킵: %s | %s", e, raw)
        return None


# ── 질문 단위 평가 ─────────────────────────────────────────────────────────────

async def evaluate_question(req: QuestionEvaluationRequest) -> QuestionEvaluationResponse:
    type_note = (
        "기술 질문 → accuracy, depth 평가 / authenticity, growth는 null"
        if req.question_type == "technical"
        else "인성 질문 → authenticity, growth 평가 / accuracy, depth는 null"
    )
    user_prompt = f"""아래 면접 답변을 평가해주세요.

[채용 정보]
- 직무: {req.job_title}
- 회사: {req.company_name}
- JD 키워드: {req.jd_keywords}

[질문 유형] {req.question_type} — {type_note}
[질문] {req.question}
[답변] {req.answer}

---

평가 항목 (score: 1~5 정수, feedback: 한국어)
- 공통: relevance, logic, specificity, conciseness, clarity, job_relevance
- technical 전용: accuracy, depth
- personality 전용: authenticity, growth

[feedback 작성 원칙 — 반드시 준수]
- 각 항목 feedback은 "그 점수를 준 근거"를 답변 내용에 기반해 구체적으로 적는다.
  추상적 표현("깊이가 부족함", "구체적이지 않음")만 쓰지 말고, 답변의 어느 부분 때문인지 밝힌다.
- 답변에 실제로 있는 표현·키워드를 인용/지목한다. 답변에 없는 내용을 있는 것처럼 평가하지 않는다.
- 감점 항목(3점 이하)은 "무엇이 빠졌는지 + 어떻게 고치면 되는지"를 함께 적는다.
- 형식 예시:
  · depth 3: "Redis로 캐싱했다"고 답했으나 왜 Memcached가 아닌 Redis인지, TTL·메모리 트레이드오프 언급이 없음. 선택의 비교 근거를 덧붙이면 좋음.
  · specificity 2: "협업을 잘했다"는 평가만 있고 구체적 상황·역할·수치가 없음. STAR 형태로 사례 한 개를 풀어 쓰면 설득력이 올라감.
- 답변이 비었거나 질문과 무관하면 해당 항목을 1~2점으로 주고 그 사유를 feedback에 명시한다.

[summary 작성 원칙]
- strengths/improvements도 답변 내용을 근거로 구체적으로 작성한다.
- improvements는 "~한 답변은 ~해서 ~이 필요해 보입니다" 형태로, 무엇을 어떻게 보완할지 명시한다.

[fact_check 작성 원칙 — 기술 검증]
- 질문이 technical이거나 답변에 검증 가능한 기술적 주장이 있으면 is_fact_check_applicable=true로 두고 아래를 채운다.
  기술 주장이 없는 인성 답변이면 is_fact_check_applicable=false, incorrect_claims/unsupported_claims는 빈 배열로 둔다.
- incorrect_claims: 명백히 틀린 기술 주장만 담는다. 각 항목은 사용자가 한 주장(user_claim), 무엇이 왜 틀렸는지(issue),
  올바른 개념(correct_explanation), 고쳐 말하는 예시(suggested_fix)를 포함한다.
- unsupported_claims: 틀렸다고 단정할 수는 없으나 근거 없이 단정한 주장(문자열). 확실하지 않으면 incorrect_claims가 아니라 여기에 둔다.
- accuracy 점수와 모순되지 않게 작성한다(틀린 주장이 있으면 accuracy 점수도 그에 맞게 낮아야 한다).

JSON만 반환:
{{
  "llm_scores": {{
    "relevance":     {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}},
    "logic":         {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}},
    "specificity":   {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}},
    "conciseness":   {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}},
    "clarity":       {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}},
    "accuracy":      {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}} 또는 null,
    "depth":         {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}} 또는 null,
    "job_relevance": {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}},
    "authenticity":  {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}} 또는 null,
    "growth":        {{"score": 정수, "feedback": "점수 근거 + 답변 인용 1~2문장"}} 또는 null
  }},
  "summary": {{"strengths": "답변 근거 기반 강점 1~2문장", "improvements": "답변 근거 기반 개선 방향 1~2문장"}},
  "fact_check": {{
    "is_fact_check_applicable": true 또는 false,
    "incorrect_claims": [
      {{"user_claim": "사용자가 한 주장", "issue": "무엇이 왜 틀렸는지", "correct_explanation": "올바른 개념", "suggested_fix": "고쳐 말하는 예시"}}
    ],
    "unsupported_claims": ["근거 없이 단정한 주장"]
  }}
}}"""

    raw = await _call_llm_json(
        system_prompt="당신은 채용 면접 평가 전문가입니다. 답변을 항목별로 평가하되, 각 점수의 근거를 답변 내용에 기반해 구체적으로 제시합니다. 기술적 주장은 사실 여부를 검증하되 확실하지 않은 내용을 틀렸다고 단정하지 않습니다. JSON 형식으로만 반환하며 JSON 외 텍스트는 포함하지 마세요.",
        user_prompt=user_prompt,
        max_output_tokens=4200,
        model=settings.OPENAI_MODEL_EVALUATION,
    )

    scores_raw = raw.get("llm_scores")
    if not isinstance(scores_raw, dict):
        scores_raw = {}
    # 점수가 null로 온 항목은 dict가 아닌 None으로 정규화한다. (Optional/필수 공통)
    for field, val in list(scores_raw.items()):
        if isinstance(val, dict) and val.get("score") is None:
            scores_raw[field] = None

    raw_summary = raw.get("summary")
    if not isinstance(raw_summary, dict):
        raw_summary = {}

    try:
        llm_scores = LLMScores(**scores_raw)
        summary = EvaluationSummary(**raw_summary)
    except (TypeError, KeyError, ValidationError) as e:
        logger.error("질문 평가 응답 구성 실패: %s | raw=%s", e, str(raw)[:500])
        raise HTTPException(status_code=500, detail=f"질문 평가 응답 구성 실패: {e}")

    return QuestionEvaluationResponse(
        llm_scores=_merge_weights(llm_scores, req.question_type),
        summary=summary,
        fact_check=_build_fact_check(raw.get("fact_check")),
    )


# ── STAR 평가 ─────────────────────────────────────────────────────────────────

async def evaluate_star(req: StarEvaluationRequest) -> StarEvaluationResponse:
    user_prompt = f"""아래 면접 답변에서 STAR 구조 각 요소가 포함되어 있는지 판단해주세요.

[질문] {req.question}
[답변] {req.answer}

STAR 기준:
- Situation: 배경 상황이나 맥락을 설명했는가
- Task: 본인의 역할이나 해결해야 할 과제를 언급했는가
- Action: 실제로 취한 행동이나 방법을 구체적으로 설명했는가
- Result: 결과나 성과를 언급했는가

단순 기술 개념 설명형 질문(정의, 원리, 장단점)은 applicable=false.
경험/행동 기반 질문은 applicable=true.

JSON만 반환:
{{
  "star_evaluation": {{
    "applicable": true 또는 false,
    "reason": "판단 이유",
    "star_breakdown": {{
      "situation": {{"present": true/false, "feedback": ""}},
      "task":      {{"present": true/false, "feedback": ""}},
      "action":    {{"present": true/false, "feedback": ""}},
      "result":    {{"present": true/false, "feedback": ""}}
    }} 또는 null,
    "star_score": 0~4 정수 또는 null
  }}
}}"""

    raw = await _call_llm_json(
        system_prompt="당신은 면접 답변 구조 분석 전문가입니다. STAR 구조 포함 여부를 판단하고 JSON 형식으로만 반환합니다.",
        user_prompt=user_prompt,
        max_output_tokens=1024,
    )

    star_data = raw["star_evaluation"]
    breakdown = None
    if star_data.get("star_breakdown"):
        breakdown = StarBreakdown(**star_data["star_breakdown"])

    return StarEvaluationResponse(
        star_evaluation=StarEvaluationDetail(
            applicable=star_data["applicable"],
            reason=star_data["reason"],
            star_breakdown=breakdown,
            star_score=star_data.get("star_score"),
        )
    )


# ── 세션 요약 ─────────────────────────────────────────────────────────────────

async def generate_session_summary(req: SessionSummaryRequest) -> SessionSummaryResponse:
    summaries_json = json.dumps(
        [s.model_dump() for s in req.per_question_summaries],
        ensure_ascii=False, indent=2,
    )
    item_avgs_json = json.dumps(req.item_averages.model_dump(), ensure_ascii=False, indent=2)
    session_score_json = json.dumps(req.session_score.model_dump(), ensure_ascii=False)

    user_prompt = f"""아래는 면접 세션 전체의 평가 데이터입니다. 종합 피드백을 생성해주세요.

[직무 정보]
- 직무: {req.job_title}
- 회사: {req.company_name}

[질문별 요약]
{summaries_json}

[항목별 평균 점수]
{item_avgs_json}

[세션 점수]
{session_score_json}

[최고 답변] 질문 {req.best_q.question_index}번 ({req.best_q.percentage:.0f}점): {req.best_q.question}
[최저 답변] 질문 {req.worst_q.question_index}번 ({req.worst_q.percentage:.0f}점): {req.worst_q.question}

JSON만 반환:
{{
  "overall": "세션 전체 흐름 기반 종합 평가 2~3문장",
  "strengths": "세션 전반에서 반복적으로 잘한 점 1~2문장",
  "improvements": "세션 전반에서 반복적으로 부족한 점 + 개선 방향 1~2문장",
  "question_highlights": [
    {{"question_index": 최고 답변 인덱스, "type": "best", "comment": "코멘트 1문장"}},
    {{"question_index": 최저 답변 인덱스, "type": "worst", "comment": "코멘트 1문장"}}
  ]
}}"""

    raw = await _call_llm_json(
        system_prompt="당신은 채용 면접 평가 전문가입니다. 세션 전체의 평가 결과를 바탕으로 종합 피드백을 생성합니다. JSON 형식으로만 반환합니다.",
        user_prompt=user_prompt,
        max_output_tokens=1024,
        model=settings.OPENAI_MODEL_EVALUATION,
    )

    highlights = [QuestionHighlight(**h) for h in raw["question_highlights"]]
    return SessionSummaryResponse(
        session_summary=SessionSummaryOutput(
            overall=raw["overall"],
            strengths=raw["strengths"],
            improvements=raw["improvements"],
            question_highlights=highlights,
        )
    )


# ── 리포트 생성 ───────────────────────────────────────────────────────────────

async def generate_report(req: ReportGenerationRequest) -> ReportGenerationResponse:
    evals_json = json.dumps(
        [e.model_dump() for e in req.question_evaluations],
        ensure_ascii=False, indent=2,
    )
    session_json = json.dumps(req.session_result.model_dump(), ensure_ascii=False, indent=2)
    voice_json = (
        json.dumps(req.voice_highlight.model_dump(), ensure_ascii=False)
        if req.voice_highlight else "null"
    )

    user_prompt = f"""아래 면접 평가 데이터를 바탕으로 종합 면접 리포트를 생성해주세요.

[직무 정보]
- 직무: {req.job_title}
- 회사: {req.company_name}

[질문별 평가 결과]
{evals_json}

[세션 전체 결과]
{session_json}

[음성/하이라이트 분석]
{voice_json}

---

작성 지침:
- 각 질문 데이터의 answer(답변 원문)와 summary를 함께 근거로 사용한다. answer가 있으면 그 내용을 직접 인용/지목해 평가하고, answer가 null이면 summary 기반으로만 작성한다.

[종합 평가 — 점수와 톤을 반드시 일치시킬 것]
- overall/strengths/weaknesses/improvements의 톤은 실제 점수와 일치해야 한다. session_result.percentage와 각 문항 percentage를 근거로 삼는다.
  점수가 낮은데(예: 다수 문항이 60점 미만, 또는 세션 percentage가 낮음) 종합평가만 지나치게 긍정적으로 쓰지 않는다. 반대로 점수가 높은데 과도하게 부정적으로 쓰지도 않는다.
- overall(3~5문장)에는 다음이 모두 드러나야 한다: (1) 전체적으로 잘한 점 (2) 반복적으로 부족했던 점 (3) 가장 큰 감점 원인 (4) 다음 연습에서 우선 개선할 포인트.
- strengths: 세션 전반에서 일관되게 잘한 점 1~2문장. 어떤 답변에서 드러났는지 구체적으로.
- weaknesses: key_weakness 항목 기준. 각 항목은 item/comment 구조. comment는 어느 답변의 어떤 부분 때문에 약점인지 근거를 든다.
- improvements: 구체적 행동 방향 1~2문장. "열심히 하세요" 같은 추상적 표현 금지.

[문항별 피드백 — 문항마다 반드시 다르게 작성할 것]
- 모든 문항에 같은 문장을 재사용하지 않는다. "구체성이 부족합니다", "STAR 기법을 활용하세요" 같은 일반론을 단독으로 반복하지 말 것.
- 질문 의도와 사용자의 실제 답변에 맞춰 관점을 달리한다. 예시:
  · 지원동기/직무 질문 → 직무 이해도, 동기, 경험 연결성 중심
  · 협업/갈등 질문 → 갈등 상황, 본인의 행동, 조율 과정, 결과 중심
  · 강점/약점 질문 → 자기 인식, 약점의 구체성, 개선 노력, 실제 변화 중심
- feedback(기존 단문): 답변 인용 + 근거 + 개선 방향 2~3문장.
- detailed_feedback(구조화): 사용자의 실제 답변을 기준으로 작성한다.
  · strength: 이 답변에서 실제로 잘 드러난 점 (답변 근거)
  · weakness: 부족한 점을, "어느 부분이 왜" 부족한지 답변에 근거해 구체적으로
  · missing_info: 답변에서 빠진 핵심 정보들 (문자열 배열). 예: 실제 충돌 의견, 본인이 제시한 기준, 확정된 결과 등
  · improvement_example: 다음 답변에 추가하면 좋은 1~3문장 예시 (사용자 답변 맥락 반영)
  · suggested_answer: 사용자의 답변을 더 낫게 고쳐 쓴 1~3문장 예시
  · retry_strategy: 다음 연습 때의 답변 전략
- [60점 미만 필수 규칙] percentage < 60인 문항은 improvement_example, suggested_answer, retry_strategy를 반드시 채운다(사용자가 다음 연습에서 바로 활용할 수 있는 1~3문장).
  percentage가 60 이상이면 이 세 필드는 생략(null)해도 된다.

[기술 사실 검증 — fact_check]
- question_type이 technical이거나 답변에 검증 가능한 기술적 주장이 있는 문항은 fact_check를 작성한다.
  · is_fact_check_applicable: 기술 검증이 필요한 답변이면 true. 기술 주장이 없는 인성 답변이면 false로 두고 나머지는 빈 배열.
  · incorrect_claims: 명백히 틀린 기술 주장만. 각 항목 {{user_claim, issue, correct_explanation, suggested_fix}}.
  · unsupported_claims: 틀렸다고 단정할 수는 없으나 근거 없이 단정한 주장(문자열 배열).
- 확실하지 않은 내용을 틀렸다고 단정하지 말 것. 애매하면 incorrect_claims가 아니라 unsupported_claims로 분류한다.

[기타]
- star_comment는 star_evaluation.applicable=true일 때만, voice_comment는 voice_feedback이 있을 때만 작성.
- voice_highlight 데이터는 별도 필드 출력 없이 overall/strengths/improvements에 자연스럽게 반영.
- recommended_questions: 이번 세션의 약점과 부족한 답변을 바탕으로 다음 연습에서 풀어볼 면접 질문 3개. 직무와 약점 항목에 맞게 구체적으로. 질문 텍스트만 문자열로 반환.
- final_advice: 다음 면접 연습을 위한 가장 중요한 조언 1~2문장.
- readiness_comment: interview_readiness.decision을 수치 노출 없이 사용자 친화적으로 해석.

JSON만 반환:
{{
  "overall": "",
  "strengths": "",
  "weaknesses": [{{"item": "항목명", "comment": "왜 약점인지 1문장"}}],
  "improvements": "",
  "question_feedback": [
    {{
      "question_index": 정수,
      "question": "",
      "question_type": "technical 또는 personality",
      "percentage": 숫자,
      "feedback": "답변 인용 + 근거 + 개선 방향 2~3문장",
      "detailed_feedback": {{
        "strength": "이 답변에서 잘 드러난 점",
        "weakness": "어느 부분이 왜 부족한지",
        "missing_info": ["빠진 정보 1", "빠진 정보 2"],
        "improvement_example": "다음 답변에 추가하면 좋은 예시 문장 (60점 미만 필수)" 또는 null,
        "suggested_answer": "고쳐 쓴 예시 답변 (60점 미만 필수)" 또는 null,
        "retry_strategy": "다음 연습 답변 전략 (60점 미만 필수)" 또는 null
      }},
      "fact_check": {{
        "is_fact_check_applicable": true 또는 false,
        "incorrect_claims": [
          {{"user_claim": "사용자가 한 주장", "issue": "무엇이 왜 틀렸는지", "correct_explanation": "올바른 개념", "suggested_fix": "고쳐 말하는 예시"}}
        ],
        "unsupported_claims": ["근거 없이 단정한 주장"]
      }},
      "star_comment": "" 또는 null,
      "voice_comment": "" 또는 null
    }}
  ],
  "recommended_questions": ["질문1", "질문2", "질문3"],
  "final_advice": "",
  "readiness_comment": ""
}}"""

    raw = await _call_llm_json(
        system_prompt=(
            "당신은 채용 면접 피드백 전문가입니다. 면접 평가 데이터와 답변 원문을 종합하여, "
            "점수의 근거와 개선 방향을 답변 내용에 기반해 구체적으로 제시하는 최종 리포트를 생성합니다. "
            "종합평가의 톤은 실제 점수와 일치시키고, 문항마다 질문 의도와 답변에 맞는 서로 다른 피드백을 작성하며, "
            "기술적 주장은 사실 여부를 검증하되 확실하지 않은 내용을 틀렸다고 단정하지 않습니다. "
            "JSON 형식으로만 반환합니다."
        ),
        user_prompt=user_prompt,
        max_output_tokens=9000,
        timeout=_REPORT_TIMEOUT,
        model=settings.OPENAI_MODEL_EVALUATION,
    )

    # 리스트 항목은 개별로 검증하여, 일부 항목이 어긋나도 리포트 전체가 실패하지 않도록 한다.
    # LLM이 리스트 필드를 null로 주거나 항목이 dict가 아니어도 안전하게 건너뛴다.
    weaknesses = []
    for w in raw.get("weaknesses") or []:
        if not isinstance(w, dict):
            continue
        try:
            weaknesses.append(WeaknessItem(**w))
        except (TypeError, ValidationError) as e:
            logger.warning("리포트 weakness 항목 스킵: %s | %s", e, w)

    question_feedback = []
    for q in raw.get("question_feedback") or []:
        if not isinstance(q, dict):
            continue
        # 중첩 구조(detailed_feedback / fact_check)는 먼저 방어적으로 만들어,
        # 일부가 깨져도 해당 필드만 None으로 떨어지고 문항 자체는 살아남게 한다.
        detailed = _build_detailed_feedback(q.get("detailed_feedback"))
        fact_check = _build_fact_check(q.get("fact_check"))
        try:
            question_feedback.append(QuestionFeedback(
                question_index=q["question_index"],
                question=q["question"],
                question_type=q["question_type"],
                percentage=q["percentage"],
                feedback=q.get("feedback") or "",
                star_comment=q.get("star_comment"),
                voice_comment=q.get("voice_comment"),
                detailed_feedback=detailed,
                fact_check=fact_check,
            ))
        except (KeyError, TypeError, ValidationError) as e:
            logger.warning("리포트 question_feedback 항목 스킵: %s | %s", e, q)

    # 텍스트/리스트 필드가 null로 와도 기본값으로 대체한다. (get(k, default)는 값이 null이면 None을 그대로 반환)
    try:
        return ReportGenerationResponse(
            overall=raw.get("overall") or "",
            strengths=raw.get("strengths") or "",
            weaknesses=weaknesses,
            improvements=raw.get("improvements") or "",
            question_feedback=question_feedback,
            recommended_questions=raw.get("recommended_questions") or [],
            final_advice=raw.get("final_advice") or "",
            readiness_comment=raw.get("readiness_comment") or "",
        )
    except ValidationError as e:
        logger.error("리포트 응답 구성 실패: %s | raw=%s", e, str(raw)[:500])
        raise HTTPException(status_code=500, detail=f"리포트 응답 구성 실패: {e}")


# ── 자소서별 종합 리포트 ────────────────────────────────────────────────────────

async def generate_self_intro_summary(req: SelfIntroReportRequest) -> SelfIntroReportResponse:
    item_averages_json = json.dumps(req.item_averages, ensure_ascii=False, indent=2)
    item_trend_json = json.dumps(
        [t.model_dump() for t in req.item_trend], ensure_ascii=False, indent=2
    )
    sessions_json = json.dumps(
        [s.model_dump() for s in req.sessions], ensure_ascii=False, indent=2
    )

    user_prompt = f"""아래는 한 자소서(=한 회사 지원)에 대한 여러 회차 연습 세션의 집계 데이터입니다.
회차 간 추세와 반복 패턴을 종합한 자소서별 리포트를 생성해주세요.

[지원 정보]
- 직무: {req.job_title}
- 회사: {req.company_name}
- 총 연습 회차: {req.total_sessions}회
- 전체 평균 점수: {req.overall_average:.0f}점 (100점 만점)
- 준비도: {req.readiness}

[항목별 평균 점수 (5점 만점)]
{item_averages_json}

[항목별 추세 (첫 회차 ↔ 마지막 회차, 5점 만점)]
{item_trend_json}

[회차별 요약 (round 오름차순, 점수는 100점 만점)]
{sessions_json}

---

작성 지침:
- 점수 척도를 절대 혼동하지 말 것: 세션/전체 점수는 100점 만점, 항목 평균(item_averages·item_trend)은 5점 만점.
- overall: 회차별 점수 흐름으로 추세(상승/정체/하락)를 진단하고 item_trend를 참고해 종합 평가를 3~5문장으로 작성. "{req.company_name} {req.job_title}" 지원에 특화된 코멘트로 작성하고 일반론은 금지.
- repeated_weakness: 여러 회차의 key_weaknesses에 반복 등장하는 항목을 "이 회사 면접에서 반복적으로 걸리는 약점"으로 구체적으로 진단. 반복이 뚜렷하지 않으면 가장 자주 등장한 약점을 기술.
- next_steps: 위 약점·추세를 바탕으로 이 회사/직무 다음 연습에서 집중할 방향을 2~3문장으로 제안.
- total_sessions가 1이면: 추세·반복을 단정하지 말 것. overall은 단일 회차 코멘트로 작성하고, repeated_weakness는 해당 회차의 주요 약점만 기술한다(반복이라는 표현 금지).
- 입력에 없는 사실을 지어내지 말 것. 답변 원문은 제공되지 않으며 집계 데이터만 사용한다.
- 톤: 격려하되 구체적 개선점을 담은 한국어 피드백.

JSON만 반환:
{{
  "overall": "추세 진단 포함 종합 피드백 3~5문장",
  "repeated_weakness": "회차 간 반복 약점 진단",
  "next_steps": "이 회사/직무 다음 연습 방향 제안 2~3문장"
}}"""

    raw = await _call_llm_json(
        system_prompt="당신은 채용 면접 피드백 전문가입니다. 한 자소서에 대한 여러 회차 연습 세션의 집계 데이터를 바탕으로, 회차 간 추세와 반복 패턴을 진단하는 자소서별 종합 피드백을 생성합니다. JSON 형식으로만 반환합니다.",
        user_prompt=user_prompt,
        max_output_tokens=1024,
        model=settings.OPENAI_MODEL_EVALUATION,
    )

    try:
        return SelfIntroReportResponse(
            self_intro_summary=SelfIntroSummaryOutput(
                overall=raw.get("overall") or "",
                repeated_weakness=raw.get("repeated_weakness") or "",
                next_steps=raw.get("next_steps") or "",
            )
        )
    except (ValidationError, AttributeError) as e:
        logger.error("자소서 종합 리포트 응답 구성 실패: %s | raw=%s", e, str(raw)[:500])
        raise HTTPException(status_code=500, detail=f"자소서 종합 리포트 응답 구성 실패: {e}")