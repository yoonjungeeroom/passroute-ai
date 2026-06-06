import asyncio
import hashlib
import json
import re
import uuid
from pathlib import Path

from fastapi import HTTPException
from openai import OpenAIError

from app.core.config import settings
from app.schemas.debate import (
    InterviewerOpeningRequest, InterviewerOpeningResponse,
    InterviewerCueRequest, InterviewerCueResponse,
    DebateOpeningRequest, DebateOpeningResponse,
    DebateRebuttalRequest, DebateRebuttalResponse,
    DebateClosingRequest, DebateClosingResponse,
    InterviewerClosingRequest, InterviewerClosingResponse,
    DebateTurnEvalRequest, DebateTurnEvalResponse,
    DebateScoreItemWithWeight, DebateTurnScores, DebateTurnEvalSummary,
    DebateSessionSummaryRequest, DebateSessionSummaryResponse, TurnHighlight,
    DebateReportRequest, DebateReportResponse, DebateWeaknessItem, DebateTurnFeedback,
    DebateTopicSuggestRequest, DebateTopicSuggestResponse, DebateTopicCandidate,
    DebateTopicDetailRequest, DebateTopicDetailResponse,
)
from app.services.debate_news import query_news
from app.services.llm_service import _client, _parse_json
from app.services.tts_service import (
    get_tts_service,
    get_speaker_for_interviewer,
    get_speaker_for_persona,
)


# ── 프롬프트 캐싱 ──────────────────────────────────────────────────────────────

_PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts" / "debate"
_PROMPTS: dict[str, tuple[str, str]] = {}


def _load_prompts() -> None:
    for md_file in _PROMPT_DIR.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        parts = re.split(r"^## USER\s*$", content, flags=re.MULTILINE)
        system = re.sub(r"^## SYSTEM\s*\n?", "", parts[0], flags=re.MULTILINE).strip()
        user = parts[1].strip() if len(parts) > 1 else ""
        _PROMPTS[md_file.stem] = (system, user)


_load_prompts()


# ── 상수 ──────────────────────────────────────────────────────────────────────

_STANCE_LABELS = {"PRO": "찬성", "CON": "반대", "NEUTRAL": "중립"}
_ROUND_LABELS = {
    "OPENING": "입론",
    "REBUTTAL_1": "반박 1",
    "REBUTTAL_2": "반박 2",
    "CLOSING": "마무리",
    "MODERATION": "사회",
}
_DIFFICULTY_GUIDES = {
    "EASY": "논거를 단순하게 제시하고 발언을 간결하게 유지한다.",
    "NORMAL": "균형 있는 논리를 전개하며 적절한 근거를 제시한다.",
    "HARD": "날카로운 반박과 구체적 근거를 제시하며 적극적으로 논점을 공략한다.",
}
_DEBATE_WEIGHTS: dict[str, float] = {
    "logic": 0.35,
    "rebuttal_quality": 0.30,
    "consistency": 0.20,
    "attitude": 0.15,
}
_ROUND_ACTIVE_FIELDS: dict[str, set[str]] = {
    "OPENING":    {"logic", "attitude"},
    "REBUTTAL_1": {"logic", "rebuttal_quality", "consistency", "attitude"},
    "REBUTTAL_2": {"logic", "rebuttal_quality", "consistency", "attitude"},
    "CLOSING":    {"logic", "consistency", "attitude"},
    "MODERATION": {"logic", "attitude"},
}

# 라운드 전환 진행 멘트(cue) — 절차적·정형 문구라 LLM 없이 고정 템플릿 사용.
# 사용자가 각 라운드를 먼저 시작하는 흐름을 안내한다.
_INTERVIEWER_CUE_TEMPLATES: dict[str, str] = {
    "REBUTTAL_START": "양측의 입론이 모두 끝났습니다. 이제 반박 라운드를 시작하겠습니다. 사용자 측부터 상대방의 주장에 반박해 주세요.",
    "REBUTTAL_EXTRA": "반박을 한 차례 더 진행하겠습니다. 사용자 측부터 추가로 반박해 주세요.",
    "CLOSING_GUIDE": "이제 토론을 마무리하겠습니다. 마지막으로 사용자 측부터 최종 변론을 말씀해 주세요.",
}

# 발언이 비었거나 토막 수준이면 LLM 평가를 건너뛰고 최저점 처리한다.
# (공백 제외 20자 미만 또는 6어절 미만)
_MIN_EVAL_CHARS = 20
_MIN_EVAL_WORDS = 6
_INSUFFICIENT_FEEDBACK = "발언이 너무 짧거나 불충분하여 평가할 수 없습니다."
_INSUFFICIENT_IMPROVEMENT = (
    "발언이 너무 짧아 평가할 수 없습니다. 주장과 근거를 갖춰 충분한 길이로 다시 답변해 주세요."
)


def _is_insufficient_utterance(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return True
    words = text.split()
    char_count = len("".join(words))
    return char_count < _MIN_EVAL_CHARS or len(words) < _MIN_EVAL_WORDS


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────────────

def _fill_template(template: str, variables: dict) -> str:
    """변수 치환 후 {{ }} 이스케이프 해제. 2차 치환 방지를 위해 단일 패스 처리."""
    if not variables:
        return template.replace("{{", "{").replace("}}", "}")
    pattern = re.compile("|".join(re.escape("{" + k + "}") for k in variables))
    result = pattern.sub(lambda m: str(variables[m.group()[1:-1]]), template)
    return result.replace("{{", "{").replace("}}", "}")


def _format_history(history) -> str:
    if not history:
        return "없음"
    lines = []
    for turn in history:
        speaker = {
            "USER": "사용자",
            "AI_COMPETITOR": "AI 경쟁자",
            "AI_INTERVIEWER": "면접관",
        }.get(turn.speaker_type, turn.speaker_type)
        lines.append(f"[{speaker} - {_ROUND_LABELS.get(turn.round_type, turn.round_type)}] {turn.content}")
    return "\n".join(lines)


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _format_news(news_items: list[dict]) -> str:
    """뉴스 조회 결과를 프롬프트용 텍스트로 변환한다."""
    if not news_items:
        return "참고할 뉴스가 없습니다. 일반적인 개발자/IT 업계의 최신 쟁점을 활용하세요."
    lines = []
    for item in news_items:
        company = item.get("company_name") or ""
        content = (item.get("content") or "").strip()[:300].replace("\n", " ")
        tag = f"[{company}] " if company else ""
        lines.append(f"- {tag}{content}")
    return "\n".join(lines)


async def _call(
    prompt_key: str,
    variables: dict,
    max_tokens: int,
    timeout: float,
    model: str | None = None,
) -> dict:
    if prompt_key not in _PROMPTS:
        raise HTTPException(status_code=500, detail=f"프롬프트 설정을 찾을 수 없습니다: {prompt_key}")
    system_tpl, user_tpl = _PROMPTS[prompt_key]
    try:
        response = await _client.responses.create(
            model=model or settings.OPENAI_MODEL,
            instructions=_fill_template(system_tpl, variables),
            input=_fill_template(user_tpl, variables),
            max_output_tokens=max_tokens,
            text={"format": {"type": "json_object"}},
            timeout=timeout,
        )
        return _parse_json(response.output_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"LLM 응답 파싱 실패: {e}")
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI 호출 실패: {e}")


def _renormalize_and_score(
    raw_scores: dict, round_type: str
) -> tuple[DebateTurnScores, float]:
    active = _ROUND_ACTIVE_FIELDS.get(round_type, {"logic", "attitude"})
    total_weight = sum(_DEBATE_WEIGHTS[f] for f in active)
    scores_data: dict = {}
    weighted_sum = 0.0
    for field in ("logic", "rebuttal_quality", "consistency", "attitude"):
        raw = raw_scores.get(field)
        if field in active and raw and raw.get("score") is not None:
            norm_w = _DEBATE_WEIGHTS[field] / total_weight
            scores_data[field] = DebateScoreItemWithWeight(
                score=raw.get("score", 0), weight=round(norm_w, 4), feedback=raw.get("feedback", "")
            )
            weighted_sum += raw["score"] * norm_w
        else:
            scores_data[field] = None
    return DebateTurnScores(**scores_data), round(weighted_sum * 20, 2)


# ── 발화 생성 5개 (model=OPENAI_MODEL_DEBATE) ─────────────────────────────────

async def generate_interviewer_opening(req: InterviewerOpeningRequest) -> InterviewerOpeningResponse:
    raw = await _call(
        "interviewer_opening",
        {
            "topic_title": req.topic_title,
            "topic_description": req.topic_description,
            "user_stance_label": _STANCE_LABELS[req.user_stance],
            "ai_stance_label": _STANCE_LABELS[req.ai_stance],
            "difficulty": req.difficulty,
            "pro_key_points_text": _format_list(req.pro_key_points),
            "con_key_points_text": _format_list(req.con_key_points),
        },
        max_tokens=512,
        timeout=settings.DEBATE_GENERATION_TIMEOUT,
        model=settings.OPENAI_MODEL_DEBATE,
    )
    content = raw["content"]
    tts = get_tts_service()
    audio_url = await tts.synthesize(content, get_speaker_for_interviewer(), f"interviewer_opening_{uuid.uuid4().hex}")
    return InterviewerOpeningResponse(content=content, audio_url=audio_url)


async def generate_competitor_opening(req: DebateOpeningRequest) -> DebateOpeningResponse:
    my_points = req.pro_key_points if req.stance == "PRO" else req.con_key_points
    opp_points = req.con_key_points if req.stance == "PRO" else req.pro_key_points
    raw = await _call(
        "competitor_opening",
        {
            "name": req.persona.name,
            "background": req.persona.background,
            "persona_system_prompt": req.persona.system_prompt_template,
            "topic_title": req.topic_title,
            "stance_label": _STANCE_LABELS[req.stance],
            "my_key_points_text": _format_list(my_points),
            "opponent_key_points_text": _format_list(opp_points),
            "difficulty": req.difficulty,
            "difficulty_guide": _DIFFICULTY_GUIDES[req.difficulty],
        },
        max_tokens=512,
        timeout=settings.DEBATE_GENERATION_TIMEOUT,
        model=settings.OPENAI_MODEL_DEBATE,
    )
    content = raw["content"]
    tts = get_tts_service()
    audio_url = await tts.synthesize(
        content,
        get_speaker_for_persona(req.persona.persona_id),
        f"competitor_opening_{uuid.uuid4().hex}",
    )
    return DebateOpeningResponse(content=content, audio_url=audio_url)


async def generate_competitor_rebuttal(req: DebateRebuttalRequest) -> DebateRebuttalResponse:
    raw = await _call(
        "competitor_rebuttal",
        {
            "name": req.persona.name,
            "background": req.persona.background,
            "persona_system_prompt": req.persona.system_prompt_template,
            "topic_title": req.topic_title,
            "stance_label": _STANCE_LABELS[req.stance],
            "rebuttal_round_label": _ROUND_LABELS[req.rebuttal_round],
            "opponent_latest_turn": req.opponent_latest_turn,
            "history_text": _format_history(req.history),
            "difficulty": req.difficulty,
            "difficulty_guide": _DIFFICULTY_GUIDES[req.difficulty],
        },
        max_tokens=512,
        timeout=settings.DEBATE_GENERATION_TIMEOUT,
        model=settings.OPENAI_MODEL_DEBATE,
    )
    content = raw["content"]
    tts = get_tts_service()
    audio_url = await tts.synthesize(
        content,
        get_speaker_for_persona(req.persona.persona_id),
        f"competitor_rebuttal_{uuid.uuid4().hex}",
    )
    return DebateRebuttalResponse(content=content, audio_url=audio_url)


async def generate_competitor_closing(req: DebateClosingRequest) -> DebateClosingResponse:
    raw = await _call(
        "competitor_closing",
        {
            "name": req.persona.name,
            "background": req.persona.background,
            "persona_system_prompt": req.persona.system_prompt_template,
            "topic_title": req.topic_title,
            "stance_label": _STANCE_LABELS[req.stance],
            "history_text": _format_history(req.history),
            "difficulty": req.difficulty,
            "difficulty_guide": _DIFFICULTY_GUIDES[req.difficulty],
        },
        max_tokens=512,
        timeout=settings.DEBATE_GENERATION_TIMEOUT,
        model=settings.OPENAI_MODEL_DEBATE,
    )
    content = raw["content"]
    tts = get_tts_service()
    audio_url = await tts.synthesize(
        content,
        get_speaker_for_persona(req.persona.persona_id),
        f"competitor_closing_{uuid.uuid4().hex}",
    )
    return DebateClosingResponse(content=content, audio_url=audio_url)


async def generate_interviewer_closing(req: InterviewerClosingRequest) -> InterviewerClosingResponse:
    raw = await _call(
        "interviewer_closing",
        {
            "topic_title": req.topic_title,
            "history_text": _format_history(req.history),
        },
        max_tokens=384,
        timeout=settings.DEBATE_GENERATION_TIMEOUT,
        model=settings.OPENAI_MODEL_DEBATE,
    )
    content = raw["content"]
    tts = get_tts_service()
    audio_url = await tts.synthesize(
        content,
        get_speaker_for_interviewer(),
        f"interviewer_closing_{uuid.uuid4().hex}",
    )
    return InterviewerClosingResponse(content=content, audio_url=audio_url)


async def generate_interviewer_cue(req: InterviewerCueRequest) -> InterviewerCueResponse:
    # 정형 템플릿이라 LLM 호출 없음. TTS는 고정 문구라 캐싱(같은 오디오 재사용).
    content = _INTERVIEWER_CUE_TEMPLATES[req.cue_type]
    speaker = get_speaker_for_interviewer()
    # 화자나 템플릿 문구가 바뀌면 캐시가 자동 무효화되도록 file_key에 화자·콘텐츠 해시를 포함한다.
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()[:8]
    safe_speaker = speaker.lower().replace("-", "_").replace(":", "_")
    file_key = f"interviewer_cue_{req.cue_type.lower()}_{safe_speaker}_{content_hash}"
    tts = get_tts_service()
    audio_url = await tts.synthesize(content, speaker, file_key, cache=True)
    return InterviewerCueResponse(content=content, audio_url=audio_url)


# ── 평가/요약/리포트 (model=OPENAI_MODEL 기본값) ──────────────────────────────

async def evaluate_debate_turn(req: DebateTurnEvalRequest) -> DebateTurnEvalResponse:
    # 토막/무발화는 LLM이 없는 내용을 지어내 고득점을 주므로, 평가 전에 차단한다.
    if _is_insufficient_utterance(req.user_content):
        active = _ROUND_ACTIVE_FIELDS.get(req.round_type, {"logic", "attitude"})
        raw_scores = {
            field: {"score": 1, "feedback": _INSUFFICIENT_FEEDBACK} for field in active
        }
        scores, weighted_score = _renormalize_and_score(raw_scores, req.round_type)
        return DebateTurnEvalResponse(
            scores=scores,
            weighted_score=weighted_score,
            summary=DebateTurnEvalSummary(
                strengths="",
                improvements=_INSUFFICIENT_IMPROVEMENT,
            ),
        )

    raw = await _call(
        "evaluate_turn",
        {
            "topic_title": req.topic_title,
            "user_stance_label": _STANCE_LABELS[req.user_stance],
            "round_label": _ROUND_LABELS.get(req.round_type, req.round_type),
            "opponent_previous_turn_text": req.opponent_previous_turn or "없음 (첫 발언)",
            "user_content": req.user_content,
            "history_text": _format_history(req.history),
            "round_type": req.round_type,
        },
        max_tokens=1024,
        timeout=settings.DEBATE_EVAL_TIMEOUT,
    )
    scores, weighted_score = _renormalize_and_score(raw["scores"], req.round_type)
    return DebateTurnEvalResponse(
        scores=scores,
        weighted_score=weighted_score,
        summary=DebateTurnEvalSummary(**raw["summary"]),
    )


async def generate_debate_session_summary(
    req: DebateSessionSummaryRequest,
) -> DebateSessionSummaryResponse:
    raw = await _call(
        "session_summary",
        {
            "topic_title": req.topic_title,
            "user_stance_label": _STANCE_LABELS[req.user_stance],
            "difficulty": req.difficulty,
            "persona_name": req.persona_name,
            "turn_evaluations_json": json.dumps(
                [t.model_dump() for t in req.turn_evaluations], ensure_ascii=False, indent=2
            ),
            "ai_competitor_turns_json": json.dumps(
                req.ai_competitor_turns, ensure_ascii=False, indent=2
            ),
        },
        max_tokens=1024,
        timeout=settings.DEBATE_EVAL_TIMEOUT,
    )
    return DebateSessionSummaryResponse(
        overall=raw["overall"],
        strengths=raw["strengths"],
        improvements=raw["improvements"],
        strategy_feedback=raw["strategy_feedback"],
        turn_highlights=[TurnHighlight(**h) for h in raw["turn_highlights"]],
    )


async def generate_debate_report(req: DebateReportRequest) -> DebateReportResponse:
    scores = [t.weighted_score for t in req.turn_evaluations]
    avg_score = round(sum(scores) / len(scores), 2) if scores else 0.0
    raw = await _call(
        "report",
        {
            "topic_title": req.topic_title,
            "user_stance_label": _STANCE_LABELS[req.user_stance],
            "difficulty": req.difficulty,
            "persona_name": req.persona_name,
            "turn_evaluations_json": json.dumps(
                [t.model_dump() for t in req.turn_evaluations], ensure_ascii=False, indent=2
            ),
            "session_summary_json": json.dumps(
                req.session_summary.model_dump(), ensure_ascii=False, indent=2
            ),
            "average_weighted_score": avg_score,
        },
        max_tokens=2048,
        timeout=settings.DEBATE_EVAL_TIMEOUT,
    )
    return DebateReportResponse(
        overall=raw["overall"],
        strengths=raw["strengths"],
        weaknesses=[DebateWeaknessItem(**w) for w in raw["weaknesses"]],
        improvements=raw["improvements"],
        turn_feedback=[DebateTurnFeedback(**t) for t in raw["turn_feedback"]],
        strategy_analysis=raw["strategy_analysis"],
        recommended_topics=raw["recommended_topics"],
        final_advice=raw["final_advice"],
        debate_readiness_comment=raw["debate_readiness_comment"],
    )


# ── 토론 주제 추천/생성 (크롤링 뉴스 기반) ────────────────────────────────────

async def generate_debate_topics(req: DebateTopicSuggestRequest) -> DebateTopicSuggestResponse:
    """키워드 기반으로 naver_news를 조회하여 토론 주제 후보 N개를 생성한다."""
    search_query = " ".join(req.keywords).strip()
    news_items = await asyncio.to_thread(query_news, search_query, None, 8)

    raw = await _call(
        "topic_candidates",
        {
            "keywords_text": ", ".join(req.keywords) if req.keywords else "없음 (일반 주제)",
            "count": req.count,
            "news_text": _format_news(news_items),
        },
        max_tokens=1024,
        timeout=settings.DEBATE_GENERATION_TIMEOUT,
        model=settings.OPENAI_MODEL_DEBATE,
    )

    candidates = [DebateTopicCandidate(**c) for c in raw["candidates"]]
    return DebateTopicSuggestResponse(candidates=candidates, news_count=len(news_items))


async def generate_debate_topic_detail(req: DebateTopicDetailRequest) -> DebateTopicDetailResponse:
    """선택된 주제에 대해 뉴스를 다시 조회하여 상세(설명 + 찬/반 논거)를 생성한다."""
    news_items = await asyncio.to_thread(query_news, req.title, None, 5)

    raw = await _call(
        "topic_detail",
        {
            "title": req.title,
            "summary": req.summary or "없음",
            "news_text": _format_news(news_items),
        },
        max_tokens=1024,
        timeout=settings.DEBATE_GENERATION_TIMEOUT,
        model=settings.OPENAI_MODEL_DEBATE,
    )

    return DebateTopicDetailResponse(
        topic_title=req.title,
        category=req.category,
        topic_description=raw["topic_description"],
        pro_key_points=raw["pro_key_points"],
        con_key_points=raw["con_key_points"],
    )
