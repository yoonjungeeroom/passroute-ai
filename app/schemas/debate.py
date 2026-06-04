from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

# ── 공통 타입 ──────────────────────────────────────────────────────────────────

DebateStance = Literal["PRO", "CON"]
DebateRound = Literal["OPENING", "REBUTTAL_1", "REBUTTAL_2", "CLOSING", "MODERATION"]
SpeakerType = Literal["USER", "AI_COMPETITOR", "AI_INTERVIEWER"]
DebateStyle = Literal["COOPERATIVE", "AGGRESSIVE", "STORYTELLING", "DATA_DRIVEN"]
Difficulty = Literal["EASY", "NORMAL", "HARD"]


class PersonaPayload(BaseModel):
    persona_id: str
    name: str
    background: str
    debate_style: DebateStyle
    difficulty: Difficulty
    strengths: list[str]
    weaknesses: list[str]
    system_prompt_template: str


class DebateTurnItem(BaseModel):
    speaker_type: SpeakerType
    round_type: DebateRound
    stance: Literal["PRO", "CON", "NEUTRAL"]
    content: str


# ── 발화 생성 (/debate/*) ──────────────────────────────────────────────────────

class InterviewerOpeningRequest(BaseModel):
    topic_title: str
    topic_description: str
    user_stance: DebateStance
    ai_stance: DebateStance
    difficulty: Difficulty
    pro_key_points: list[str]
    con_key_points: list[str]


class InterviewerOpeningResponse(BaseModel):
    content: str
    audio_url: str | None = None


class DebateOpeningRequest(BaseModel):
    topic_title: str
    topic_description: str
    stance: DebateStance
    difficulty: Difficulty
    persona: PersonaPayload
    pro_key_points: list[str]
    con_key_points: list[str]


class DebateOpeningResponse(BaseModel):
    content: str
    audio_url: str | None = None


class DebateRebuttalRequest(BaseModel):
    topic_title: str
    stance: DebateStance
    difficulty: Difficulty
    persona: PersonaPayload
    rebuttal_round: Literal["REBUTTAL_1", "REBUTTAL_2"]
    opponent_latest_turn: str
    history: list[DebateTurnItem]


class DebateRebuttalResponse(BaseModel):
    content: str
    audio_url: str | None = None


class DebateClosingRequest(BaseModel):
    topic_title: str
    stance: DebateStance
    difficulty: Difficulty
    persona: PersonaPayload
    history: list[DebateTurnItem]


class DebateClosingResponse(BaseModel):
    content: str
    audio_url: str | None = None


class InterviewerClosingRequest(BaseModel):
    topic_title: str
    history: list[DebateTurnItem]


class InterviewerClosingResponse(BaseModel):
    content: str
    audio_url: str | None = None


# ── 평가 (/evaluate/debate-turn) ──────────────────────────────────────────────

class DebateTurnEvalRequest(BaseModel):
    topic_title: str
    user_stance: DebateStance
    round_type: DebateRound
    user_content: str
    opponent_previous_turn: Optional[str] = None
    history: list[DebateTurnItem]


class DebateScoreItem(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str


class DebateScoreItemWithWeight(BaseModel):
    score: int = Field(ge=1, le=5)
    weight: float
    feedback: str


class DebateTurnScores(BaseModel):
    logic: DebateScoreItemWithWeight
    rebuttal_quality: Optional[DebateScoreItemWithWeight] = None
    consistency: Optional[DebateScoreItemWithWeight] = None
    attitude: DebateScoreItemWithWeight


class DebateTurnEvalSummary(BaseModel):
    strengths: str
    improvements: str


class DebateTurnEvalResponse(BaseModel):
    scores: DebateTurnScores
    weighted_score: float = Field(ge=0, le=100)
    summary: DebateTurnEvalSummary


# ── 세션 요약 (/debate/session-summary) ───────────────────────────────────────

class DebateTurnEvalItem(BaseModel):
    round_type: DebateRound
    user_content: str
    weighted_score: float
    summary: DebateTurnEvalSummary


class DebateSessionSummaryRequest(BaseModel):
    topic_title: str
    user_stance: DebateStance
    difficulty: Difficulty
    persona_name: str
    turn_evaluations: list[DebateTurnEvalItem]
    ai_competitor_turns: list[str]


class TurnHighlight(BaseModel):
    round_type: DebateRound
    highlight_type: Literal["best", "worst"]
    comment: str


class DebateSessionSummaryResponse(BaseModel):
    overall: str
    strengths: str
    improvements: str
    strategy_feedback: str
    turn_highlights: list[TurnHighlight]


# ── 리포트 (/report/debate/generate) ─────────────────────────────────────────

class DebateReportRequest(BaseModel):
    topic_title: str
    user_stance: DebateStance
    difficulty: Difficulty
    persona_name: str
    turn_evaluations: list[DebateTurnEvalItem]
    session_summary: DebateSessionSummaryResponse


class DebateWeaknessItem(BaseModel):
    item: str
    comment: str


class DebateTurnFeedback(BaseModel):
    round_type: DebateRound
    weighted_score: float
    feedback: str


class DebateReportResponse(BaseModel):
    overall: str
    strengths: str
    weaknesses: list[DebateWeaknessItem]
    improvements: str
    turn_feedback: list[DebateTurnFeedback]
    strategy_analysis: str
    recommended_topics: list[str]
    final_advice: str
    debate_readiness_comment: str


# ── 토론 주제 추천/생성 (/debate/topics/*) ────────────────────────────────────

TopicCategory = Literal["AI_ETHICS", "RECRUITMENT", "DEV_CULTURE", "TECH_TREND"]


class DebateTopicCandidate(BaseModel):
    title: str
    description: str
    category: TopicCategory


class DebateTopicSuggestRequest(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    count: int = Field(default=3, ge=1, le=5)


class DebateTopicSuggestResponse(BaseModel):
    candidates: list[DebateTopicCandidate]
    news_count: int


class DebateTopicDetailRequest(BaseModel):
    title: str
    summary: Optional[str] = None
    category: Optional[TopicCategory] = None


class DebateTopicDetailResponse(BaseModel):
    topic_title: str
    category: Optional[TopicCategory] = None
    topic_description: str
    pro_key_points: list[str]
    con_key_points: list[str]
