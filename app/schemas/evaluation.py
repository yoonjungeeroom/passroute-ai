from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator


# ── 공통 ──────────────────────────────────────────────────────────────────────

class EvaluationSummary(BaseModel):
    strengths: str
    improvements: str


# ── 기술 사실 검증 (fact_check) ────────────────────────────────────────────────
# 기술 면접 답변 또는 기술적 주장이 포함된 답변에서, accuracy 점수만으로는 드러나지 않는
# "무엇이 틀렸고 / 올바른 개념은 무엇이며 / 어떻게 고쳐 말하면 되는지"를 분리해 제공한다.

class IncorrectClaim(BaseModel):
    user_claim: str             # 사용자가 한 (틀린) 기술 주장 원문 또는 요지
    issue: str                  # 무엇이 왜 틀렸는지
    correct_explanation: str    # 올바른 개념 설명
    suggested_fix: str          # 고쳐 말하는 예시 문장


class FactCheck(BaseModel):
    # 기술 검증이 필요한 답변인지 여부. 인성 답변 등 기술 주장이 없으면 False.
    is_fact_check_applicable: bool = False
    # 명백히 틀린 기술 주장만 담는다. (확실치 않으면 unsupported_claims로)
    incorrect_claims: list[IncorrectClaim] = Field(default_factory=list)
    # 근거 없이 단정했으나 틀렸다고 단정할 수는 없는 주장 (needsReview 성격)
    unsupported_claims: list[str] = Field(default_factory=list)
    # 항목이 많지 않을 때 전체 차원의 정정/예시를 담는 선택 필드. 비워도 됨.
    correct_explanation: Optional[str] = None
    suggested_fix: Optional[str] = None


# ── 질문 단위 평가 (/evaluate/question) ────────────────────────────────────────

class QuestionEvaluationRequest(BaseModel):
    job_title: str
    company_name: str
    jd_keywords: list[str]
    question_type: Literal["technical", "personality"]
    question: str
    answer: str


class ScoreItem(BaseModel):
    score: int = Field(ge=1, le=5)
    feedback: str


class ScoreItemWithWeight(BaseModel):
    score: int = Field(ge=1, le=5)
    weight: float
    feedback: str


class LLMScores(BaseModel):
    relevance: ScoreItem
    logic: ScoreItem
    specificity: ScoreItem
    conciseness: ScoreItem
    clarity: ScoreItem
    job_relevance: ScoreItem
    accuracy: Optional[ScoreItem] = None
    depth: Optional[ScoreItem] = None
    authenticity: Optional[ScoreItem] = None
    growth: Optional[ScoreItem] = None


class LLMScoresWithWeight(BaseModel):
    relevance: ScoreItemWithWeight
    logic: ScoreItemWithWeight
    specificity: ScoreItemWithWeight
    conciseness: ScoreItemWithWeight
    clarity: ScoreItemWithWeight
    job_relevance: ScoreItemWithWeight
    accuracy: Optional[ScoreItemWithWeight] = None
    depth: Optional[ScoreItemWithWeight] = None
    authenticity: Optional[ScoreItemWithWeight] = None
    growth: Optional[ScoreItemWithWeight] = None


class QuestionEvaluationResponse(BaseModel):
    llm_scores: LLMScoresWithWeight
    summary: EvaluationSummary
    # 기술 질문/기술 주장 답변일 때만 채워진다. 그 외에는 None (하위호환).
    fact_check: Optional[FactCheck] = None


# ── STAR 평가 (/evaluate/star) ────────────────────────────────────────────────

class StarEvaluationRequest(BaseModel):
    question: str
    answer: str


class StarBreakdownItem(BaseModel):
    present: bool
    feedback: str


class StarBreakdown(BaseModel):
    situation: StarBreakdownItem
    task: StarBreakdownItem
    action: StarBreakdownItem
    result: StarBreakdownItem


class StarEvaluationDetail(BaseModel):
    applicable: bool
    reason: str
    star_breakdown: Optional[StarBreakdown] = None
    star_score: Optional[int] = Field(default=None, ge=0, le=4)

    @model_validator(mode='after')
    def check_non_applicable_fields(self) -> 'StarEvaluationDetail':
        if not self.applicable:
            if self.star_breakdown is not None or self.star_score is not None:
                raise ValueError("applicable=false일 때 star_breakdown, star_score는 null이어야 합니다.")
        return self


class StarEvaluationResponse(BaseModel):
    star_evaluation: StarEvaluationDetail


# ── 세션 요약 (/evaluate/session-summary) ─────────────────────────────────────

class QuestionSummaryItem(BaseModel):
    question_index: int
    question_type: Literal["technical", "personality"]
    question: str
    percentage: float = Field(ge=0, le=100)
    summary: EvaluationSummary


class ItemAvg(BaseModel):
    avg: float = Field(ge=1, le=5)
    evaluated_count: int = Field(ge=0)


class ItemAverages(BaseModel):
    relevance: ItemAvg
    logic: ItemAvg
    specificity: ItemAvg
    conciseness: ItemAvg
    clarity: ItemAvg
    job_relevance: ItemAvg
    accuracy: Optional[ItemAvg] = None
    depth: Optional[ItemAvg] = None
    authenticity: Optional[ItemAvg] = None
    growth: Optional[ItemAvg] = None


class SessionScore(BaseModel):
    raw: float
    percentage: float = Field(ge=0, le=100)
    consistency_score: float = Field(ge=0, le=1)


class BestWorstQ(BaseModel):
    question_index: int
    question: str
    percentage: float = Field(ge=0, le=100)


class SessionSummaryRequest(BaseModel):
    job_title: str
    company_name: str
    per_question_summaries: list[QuestionSummaryItem]
    item_averages: ItemAverages
    session_score: SessionScore
    best_q: BestWorstQ
    worst_q: BestWorstQ


class QuestionHighlight(BaseModel):
    question_index: int
    type: Literal["best", "worst"]
    comment: str


class SessionSummaryOutput(BaseModel):
    overall: str
    strengths: str
    improvements: str
    question_highlights: list[QuestionHighlight]


class SessionSummaryResponse(BaseModel):
    session_summary: SessionSummaryOutput


# ── 리포트 생성 (/report/generate) ───────────────────────────────────────────

class StarEvalForReport(BaseModel):
    applicable: bool
    star_score: Optional[int] = Field(default=None, ge=0, le=4)


class QuestionEvalForReport(BaseModel):
    question_index: int
    question_type: Literal["technical", "personality"]
    question: str
    # 답변 원문(STT). 리포트가 "어떤 답변의 어느 부분이 왜 문제인지"를 근거로 들도록 전달한다.
    # 백엔드 점진 롤아웃을 위해 Optional로 두며, 없으면 기존처럼 요약 기반으로만 작성한다.
    answer: Optional[str] = None
    percentage: float = Field(ge=0, le=100)
    summary: EvaluationSummary
    star_evaluation: StarEvalForReport
    voice_feedback: Optional[str] = None


class InterviewReadiness(BaseModel):
    decision: Literal["READY", "NEEDS_REVIEW", "NEEDS_IMPROVEMENT"]
    reason: str


class SessionResultForReport(BaseModel):
    percentage: float = Field(ge=0, le=100)
    consistency_score: float = Field(ge=0, le=1)
    item_averages: ItemAverages
    key_weakness: list[str]
    interview_readiness: InterviewReadiness


class VoiceHighlightMoment(BaseModel):
    question_index: int
    timestamp_range: str
    reason: str


class VoiceHighlight(BaseModel):
    best_moment: Optional[VoiceHighlightMoment] = None
    improvement_moment: Optional[VoiceHighlightMoment] = None


class ReportGenerationRequest(BaseModel):
    job_title: str
    company_name: str
    question_evaluations: list[QuestionEvalForReport]
    session_result: SessionResultForReport
    voice_highlight: Optional[VoiceHighlight] = None


class WeaknessItem(BaseModel):
    item: str
    comment: str


class QuestionDetailedFeedback(BaseModel):
    """문항별 구조화 피드백. 점수 나열 대신 '무엇을 어떻게 고칠지'를 분리해 담는다.

    improvement_example / suggested_answer / retry_strategy는 60점 미만 문항에서 필수로
    채워지며, 그 외 문항에서는 None일 수 있다(하위호환·기본값 처리).
    """
    strength: str = ""                                      # 잘한 점 (답변 근거)
    weakness: str = ""                                      # 부족한 점 (답변의 어느 부분 때문인지)
    missing_info: list[str] = Field(default_factory=list)   # 답변에서 빠진 핵심 정보
    improvement_example: Optional[str] = None               # 다음 답변에 추가하면 좋은 예시 문장
    suggested_answer: Optional[str] = None                  # 사용자의 답변을 고쳐 쓴 예시
    retry_strategy: Optional[str] = None                    # 다음 연습에서의 답변 전략


class QuestionFeedback(BaseModel):
    question_index: int
    question: str
    question_type: Literal["technical", "personality"]
    percentage: float = Field(ge=0, le=100)
    feedback: str
    star_comment: Optional[str] = None
    voice_comment: Optional[str] = None
    # ── 신규 (nullable, 하위호환) ──────────────────────────────────────────────
    detailed_feedback: Optional[QuestionDetailedFeedback] = None
    fact_check: Optional[FactCheck] = None


class ReportGenerationResponse(BaseModel):
    overall: str
    strengths: str
    weaknesses: list[WeaknessItem]
    improvements: str
    question_feedback: list[QuestionFeedback]
    recommended_questions: list[str]
    final_advice: str
    readiness_comment: str


# ── 자소서별 종합 리포트 (/report/self-intro/generate) ─────────────────────────
# 한 자소서(=한 회사 지원)의 여러 회차 연습 세션을 종합. 통계 집계는 백엔드가 끝내고,
# 여기서는 회차 간 추세·반복 패턴을 자연어로 진단하는 종합 피드백 문단만 생성한다.

class ItemTrendItem(BaseModel):
    item: str                               # snake_case 항목 키 (relevance, depth 등)
    first_avg: float = Field(ge=0, le=5)    # 항목 평균은 5점 만점
    last_avg: float = Field(ge=0, le=5)
    direction: Literal["UP", "DOWN", "STABLE"]


class SessionBrief(BaseModel):
    round: int = Field(ge=1)                # 회차 (1부터)
    score: float = Field(ge=0, le=100)      # 세션 점수는 100점 만점
    key_weaknesses: list[str]               # 그 회차의 핵심 약점 카테고리


class SelfIntroReportRequest(BaseModel):
    job_title: str
    company_name: str
    total_sessions: int = Field(ge=1)
    overall_average: float = Field(ge=0, le=100)
    item_averages: dict[str, float]         # 전 회차 항목별 평균(0~5), 키는 snake_case
    item_trend: list[ItemTrendItem]         # 첫↔마지막 회차 항목 비교
    sessions: list[SessionBrief]            # 전 회차(round 오름차순)
    # 주의: InterviewReadiness(decision/reason)와 무관 — 단순 Literal 필드
    readiness: Literal["READY", "NEEDS_REVIEW", "NEEDS_IMPROVEMENT"]

    @model_validator(mode="after")
    def check_item_averages_scale(self) -> "SelfIntroReportRequest":
        # 항목 평균은 5점 만점 — 범위를 벗어난 값이 들어오면 거부한다.
        for k, v in self.item_averages.items():
            if not (0.0 <= v <= 5.0):
                raise ValueError(f"item_averages 값은 0~5 범위여야 합니다. ({k}: {v})")
        return self


class SelfIntroSummaryOutput(BaseModel):
    overall: str                            # 추세 진단 포함 종합 피드백 (3~5문장)
    repeated_weakness: str                  # 회차 간 반복 약점 진단
    next_steps: str                         # 다음 연습 방향 제안 (2~3문장)


class SelfIntroReportResponse(BaseModel):
    self_intro_summary: SelfIntroSummaryOutput