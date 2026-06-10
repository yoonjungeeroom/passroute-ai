from fastapi import APIRouter
from app.schemas.evaluation import (
    QuestionEvaluationRequest, QuestionEvaluationResponse,
    StarEvaluationRequest, StarEvaluationResponse,
    SessionSummaryRequest, SessionSummaryResponse,
    ReportGenerationRequest, ReportGenerationResponse,
    SelfIntroReportRequest, SelfIntroReportResponse,
)
from app.schemas.debate import (
    DebateTurnEvalRequest, DebateTurnEvalResponse,
    DebateSessionSummaryRequest, DebateSessionSummaryResponse,
    DebateReportRequest, DebateReportResponse,
)
from app.services.llm_service import (
    evaluate_question,
    evaluate_star,
    generate_session_summary,
    generate_report,
    generate_self_intro_summary,
)
from app.services.debate_llm_service import (
    evaluate_debate_turn,
    generate_debate_session_summary,
    generate_debate_report,
)

router = APIRouter(tags=["evaluation"])


# ── 1대1 면접 평가 ─────────────────────────────────────────────────────────────

@router.post("/evaluate/question", response_model=QuestionEvaluationResponse)
async def evaluate_question_endpoint(req: QuestionEvaluationRequest) -> QuestionEvaluationResponse:
    return await evaluate_question(req)


@router.post("/evaluate/star", response_model=StarEvaluationResponse)
async def evaluate_star_endpoint(req: StarEvaluationRequest) -> StarEvaluationResponse:
    return await evaluate_star(req)


@router.post("/evaluate/session-summary", response_model=SessionSummaryResponse)
async def session_summary_endpoint(req: SessionSummaryRequest) -> SessionSummaryResponse:
    return await generate_session_summary(req)


@router.post("/report/generate", response_model=ReportGenerationResponse)
async def report_generate_endpoint(req: ReportGenerationRequest) -> ReportGenerationResponse:
    return await generate_report(req)


@router.post("/report/self-intro/generate", response_model=SelfIntroReportResponse)
async def self_intro_report_generate_endpoint(req: SelfIntroReportRequest) -> SelfIntroReportResponse:
    return await generate_self_intro_summary(req)


# ── 토론 면접 평가 ─────────────────────────────────────────────────────────────

@router.post("/evaluate/debate-turn", response_model=DebateTurnEvalResponse)
async def evaluate_debate_turn_endpoint(req: DebateTurnEvalRequest) -> DebateTurnEvalResponse:
    return await evaluate_debate_turn(req)


@router.post("/debate/session-summary", response_model=DebateSessionSummaryResponse)
async def debate_session_summary_endpoint(req: DebateSessionSummaryRequest) -> DebateSessionSummaryResponse:
    return await generate_debate_session_summary(req)


@router.post("/report/debate/generate", response_model=DebateReportResponse)
async def debate_report_generate_endpoint(req: DebateReportRequest) -> DebateReportResponse:
    return await generate_debate_report(req)
