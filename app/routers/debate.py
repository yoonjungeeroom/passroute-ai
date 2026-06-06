from fastapi import APIRouter
from app.schemas.debate import (
    InterviewerOpeningRequest, InterviewerOpeningResponse,
    InterviewerCueRequest, InterviewerCueResponse,
    DebateOpeningRequest, DebateOpeningResponse,
    DebateRebuttalRequest, DebateRebuttalResponse,
    DebateClosingRequest, DebateClosingResponse,
    InterviewerClosingRequest, InterviewerClosingResponse,
    DebateTopicSuggestRequest, DebateTopicSuggestResponse,
    DebateTopicDetailRequest, DebateTopicDetailResponse,
)
from app.services.debate_llm_service import (
    generate_interviewer_opening,
    generate_interviewer_cue,
    generate_competitor_opening,
    generate_competitor_rebuttal,
    generate_competitor_closing,
    generate_interviewer_closing,
    generate_debate_topics,
    generate_debate_topic_detail,
)

router = APIRouter(tags=["debate"])


@router.post("/debate/interviewer-opening", response_model=InterviewerOpeningResponse)
async def interviewer_opening_endpoint(req: InterviewerOpeningRequest) -> InterviewerOpeningResponse:
    return await generate_interviewer_opening(req)


@router.post("/debate/opening", response_model=DebateOpeningResponse)
async def competitor_opening_endpoint(req: DebateOpeningRequest) -> DebateOpeningResponse:
    return await generate_competitor_opening(req)


@router.post("/debate/rebuttal", response_model=DebateRebuttalResponse)
async def competitor_rebuttal_endpoint(req: DebateRebuttalRequest) -> DebateRebuttalResponse:
    return await generate_competitor_rebuttal(req)


@router.post("/debate/closing", response_model=DebateClosingResponse)
async def competitor_closing_endpoint(req: DebateClosingRequest) -> DebateClosingResponse:
    return await generate_competitor_closing(req)


@router.post("/debate/interviewer-closing", response_model=InterviewerClosingResponse)
async def interviewer_closing_endpoint(req: InterviewerClosingRequest) -> InterviewerClosingResponse:
    return await generate_interviewer_closing(req)


@router.post("/debate/interviewer-cue", response_model=InterviewerCueResponse)
async def interviewer_cue_endpoint(req: InterviewerCueRequest) -> InterviewerCueResponse:
    return await generate_interviewer_cue(req)


# ── 토론 주제 추천/생성 (크롤링 뉴스 기반) ────────────────────────────────────

@router.post("/debate/topics/suggest", response_model=DebateTopicSuggestResponse)
async def suggest_debate_topics_endpoint(req: DebateTopicSuggestRequest) -> DebateTopicSuggestResponse:
    return await generate_debate_topics(req)


@router.post("/debate/topics/detail", response_model=DebateTopicDetailResponse)
async def debate_topic_detail_endpoint(req: DebateTopicDetailRequest) -> DebateTopicDetailResponse:
    return await generate_debate_topic_detail(req)
