import json
import os
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "mysql+aiomysql://test:test@localhost/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("OPENAI_API_KEY", "test")


def _make_report_request():
    from app.schemas.evaluation import (
        EvaluationSummary,
        InterviewReadiness,
        ItemAvg,
        ItemAverages,
        QuestionEvalForReport,
        ReportGenerationRequest,
        SessionResultForReport,
        StarEvalForReport,
    )

    item_avgs = ItemAverages(
        relevance=ItemAvg(avg=3.0, evaluated_count=1),
        logic=ItemAvg(avg=3.0, evaluated_count=1),
        specificity=ItemAvg(avg=3.0, evaluated_count=1),
        conciseness=ItemAvg(avg=3.0, evaluated_count=1),
        clarity=ItemAvg(avg=3.0, evaluated_count=1),
        job_relevance=ItemAvg(avg=3.0, evaluated_count=1),
    )
    return ReportGenerationRequest(
        job_title="백엔드 개발자",
        company_name="테스트회사",
        question_evaluations=[
            QuestionEvalForReport(
                question_index=0,
                question_type="technical",
                question="JWT란?",
                percentage=80.0,
                summary=EvaluationSummary(strengths="명확함", improvements="깊이 부족"),
                star_evaluation=StarEvalForReport(applicable=False, star_score=None),
            )
        ],
        session_result=SessionResultForReport(
            percentage=80.0,
            consistency_score=0.9,
            item_averages=item_avgs,
            key_weakness=["depth"],
            interview_readiness=InterviewReadiness(decision="NEEDS_REVIEW", reason="보완 필요"),
        ),
    )


@pytest.mark.asyncio
async def test_report_skips_malformed_items_and_fills_defaults():
    """망가진 리스트 항목은 스킵, 누락된 텍스트 필드는 기본값으로 채워 리포트가 생성된다."""
    from app.services import llm_service

    # question_feedback: 하나는 정상, 하나는 question_type Literal 위반(스킵돼야 함)
    # weaknesses: 하나는 정상, 하나는 키 누락(스킵돼야 함)
    # overall/final_advice 등 일부 텍스트 필드는 누락 → 기본값 "" 로 채워져야 함
    fake_raw = {
        "strengths": "강점",
        "improvements": "개선점",
        "weaknesses": [
            {"item": "depth", "comment": "깊이 부족"},
            {"item": "broken"},  # comment 누락 → 스킵
        ],
        "question_feedback": [
            {
                "question_index": 0,
                "question": "JWT란?",
                "question_type": "technical",
                "percentage": 80.0,
                "feedback": "좋음",
            },
            {
                "question_index": 1,
                "question": "깨진 항목",
                "question_type": "",  # Literal 위반 → 스킵
                "percentage": 50.0,
                "feedback": "x",
            },
        ],
        "recommended_questions": ["q1", "q2", "q3"],
        # overall, final_advice, readiness_comment 누락
    }

    with patch.object(llm_service, "_call_llm_json", new=AsyncMock(return_value=fake_raw)):
        result = await llm_service.generate_report(_make_report_request())

    # 정상 항목만 남고 망가진 항목은 스킵
    assert len(result.weaknesses) == 1
    assert result.weaknesses[0].item == "depth"
    assert len(result.question_feedback) == 1
    assert result.question_feedback[0].question_index == 0
    # 누락 텍스트 필드는 기본값
    assert result.overall == ""
    assert result.final_advice == ""
    assert result.readiness_comment == ""
    assert result.strengths == "강점"


@pytest.mark.asyncio
async def test_report_uses_report_timeout():
    """리포트 호출은 전역 60초가 아니라 _REPORT_TIMEOUT을 명시적으로 넘긴다."""
    from app.services import llm_service

    mock = AsyncMock(return_value={
        "overall": "o", "strengths": "s", "improvements": "i",
        "weaknesses": [], "question_feedback": [],
        "recommended_questions": [], "final_advice": "f", "readiness_comment": "r",
    })
    with patch.object(llm_service, "_call_llm_json", new=mock):
        await llm_service.generate_report(_make_report_request())

    assert mock.await_args.kwargs["timeout"] == llm_service._REPORT_TIMEOUT
