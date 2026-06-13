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
async def test_report_handles_null_list_and_text_fields():
    """LLM이 리스트/텍스트 필드를 null로 반환해도 크래시 없이 기본값으로 리포트가 생성된다."""
    from app.services import llm_service

    fake_raw = {
        "overall": None,
        "strengths": None,
        "improvements": None,
        "weaknesses": None,          # null 리스트 → for 루프 TypeError 방지
        "question_feedback": None,   # null 리스트
        "recommended_questions": None,
        "final_advice": None,
        "readiness_comment": None,
    }

    with patch.object(llm_service, "_call_llm_json", new=AsyncMock(return_value=fake_raw)):
        result = await llm_service.generate_report(_make_report_request())

    assert result.overall == ""
    assert result.weaknesses == []
    assert result.question_feedback == []
    assert result.recommended_questions == []
    assert result.readiness_comment == ""


@pytest.mark.asyncio
async def test_evaluate_question_null_llm_scores_returns_clean_500():
    """llm_scores가 null로 와도 AttributeError가 아니라 명확한 HTTPException(500)으로 수렴한다."""
    from fastapi import HTTPException
    from app.schemas.evaluation import QuestionEvaluationRequest
    from app.services import llm_service

    req = QuestionEvaluationRequest(
        job_title="백엔드", company_name="테스트", jd_keywords=["JWT"],
        question_type="technical", question="JWT란?", answer="토큰 기반 인증",
    )
    fake_raw = {"llm_scores": None, "summary": None}

    with patch.object(llm_service, "_call_llm_json", new=AsyncMock(return_value=fake_raw)):
        with pytest.raises(HTTPException) as exc:
            await llm_service.evaluate_question(req)
    assert exc.value.status_code == 500


@pytest.mark.asyncio
async def test_report_parses_detailed_feedback_and_fact_check():
    """문항별 detailed_feedback / fact_check가 정상 파싱되어 응답에 실린다."""
    from app.services import llm_service

    fake_raw = {
        "overall": "o", "strengths": "s", "improvements": "i",
        "weaknesses": [], "recommended_questions": [],
        "final_advice": "f", "readiness_comment": "r",
        "question_feedback": [
            {
                "question_index": 0,
                "question": "GET과 POST의 차이는?",
                "question_type": "technical",
                "percentage": 45.0,
                "feedback": "방향이 반대로 설명됨",
                "detailed_feedback": {
                    "strength": "용어는 알고 있음",
                    "weakness": "GET/POST 용도를 반대로 설명",
                    "missing_info": ["멱등성", "캐시 가능 여부"],
                    "improvement_example": "GET은 조회, POST는 생성에 사용한다고 설명",
                    "suggested_answer": "GET은 리소스 조회, POST는 리소스 생성에 사용합니다.",
                    "retry_strategy": "메서드별 용도와 멱등성을 먼저 정리하고 답하기",
                },
                "fact_check": {
                    "is_fact_check_applicable": True,
                    "incorrect_claims": [
                        {
                            "user_claim": "GET은 생성, POST는 조회",
                            "issue": "용도가 반대",
                            "correct_explanation": "GET은 조회, POST는 생성",
                            "suggested_fix": "GET은 조회할 때, POST는 생성할 때 사용한다고 설명",
                        }
                    ],
                    "unsupported_claims": [],
                },
            }
        ],
    }

    with patch.object(llm_service, "_call_llm_json", new=AsyncMock(return_value=fake_raw)):
        result = await llm_service.generate_report(_make_report_request())

    qf = result.question_feedback[0]
    assert qf.detailed_feedback is not None
    assert qf.detailed_feedback.missing_info == ["멱등성", "캐시 가능 여부"]
    assert qf.detailed_feedback.suggested_answer.startswith("GET은 리소스 조회")
    assert qf.fact_check is not None
    assert qf.fact_check.is_fact_check_applicable is True
    assert qf.fact_check.incorrect_claims[0].correct_explanation == "GET은 조회, POST는 생성"


@pytest.mark.asyncio
async def test_report_malformed_nested_fields_degrade_to_none():
    """detailed_feedback / fact_check가 깨져도 문항은 살아남고 해당 필드만 None이 된다."""
    from app.services import llm_service

    fake_raw = {
        "overall": "o", "strengths": "s", "improvements": "i",
        "weaknesses": [], "recommended_questions": [],
        "final_advice": "f", "readiness_comment": "r",
        "question_feedback": [
            {
                "question_index": 0,
                "question": "JWT란?",
                "question_type": "technical",
                "percentage": 70.0,
                "feedback": "좋음",
                "detailed_feedback": "문자열로 잘못 옴",        # dict 아님 → None
                "fact_check": {
                    "is_fact_check_applicable": True,
                    "incorrect_claims": ["dict가 아닌 항목"],    # 스킵됨
                    "unsupported_claims": None,                  # None 허용
                },
            }
        ],
    }

    with patch.object(llm_service, "_call_llm_json", new=AsyncMock(return_value=fake_raw)):
        result = await llm_service.generate_report(_make_report_request())

    assert len(result.question_feedback) == 1
    qf = result.question_feedback[0]
    assert qf.detailed_feedback is None
    # fact_check 자체는 살아있되 깨진 claim은 스킵된다.
    assert qf.fact_check is not None
    assert qf.fact_check.incorrect_claims == []
    assert qf.fact_check.unsupported_claims == []


def test_build_fact_check_string_list_fields_do_not_split_into_chars():
    """unsupported_claims가 문자열로 와도 글자 단위로 쪼개지지 않고 []가 된다."""
    from app.services.llm_service import _build_fact_check

    fc = _build_fact_check({
        "is_fact_check_applicable": True,
        "incorrect_claims": [],
        "unsupported_claims": "None",   # 리스트 아님 → [] 로 처리되어야 함
    })
    assert fc is not None
    assert fc.unsupported_claims == []


def test_build_fact_check_invalid_claim_degrades_to_none():
    """claim 필드가 검증 불가 타입이면 fact_check 전체가 None으로 격하되고 예외가 새지 않는다."""
    from app.services.llm_service import _build_fact_check

    fc = _build_fact_check({
        "is_fact_check_applicable": True,
        # user_claim에 dict가 들어와 str 검증 실패 → helper 밖으로 예외가 나가면 안 됨
        "incorrect_claims": [{"user_claim": {"nested": "obj"}, "issue": "x",
                              "correct_explanation": "y", "suggested_fix": "z"}],
    })
    assert fc is None


def test_build_detailed_feedback_string_missing_info_does_not_split_into_chars():
    """missing_info가 문자열로 와도 글자 단위로 쪼개지지 않고 []가 된다."""
    from app.services.llm_service import _build_detailed_feedback

    df = _build_detailed_feedback({
        "strength": "s",
        "weakness": "w",
        "missing_info": "N/A",   # 리스트 아님 → []
    })
    assert df is not None
    assert df.missing_info == []


@pytest.mark.asyncio
async def test_evaluate_question_parses_fact_check():
    """기술 질문 평가 응답에 fact_check가 실린다."""
    from app.schemas.evaluation import QuestionEvaluationRequest
    from app.services import llm_service

    def _score(s):
        return {"score": s, "feedback": "근거"}

    fake_raw = {
        "llm_scores": {
            "relevance": _score(3), "logic": _score(3), "specificity": _score(3),
            "conciseness": _score(3), "clarity": _score(3), "job_relevance": _score(3),
            "accuracy": _score(2), "depth": _score(3),
        },
        "summary": {"strengths": "용어 인지", "improvements": "정확성 보완 필요"},
        "fact_check": {
            "is_fact_check_applicable": True,
            "incorrect_claims": [
                {"user_claim": "GET은 생성", "issue": "반대", "correct_explanation": "GET은 조회", "suggested_fix": "GET은 조회에 사용"}
            ],
            "unsupported_claims": ["근거 없는 단정"],
        },
    }
    req = QuestionEvaluationRequest(
        job_title="백엔드", company_name="테스트", jd_keywords=["HTTP"],
        question_type="technical", question="GET과 POST 차이?", answer="GET은 생성, POST는 조회",
    )

    with patch.object(llm_service, "_call_llm_json", new=AsyncMock(return_value=fake_raw)):
        result = await llm_service.evaluate_question(req)

    assert result.fact_check is not None
    assert result.fact_check.is_fact_check_applicable is True
    assert result.fact_check.incorrect_claims[0].suggested_fix == "GET은 조회에 사용"
    assert result.fact_check.unsupported_claims == ["근거 없는 단정"]


@pytest.mark.asyncio
async def test_evaluate_question_without_fact_check_is_none():
    """fact_check가 없으면(인성 질문 등) None으로 안전하게 처리된다."""
    from app.schemas.evaluation import QuestionEvaluationRequest
    from app.services import llm_service

    def _score(s):
        return {"score": s, "feedback": "근거"}

    fake_raw = {
        "llm_scores": {
            "relevance": _score(4), "logic": _score(4), "specificity": _score(4),
            "conciseness": _score(4), "clarity": _score(4), "job_relevance": _score(4),
            "authenticity": _score(4), "growth": _score(4),
        },
        "summary": {"strengths": "진정성", "improvements": "구체화"},
        # fact_check 없음
    }
    req = QuestionEvaluationRequest(
        job_title="백엔드", company_name="테스트", jd_keywords=["협업"],
        question_type="personality", question="갈등 경험은?", answer="조율했습니다",
    )

    with patch.object(llm_service, "_call_llm_json", new=AsyncMock(return_value=fake_raw)):
        result = await llm_service.evaluate_question(req)

    assert result.fact_check is None


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
