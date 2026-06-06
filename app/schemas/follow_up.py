from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class QATurn(BaseModel):
    """대화 이력의 한 턴(질문 + 답변)."""

    question: str
    answer: str


class FollowUpRequest(BaseModel):
    """꼬리 질문 생성 요청.

    conversation을 리스트로 받아 다단계 꼬리 질문 확장이 가능하다.
    conversation[-1]이 현재 턴이며, 이전 인덱스는 꼬리 질문 이력이다.
    user_id가 있으면 ChromaDB에서 이력서를 검색하여 꼬리 질문에 활용한다.
    """

    interview_type: Literal["technical", "personality"]
    difficulty: Literal["low", "middle", "high"]
    conversation: list[QATurn] = Field(min_length=1)
    user_id: Union[str, int, None] = None
    persona: Optional[
        Literal["HR_MANAGER", "TEAM_LEAD", "EXECUTIVE", "TECH_INTERVIEWER"]
    ] = None

    @field_validator("user_id", mode="before")
    @classmethod
    def coerce_user_id_to_str(cls, v):
        if v is not None:
            return str(v)
        return v


class FollowUpResponse(BaseModel):
    """꼬리 질문 생성 응답.

    has_follow_up이 False이면 follow_up_question은 None이다.
    """

    has_follow_up: bool
    follow_up_question: str | None = None
    reason: str | None = None
    audio_url: str | None = None
