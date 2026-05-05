from fastapi import APIRouter

from app.schemas.follow_up import FollowUpRequest, FollowUpResponse
from app.services.follow_up import generate_follow_up

router = APIRouter(prefix="/api", tags=["follow-up"])


@router.post("/follow-up", response_model=FollowUpResponse)
async def create_follow_up(body: FollowUpRequest) -> FollowUpResponse:
    """면접 답변에 대한 꼬리 질문을 생성한다.

    모든 답변에 꼬리 질문이 생성되지는 않으며,
    AI가 답변의 깊이와 구체성을 판단하여 필요한 경우에만 생성한다.
    """
    return await generate_follow_up(request=body)
