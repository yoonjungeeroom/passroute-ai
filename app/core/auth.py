from fastapi import Depends, HTTPException, Header
from jose import jwt, JWTError

SECRET_KEY = "Spring Boot랑 똑같은 시크릿 키"

async def get_current_user(authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload.get("sub"))
        return user_id
    except JWTError:
        raise HTTPException(status_code=401, detail="인증 실패")