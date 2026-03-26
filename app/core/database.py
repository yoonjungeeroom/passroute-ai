from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings
import ssl

# SSL 설정 (RDS 연결용)
ssl_ctx = ssl.create_default_context(cafile="global-bundle.pem")

# 엔진 생성
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=True,              # SQL 로그 출력 (개발용, 배포 시 False)
    pool_size=10,           # 커넥션 풀 사이즈
    max_overflow=20,        # 최대 초과 커넥션
    pool_pre_ping=True,     # 연결 끊겼을 때 자동 재연결
    connect_args={"ssl": ssl_ctx}
)

# 세션 팩토리
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

# DB 세션 의존성 주입
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise