from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    CLOVA_SECRET_KEY: str = ""
    CLOVA_INVOKE_URL: str = ""
    SSL_CERT_PATH: str = "/app/global-bundle.pem"
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    AWS_S3_BUCKET_NAME: str = ""
    
    # ChromaDB
    CHROMADB_HOST: str = "localhost"
    CHROMADB_PORT: int = 8000

    # 임베딩 모델
    EMBEDDING_MODEL: str = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
    MODELS_DIR: str = "/app/models"
    
    # 토론 면접 모델 / 타임아웃
    OPENAI_MODEL_DEBATE: str = "gpt-4o"
    DEBATE_GENERATION_TIMEOUT: float = 120.0
    DEBATE_EVAL_TIMEOUT: float = 60.0

    # TTS (Google Cloud Text-to-Speech)
    TTS_ENABLED: bool = False
    GOOGLE_TTS_CREDENTIALS_JSON: str = ""
    DEBATE_TTS_INTERVIEWER_SPEAKER: str = "ko-KR-Neural2-A"
    DEBATE_TTS_PERSONA_SPEAKERS: str = (
        "persona_01_stable:ko-KR-Neural2-B,"
        "persona_02_aggressive:ko-KR-Neural2-C,"
        "persona_03_creative:ko-KR-Wavenet-C,"
        "persona_04_veteran:ko-KR-Wavenet-B,"
        "persona_05_nondev:ko-KR-Wavenet-A"
    )
    TTS_S3_PREFIX: str = "tts/debate"

    # 1:1 면접 TTS (면접관 페르소나별 화자)
    INTERVIEW_TTS_PERSONA_SPEAKERS: str = (
        "HR_MANAGER:ko-KR-Neural2-A,"
        "TEAM_LEAD:ko-KR-Neural2-C,"
        "EXECUTIVE:ko-KR-Wavenet-D,"
        "TECH_INTERVIEWER:ko-KR-Neural2-B"
    )
    INTERVIEW_TTS_S3_PREFIX: str = "tts/interview"

    # SQL 로그 출력 여부 (운영 환경에서는 False)
    SQL_ECHO: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

settings = Settings()
