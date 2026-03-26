from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str
    OPENAI_API_KEY: str
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-northeast-2"
    AWS_S3_BUCKET_NAME: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        
settings = Settings()