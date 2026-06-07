from sqlalchemy import BigInteger, Column, Float, Integer, String
from app.core.database import Base


class VoiceAnalysis(Base):
    __tablename__ = "voice_analysis"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    question_id = Column(String(255), nullable=False)
    avg_wpm = Column(Float)
    avg_silence_duration = Column(Float)
    filler_count = Column(Integer)
