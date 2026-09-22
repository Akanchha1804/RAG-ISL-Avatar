import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, DateTime
from sqlalchemy.dialects.postgresql import UUID
from database import Base


class TranslationHistory(Base):
    __tablename__ = "translation_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_text = Column(Text, nullable=False)
    output_glosses = Column(Text)
    matched_sentence = Column(Text)
    similarity = Column(Float)
    landmark_file = Column(String(500))
    method = Column(String(50))
    stt_transcript = Column(Text)
    stt_language = Column(String(20))
    stt_duration = Column(Float)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
