import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, DateTime, Integer
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


class SentenceClip(Base):
    """Authoritative sentence-level clip mapping (M5/Phase 3).

    Seeded from sentence_mapping.json; canonical_gloss is filled when the
    BLEU reference set locks (NULL until then — retrieval uses the live
    grammar instead, never a stale copy).
    """

    __tablename__ = "sentence_clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    text_norm = Column(Text, nullable=False, unique=True)
    text_display = Column(Text)
    glosses = Column(Text)
    canonical_gloss = Column(Text, nullable=True)
    landmark_file = Column(String(500))
    status = Column(String(20), default="ready")
    video_count = Column(Integer)
    detection_rate = Column(Float)


class SignClip(Base):
    """Authoritative per-sign inventory (M5/Phase 3).

    One row per canonical gloss token. CISLR rows carry uid/clip metadata;
    corpus-only tokens carry uid NULL + a sentence landmark reference.
    A row with no uid AND no landmark_clip is explicitly unplayable and
    surfaces in unsupported_tokens (never silently substituted).
    """

    __tablename__ = "sign_clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gloss = Column(String(100), nullable=False, unique=True)
    uid = Column(String(100), nullable=True)
    category = Column(String(100), nullable=True)
    duration = Column(Float, nullable=True)
    landmark_clip = Column(String(200), nullable=True)
    landmark_file = Column(String(500), nullable=True)
    source = Column(String(30), default="cislr")


class PhraseClip(Base):
    """Schema-ready phrase-level clips (M5/Phase 3).

    Unpopulated until M6 builds phrase assets; retrieval composes sign
    clips meanwhile. Kept so sentence- and sign-level rows never have to
    stretch into jobs they don't fit.
    """

    __tablename__ = "phrase_clips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    phrase = Column(Text, nullable=False, unique=True)
    token_count = Column(Integer)
    source = Column(String(30), default="composed")
    note = Column(Text, nullable=True)
