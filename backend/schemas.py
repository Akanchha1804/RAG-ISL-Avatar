from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class TextInput(BaseModel):
    text: str
    use_rag: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


class RetrievedExample(BaseModel):
    """One retrieved training example (same shape as GET /api/search)."""
    sentence: str
    glosses: str = ""
    landmark_file: str = ""
    similarity: float
    distance: Optional[float] = None


class TranslationResponse(BaseModel):
    input_text: str
    transcript: Optional[str] = None
    matched_sentence: Optional[str] = None
    glosses: str
    gloss_sequence: List[str] = []
    retrieved_examples: List[RetrievedExample] = []
    use_rag: bool = True
    top_k: int = 5
    landmark_file: str = ""
    similarity: Optional[float] = None
    landmark_url: str = ""
    method: str
    animation: Optional[dict] = None


class ClipItem(BaseModel):
    gloss: str
    sign_id: Optional[str] = None
    clip_id: str
    duration_ms: Optional[int] = None
    category: Optional[str] = None
    source: str
    landmark_file: Optional[str] = None


class AnimateRequest(BaseModel):
    gloss_sequence: List[str]


class AnimateResponse(BaseModel):
    clip_playlist: List[ClipItem]
    resolved_tokens: List[str]
    unresolved_tokens: List[str]


class TranslationHistoryResponse(BaseModel):
    id: str
    input_text: str
    output_glosses: Optional[str]
    matched_sentence: Optional[str]
    similarity: Optional[float]
    method: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class GlossEntry(BaseModel):
    word: str
    found: bool
    gloss: Optional[str]
    uid: Optional[str]
    category: Optional[str]
