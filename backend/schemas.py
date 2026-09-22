from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class TextInput(BaseModel):
    text: str
    use_rag: bool = True
    top_k: int = 5


class TranslationResponse(BaseModel):
    input_text: str
    transcript: Optional[str] = None
    matched_sentence: Optional[str] = None
    glosses: str
    landmark_file: str
    similarity: Optional[float] = None
    landmark_url: str
    method: str


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
