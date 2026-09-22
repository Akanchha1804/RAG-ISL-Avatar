from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from models import TranslationHistory


async def log_translation(
    db: AsyncSession,
    input_text: str,
    output_glosses: str = None,
    matched_sentence: str = None,
    similarity: float = None,
    landmark_file: str = None,
    method: str = None,
    stt_transcript: str = None,
    stt_language: str = None,
    stt_duration: float = None,
):
    record = TranslationHistory(
        input_text=input_text,
        output_glosses=output_glosses,
        matched_sentence=matched_sentence,
        similarity=similarity,
        landmark_file=landmark_file,
        method=method,
        stt_transcript=stt_transcript,
        stt_language=stt_language,
        stt_duration=stt_duration,
    )
    db.add(record)
    await db.commit()
    return record


async def get_translation_history(db: AsyncSession, limit: int = 50, offset: int = 0):
    result = await db.execute(
        select(TranslationHistory)
        .order_by(TranslationHistory.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


async def get_translation_count(db: AsyncSession):
    result = await db.execute(select(func.count(TranslationHistory.id)))
    return result.scalar()
