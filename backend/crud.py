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


async def log_translation_background(**kwargs) -> None:
    """Background-task entry point for persisting a translation.

    Opens its OWN session inside the task. Sessions from request-scope
    dependencies (or `async with AsyncSessionLocal()` blocks in the endpoint)
    may already be closed by the time a BackgroundTask runs, so a session must
    never be passed in from the endpoint.

    Database failure is non-fatal for the academic demo: errors are logged
    and swallowed so the API response is unaffected.
    """
    try:
        from database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            await log_translation(session, **kwargs)
    except Exception as e:
        print(f"[DB] Translation log skipped (non-fatal): {e}")


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
