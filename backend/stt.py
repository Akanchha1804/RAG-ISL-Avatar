"""
Speech-to-Text module using faster-whisper (M1).

Multilingual: English + Hindi/Hinglish via auto language detection.
Default model "small" supports both (distil-small.en was English-only).

PyAV (bundled with faster-whisper) decodes webm/mp4/ogg/wav directly,
so no system ffmpeg is required. VAD filters silence/noise.

Contract (frozen for M2-M7):
    transcribe* -> {transcript, language, language_probability, duration}
    language=None (auto-detect) or ISO code "en"/"hi" to force.
"""

import os
import re
import shutil
import tempfile
import threading

from pathlib import Path

_model = None
_model_lock = threading.Lock()
# Multilingual default: "large-v3-turbo" (~800M params, int8 CPU).
# Benchmarked on Hindi (FLEURS hi_in, clean read speech): mean WER 0.26 vs
# 0.46 for "small" — small's near-miss garbling broke the hi->en stage.
# Cost: ~3x slower inference (~20s for a 5-9s clip on CPU, well inside the
# 180s frontend guard). Override without code edits:
# STT_MODEL_SIZE=small (faster, worse Hindi) / distil-large-v3 / large-v3.
MODEL_SIZE = os.environ.get("STT_MODEL_SIZE", "large-v3-turbo")
_DEVICE = os.environ.get("STT_DEVICE", "cpu")
_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8")

# 25 MB upload guard: ~5 min of opus/webm; larger files are rejected
# before touching the model so one upload cannot OOM the CPU worker.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

_WS_COLLAPSE = re.compile(r"\s+")


def get_model():
    global _model
    if _model is not None:
        return _model
    # Startup warmup and the first request race on a cold server; without
    # the lock both load a second ~800MB copy into RAM.
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel

            print(f"Loading faster-whisper model: {MODEL_SIZE} "
                  f"(device={_DEVICE}, compute={_COMPUTE_TYPE})")
            _model = WhisperModel(MODEL_SIZE, device=_DEVICE, compute_type=_COMPUTE_TYPE)
            print("Model loaded.")
    return _model


def reset_model_state():
    """Test hook: drop the cached model so tests can inject a fake."""
    global _model
    _model = None


def _clean_transcript(parts: list) -> str:
    """Join segment texts, collapse whitespace, preserve script (Latin/Devanagari)."""
    text = " ".join(p.strip() for p in parts if p and p.strip())
    return _WS_COLLAPSE.sub(" ", text).strip()


def transcribe(audio_path: str, language: str | None = None) -> dict:
    """Transcribe an audio file. language=None auto-detects (en/hi/Hinglish)."""
    model = get_model()
    lang = (language or "").strip().lower() or None
    try:
        segments, info = model.transcribe(
            audio_path,
            language=lang,
            beam_size=5,
            vad_filter=True,
        )
    except Exception as exc:
        # VAD (onnxruntime/Silero) occasionally rejects odd containers;
        # retry once without it rather than failing the request.
        msg = str(exc).lower()
        if "vad" in msg or "onnx" in msg:
            segments, info = model.transcribe(
                audio_path, language=lang, beam_size=5, vad_filter=False
            )
        else:
            raise

    transcript_parts = []
    logprobs = []
    for segment in segments:
        transcript_parts.append(segment.text.strip())
        try:
            logprobs.append(float(segment.avg_logprob))
        except (AttributeError, TypeError, ValueError):
            pass

    import math

    # exp(mean logprob) in (0, 1]: ~0.9 confident, <0.5 shaky. Display-only
    # signal so the UI can ask for a re-record instead of cascading garbage
    # through translation -> glosses (M2 Hindi failure mode).
    confidence = (
        round(math.exp(sum(logprobs) / len(logprobs)), 3) if logprobs else 0.0
    )

    return {
        "transcript": _clean_transcript(transcript_parts),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "transcript_confidence": confidence,
    }


def transcribe_bytes(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: str | None = None,
) -> dict:
    if not audio_bytes:
        raise ValueError("Empty audio upload")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise ValueError(
            f"Audio too large ({len(audio_bytes)} bytes, max {MAX_AUDIO_BYTES})"
        )

    # The filename comes from the client: strip any directory components so a
    # crafted name such as "../../x.wav" cannot escape the temp directory.
    safe_name = os.path.basename(str(filename or "").replace("\\", "/")) or "audio.wav"
    # Keep a real audio extension so the decoder picks the right demuxer;
    # default to .wav when the client sends an extensionless/garbage name.
    suffix = Path(safe_name).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac"}:
        safe_name = "audio.wav"

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, safe_name)

    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        result = transcribe(tmp_path, language=language)
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except OSError:
            pass

    return result
