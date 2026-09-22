"""
Speech-to-Text module using Distil-Whisper via faster-whisper.
"""

from faster_whisper import WhisperModel
from pathlib import Path

_model = None
MODEL_SIZE = "distil-small.en"


def get_model():
    global _model
    if _model is None:
        print(f"Loading Distil-Whisper model: {MODEL_SIZE}")
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("Model loaded.")
    return _model


def transcribe(audio_path: str) -> dict:
    model = get_model()
    segments, info = model.transcribe(audio_path, beam_size=5)

    transcript_parts = []
    for segment in segments:
        transcript_parts.append(segment.text.strip())

    transcript = " ".join(transcript_parts)

    return {
        "transcript": transcript,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2)
    }


def transcribe_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    import tempfile
    import os

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, filename)

    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)

    try:
        result = transcribe(tmp_path)
    finally:
        os.remove(tmp_path)
        os.rmdir(tmp_dir)

    return result
