"""
Hindi -> English translation stage (M2).

Why this exists: the gloss pipeline (retrieval, T5, CISLR vocabulary) only
understands English, but M1 emits Hindi transcripts in Devanagari script.
Without translation, Hindi input fell through to a garbage word-by-word echo
of Hindi words that no animation source can resolve.

Approach: script-detect Devanagari -> OPUS-MT hi->en (Helsinki-NLP, ~78M
params, ~0.5s/sentence on CPU) -> the unchanged English pipeline. Retrieval
(M3) and gloss generation then operate on English with no Hindi-specific
branches anywhere else.

Scope notes:
  - Devanagari input (what faster-whisper emits for language="hi") is fully
    supported. Romanized Hinglish ("main kal market ja raha hoon") is NOT:
    the hi->en model mistranslates it, so Latin-script input stays on the
    English path (best-effort, documented gap).
  - Non-Latin, non-Devanagari scripts are reported as unsupported_language
    by the caller instead of producing silent garbage.
"""

import os
import re

MODEL_NAME = os.environ.get("MT_MODEL", "Helsinki-NLP/opus-mt-hi-en")
DISABLED = os.environ.get("MT_DISABLED", "0") == "1"

_model = None
_tokenizer = None
_load_failed = False

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")


def reset_model_state():
    """Test hook: forget load failures and drop cached objects."""
    global _model, _tokenizer, _load_failed
    _model = None
    _tokenizer = None
    _load_failed = False


def is_hindi_text(text: str) -> bool:
    """True when the text contains Devanagari script (M1 Hindi output)."""
    return bool(text and DEVANAGARI_RE.search(text))


def is_unsupported_script(text: str) -> bool:
    """True when text has letters but neither Latin nor Devanagari ones.

    Catches e.g. Tamil/Gurmukhi input that neither the English pipeline
    nor the hi->en translator can handle, so the caller can say so
    explicitly instead of echoing garbage glosses.
    """
    if not text or not text.strip():
        return False
    has_letter = bool(re.search(r"[^\W\d_]", text, re.UNICODE))
    if not has_letter:
        return False
    return not LATIN_RE.search(text) and not DEVANAGARI_RE.search(text)


def load_model() -> bool:
    global _model, _tokenizer, _load_failed
    if _model is not None:
        return True
    if _load_failed or DISABLED:
        return False
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"[MT] Loading {MODEL_NAME} (hi->en)...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        print("[MT] Model loaded.")
        return True
    except Exception as e:
        print(f"[MT] Failed to load model: {e}")
        _model = None
        _tokenizer = None
        _load_failed = True
        return False


def translate_hi_to_en(text: str) -> str | None:
    """Translate Devanagari Hindi to English. None when unavailable/failed."""
    if not text or not text.strip():
        return None
    if not load_model():
        return None
    try:
        import torch

        inputs = _tokenizer(
            [text.strip()],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            out = _model.generate(
                **inputs, max_length=128, num_beams=4, early_stopping=True
            )
        result = _tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return result or None
    except Exception as e:
        print(f"[MT] Inference error: {e}")
        return None
