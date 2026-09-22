"""
Gloss Generator Module.
Uses fine-tuned Flan-T5-small for English-to-ISL gloss generation.
Falls back to RAG/word-by-word if model not available.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_DIR = BACKEND_DIR / "models" / "isl_gloss_t5"

_model = None
_tokenizer = None
_model_available = False


def is_model_available():
    global _model_available
    if not _model_available and MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        _model_available = True
    return _model_available


def load_model():
    global _model, _tokenizer, _model_available

    if _model is not None:
        return True

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print(f"[GlossGenerator] Model not found at {MODEL_DIR}")
        _model_available = False
        return False

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        print("[GlossGenerator] Loading fine-tuned Flan-T5 model...")
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))

        if torch.cuda.is_available():
            _model = _model.cuda()

        _model_available = True
        print("[GlossGenerator] Model loaded successfully.")
        return True
    except Exception as e:
        print(f"[GlossGenerator] Failed to load model: {e}")
        _model_available = False
        return False


def generate_gloss(english_sentence: str, max_length: int = 64) -> str:
    if not load_model():
        return None

    try:
        import torch

        input_text = f"translate English to ISL: {english_sentence}"
        inputs = _tokenizer(
            input_text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

        gloss = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        return gloss.strip()
    except Exception as e:
        print(f"[GlossGenerator] Inference error: {e}")
        return None
