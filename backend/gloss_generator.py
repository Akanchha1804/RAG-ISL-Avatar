"""
Gloss Generator Module.
Uses fine-tuned Flan-T5-small for English-to-ISL gloss generation.

Generation ALWAYS runs on the bare training-format prompt
"translate English to ISL: <sentence>" with NO retrieved examples injected
(see main.stage_generate). RAG influences generation only indirectly: a
high-confidence retrieval hit takes the extractive "rag" path and T5 is
never called for it. The few-shot prompt builder is kept for offline
experiments only — injecting retrieved (EN -> ISL) pairs into the prompt
made the model stitch retrieved glosses together instead of translating
the query, so it must not be used in the live pipeline.
"""

from pathlib import Path

from paths import get_paths

_paths = get_paths()
MODEL_DIR = _paths.model_dir

_model = None
_tokenizer = None
_model_available = False
# Set once a load attempt has failed so a broken directory is not re-read on
# every request (each attempt costs a ~300 MB from_pretrained).
_load_failed = False

TRAINING_PREFIX = "translate English to ISL: "


def is_model_available():
    global _model_available
    if _load_failed:
        return False
    if not _model_available and MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        _model_available = True
    return _model_available


def reset_model_state():
    """Test hook: forget any previous load failure and drop the cached model."""
    global _model, _tokenizer, _model_available, _load_failed
    _model = None
    _tokenizer = None
    _model_available = False
    _load_failed = False


def load_model():
    global _model, _tokenizer, _model_available, _load_failed

    if _model is not None:
        return True

    if _load_failed:
        return False

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print(f"[GlossGenerator] Model not found at {MODEL_DIR}")
        _model_available = False
        _load_failed = True
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
        _model = None
        _tokenizer = None
        _model_available = False
        _load_failed = True
        return False


def build_prompt(
    english_sentence: str,
    retrieved_examples: list | None = None,
    use_rag: bool = True,
) -> str:
    """Build the generator input.

    RAG on  (use_rag=True, examples present):
        Examples:
        EN: <retrieved_sentence> -> ISL: <retrieved_glosses>
        ...
        translate English to ISL: <query sentence>

    RAG off (use_rag=False or no examples):
        translate English to ISL: <query sentence>
    """
    query_line = f"{TRAINING_PREFIX}{english_sentence}"

    if not use_rag or not retrieved_examples:
        return query_line

    lines = ["Examples:"]
    for example in retrieved_examples:
        sentence = (example or {}).get("sentence", "")
        glosses = (example or {}).get("glosses", "")
        if sentence and glosses:
            lines.append(f"EN: {sentence} -> ISL: {glosses}")
    lines.append(query_line)
    return "\n".join(lines)


def generate_gloss(
    english_sentence: str,
    retrieved_examples: list | None = None,
    use_rag: bool = True,
    max_length: int | None = None,
) -> str | None:
    """Generate an ISL gloss string for one English sentence.

    Returns None when the model is unavailable or inference fails, so the
    pipeline can fall back without crashing.
    """
    if not load_model():
        return None

    try:
        import torch

        input_text = build_prompt(
            english_sentence,
            retrieved_examples=retrieved_examples,
            use_rag=use_rag,
        )

        # RAG prompts are longer; keep the query line from being truncated.
        if max_length is None:
            max_length = 256 if (use_rag and retrieved_examples) else 128

        inputs = _tokenizer(
            input_text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True,
            )

        gloss = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        return gloss.strip()
    except Exception as e:
        print(f"[GlossGenerator] Inference error: {e}")
        return None
