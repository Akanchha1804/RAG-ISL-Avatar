"""Gloss generation branches: direct_lookup, rag, t5, word_by_word."""

import pytest


def test_exact_lookup_is_case_insensitive_on_keys(monkeypatch):
    """Capitalized mapping keys must still hit direct_lookup (3 of 101)."""
    import main

    monkeypatch.setattr(
        main, "load_mapping",
        lambda: {"No Need To Worry": {"glosses": "NO NEED WORRY",
                                      "landmark_file": "n.json"}},
    )
    from main import stage_generate

    out = stage_generate("no need to worry", [], True)
    assert out["method"] == "direct_lookup"
    assert out["matched_sentence"] == "no need to worry"
    assert out["similarity"] == 1.0


def test_exact_lookup_returns_direct_match():
    from main import stage_generate

    out = stage_generate("hello how are you", [], True)
    assert out["method"] == "direct_lookup"
    # M4: curated tokens pass through canonical ISL order
    # (HI -> NAMASTE greeting sign, WH-word HOW final).
    assert out["glosses"] == "NAMASTE ARE_YOU HOW"
    assert out["gloss_sequence"] == ["NAMASTE", "ARE_YOU", "HOW"]
    assert out["similarity"] == 1.0
    assert out["landmark_file"] == "hello_landmarks.json"
    assert out["landmark_url"] == "/landmarks/hello_landmarks.json"
    assert out["matched_sentence"] == "hello how are you"


def test_high_similarity_uses_extractive_glosses(monkeypatch):
    """Retrieved similarity >= threshold -> curated glosses, no T5 call."""
    import gloss_generator
    from main import stage_generate

    generate_called = []

    def fake_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        generate_called.append(sentence)
        return "SHOULD NOT BE USED"

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(gloss_generator, "generate_gloss", fake_generate)

    retrieved = [
        {
            "sentence": "hello how are you",
            "glosses": "HI HOW ARE_YOU",
            "landmark_file": "hello_landmarks.json",
            "similarity": 0.8,
            "distance": 0.25,
        }
    ]

    out = stage_generate("hello how are you today", retrieved, True)

    assert out["method"] == "rag"
    assert out["glosses"] == "NAMASTE ARE_YOU HOW"
    assert out["matched_sentence"] == "hello how are you"
    assert out["similarity"] == 0.8
    assert out["landmark_file"] == "hello_landmarks.json"
    assert out["landmark_url"] == "/landmarks/hello_landmarks.json"
    # extractive path returns the curated glosses, so T5 is skipped
    assert generate_called == []


def test_t5_generation_ignores_retrieved_examples(monkeypatch):
    """Below threshold, T5 runs on the bare prompt with NO examples.

    This is the regression guard for the stitched-gloss bug: the
    model was only fine-tuned on "translate English to ISL: <s>", so
    injecting few-shot examples made it echo them instead of
    translating (e.g. "hi" -> "... -> ISL: HI HOW YOU -> ISL: ...").
    """
    import gloss_generator
    from main import stage_generate

    captured = {}

    def fake_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        captured["sentence"] = sentence
        captured["retrieved"] = retrieved_examples
        captured["use_rag"] = use_rag
        return "HI"

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(gloss_generator, "generate_gloss", fake_generate)

    retrieved = [
        {
            "sentence": "hi how are you",
            "glosses": "HI HOW YOU",
            "landmark_file": "hi_how_are_you_landmarks.json",
            "similarity": 0.6,
            "distance": 0.625,
        }
    ]

    out = stage_generate("hi", retrieved, True)

    assert out["method"] == "t5"
    # M4: T5 output is canonicalized (HI -> NAMASTE greeting sign).
    assert out["glosses"] == "NAMASTE"
    assert out["landmark_file"] == ""
    assert out["matched_sentence"] is None
    assert out["similarity"] is None
    # generation must not receive the retrieved examples
    assert captured["retrieved"] is None
    assert captured["use_rag"] is False
    assert captured["sentence"] == "hi"


def test_word_by_word_when_t5_invents_tokens(monkeypatch):
    """A generation that introduces a word absent from the input is
    rejected in favour of the faithful word-by-word gloss.

    Regression test for "i am going to school" -> "I WANT TO
    SCHOOL": the model hallucinated "want", which was never in the
    input, so the correct fallback is the literal gloss - canonicalized
    ("am" is an unsigned be-verb and is dropped, "going" maps to the
    "go" sign, single verb final per M4 SOV) so every token resolves
    to a sign clip.
    """
    import gloss_generator
    from main import stage_generate

    def fake_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        return "I WANT TO SCHOOL"  # "want" is not in the input

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(gloss_generator, "generate_gloss", fake_generate)

    out = stage_generate("i am going to school", [], True)

    assert out["method"] == "word_by_word"
    # M4: be-verb dropped, GOING -> GO, single verb final (SOV).
    assert out["glosses"] == "I TO SCHOOL GO"
    assert out["gloss_sequence"] == ["I", "TO", "SCHOOL", "GO"]
    assert out["landmark_file"] == ""


def test_t5_without_rag(monkeypatch):
    import gloss_generator
    from main import stage_generate

    captured = {}

    def fake_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        captured["retrieved"] = retrieved_examples
        captured["use_rag"] = use_rag
        return "SOMETHING NEW"

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(gloss_generator, "generate_gloss", fake_generate)

    out = stage_generate("something entirely new", [], False)

    assert out["method"] == "t5"
    assert captured["use_rag"] is False
    assert captured["retrieved"] is None


def test_rag_extractive_when_model_unavailable(monkeypatch):
    """use_rag=True, T5 off, high-similarity retrieval -> method 'rag'."""
    import gloss_generator
    from main import stage_generate

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: False)

    retrieved = [
        {
            "sentence": "hello how are you",
            "glosses": "HI HOW ARE_YOU",
            "landmark_file": "hello_landmarks.json",
            "similarity": 0.9,
            "distance": 0.11,
        }
    ]

    out = stage_generate("hello how are you today", retrieved, True)

    assert out["method"] == "rag"
    assert out["glosses"] == "NAMASTE ARE_YOU HOW"
    assert out["matched_sentence"] == "hello how are you"
    assert out["similarity"] == 0.9
    assert out["landmark_file"] == "hello_landmarks.json"


def test_word_by_word_when_similarity_below_threshold(monkeypatch):
    import gloss_generator
    from main import stage_generate

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: False)

    retrieved = [
        {
            "sentence": "thank you",
            "glosses": "THANK-YOU",
            "landmark_file": "thank_you_landmarks.json",
            "similarity": 0.35,
            "distance": 1.86,
        }
    ]

    out = stage_generate("zzz completely unrelated zzz", retrieved, True)

    assert out["method"] == "word_by_word"
    assert out["gloss_sequence"] == ["ZZZ", "COMPLETELY", "UNRELATED", "ZZZ"]
    assert out["landmark_file"] == ""


def test_word_by_word_when_rag_disabled_and_no_model(monkeypatch):
    import gloss_generator
    from main import stage_generate

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: False)

    out = stage_generate("good morning friend", [], False)
    assert out["method"] == "word_by_word"
    # M4: time word fronted.
    assert out["gloss_sequence"] == ["MORNING", "GOOD", "FRIEND"]
