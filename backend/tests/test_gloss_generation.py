"""Gloss generation branches: direct_lookup, t5_rag, t5, rag, word_by_word."""

import pytest


def test_exact_lookup_returns_direct_match():
    from main import stage_generate

    out = stage_generate("hello how are you", [], True)
    assert out["method"] == "direct_lookup"
    assert out["glosses"] == "HI HOW ARE_YOU"
    assert out["gloss_sequence"] == ["HI", "HOW", "ARE_YOU"]
    assert out["similarity"] == 1.0
    assert out["landmark_file"] == "hello_landmarks.json"
    assert out["landmark_url"] == "/landmarks/hello_landmarks.json"
    assert out["matched_sentence"] == "hello how are you"


def test_t5_rag_receives_retrieved_examples(monkeypatch):
    import gloss_generator
    from main import stage_generate

    captured = {}

    def fake_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        captured["sentence"] = sentence
        captured["retrieved"] = retrieved_examples
        captured["use_rag"] = use_rag
        return "HI HOW ARE_YOU"

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

    assert out["method"] == "t5_rag"
    assert out["glosses"] == "HI HOW ARE_YOU"
    assert captured["use_rag"] is True
    assert captured["retrieved"] == retrieved
    # extractive: top similarity above threshold attaches landmark file
    assert out["landmark_file"] == "hello_landmarks.json"


def test_t5_without_rag(monkeypatch):
    import gloss_generator
    from main import stage_generate

    captured = {}

    def fake_generate(sentence, retrieved_examples=None, use_rag=True, max_length=None):
        captured["retrieved"] = retrieved_examples
        captured["use_rag"] = use_rag
        return "WOW NICE"

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(gloss_generator, "generate_gloss", fake_generate)

    out = stage_generate("something entirely new", [], False)

    assert out["method"] == "t5"
    assert captured["use_rag"] is False
    assert captured["retrieved"] == []


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
    assert out["glosses"] == "HI HOW ARE_YOU"
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
    assert out["gloss_sequence"] == ["GOOD", "MORNING", "FRIEND"]
