"""M2: Hindi->English translation stage + English-path hardening. No real models."""

import pytest


def test_is_hindi_text():
    from mt import is_hindi_text

    assert is_hindi_text("कृपया मेरी मदद करो")
    assert is_hindi_text("hello नमस्ते")
    assert not is_hindi_text("please help me")
    assert not is_hindi_text("")
    assert not is_hindi_text("123 !?")


def test_is_unsupported_script():
    from mt import is_unsupported_script

    assert is_unsupported_script("வணக்கம்")  # Tamil
    assert not is_unsupported_script("please help me")
    assert not is_unsupported_script("कृपया मदद करो")
    assert not is_unsupported_script("")
    assert not is_unsupported_script("123 !?")


def test_translate_english_passthrough():
    from main import stage_translate

    text, meta = stage_translate("please help me", None)
    assert text == "please help me"
    assert meta is None


def test_translate_empty_passthrough():
    from main import stage_translate

    assert stage_translate("", None) == ("", None)
    assert stage_translate("   ", "hi") == ("   ", None)


def test_translate_hindi_uses_mt(monkeypatch):
    import mt
    from main import stage_translate

    monkeypatch.setattr(mt, "translate_hi_to_en", lambda t: "Please help me")
    text, meta = stage_translate("कृपया मेरी मदद करो", None)
    assert text == "Please help me"
    assert meta["detected_language"] == "hi"
    assert meta["source_text"] == "कृपया मेरी मदद करो"
    assert meta["translated_text"] == "Please help me"


def test_translate_explicit_hi_without_devanagari_not_translated(monkeypatch):
    """Romanized Hinglish must NOT go through hi->en (it mistranslates)."""
    import mt
    from main import stage_translate

    called = []
    monkeypatch.setattr(mt, "translate_hi_to_en", lambda t: called.append(t) or "garbage")
    text, meta = stage_translate("main kal market ja raha hoon", "hi")
    assert text == "main kal market ja raha hoon"
    assert called == []
    assert meta["detected_language"] == "hi-latn"


def test_translate_mt_failure_degrades_loudly(monkeypatch):
    import mt
    from main import stage_translate

    monkeypatch.setattr(mt, "translate_hi_to_en", lambda t: None)
    text, meta = stage_translate("कृपया मेरी मदद करो", "hi")
    assert text == "कृपया मेरी मदद करो"  # original flows through
    assert meta["warning"] == "translation unavailable; original text used"


def test_translate_unsupported_script():
    from main import stage_translate

    text, meta = stage_translate("வணக்கம்", None)
    assert text == ""
    assert meta["detected_language"] == "unsupported"


def test_t5_output_uppercased_for_consistency(monkeypatch):
    """T5 echo must match word_by_word casing convention (glosses UPPERCASE)."""
    import gloss_generator
    from main import stage_generate

    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: True)
    monkeypatch.setattr(
        gloss_generator, "generate_gloss",
        lambda s, retrieved_examples=None, use_rag=True, max_length=None: "my brother plays football",
    )
    out = stage_generate("my brother plays football", [], False)
    assert out["method"] == "t5"
    assert out["gloss_sequence"] == ["MY", "BROTHER", "PLAYS", "FOOTBALL"]
    assert out["glosses"] == "MY BROTHER PLAYS FOOTBALL"


def test_hindi_translate_endpoint(monkeypatch, client):
    """Devanagari text -> translated -> English pipeline (word_by_word here)."""
    import gloss_generator
    import mt

    monkeypatch.setattr(mt, "translate_hi_to_en", lambda t: "Please help me")
    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: False)

    resp = client.post("/api/translate", json={"text": "कृपया मेरी मदद करो"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["translation"]["detected_language"] == "hi"
    assert body["translation"]["translated_text"] == "Please help me"
    # M4: single verb HELP final (SOV).
    assert body["glosses"] == "PLEASE ME HELP"
    assert body["method"] == "word_by_word"


def test_unsupported_script_endpoint(client):
    resp = client.post("/api/translate", json={"text": "வணக்கம்"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "unsupported_language"
    assert body["glosses"] == ""
    assert body["translation"]["detected_language"] == "unsupported"


def test_ws_hindi_carries_translation(monkeypatch, client):
    import gloss_generator
    import mt

    monkeypatch.setattr(mt, "translate_hi_to_en", lambda t: "Please help me")
    # Keep it fast/deterministic: skip the real T5, exercise word_by_word.
    monkeypatch.setattr(gloss_generator, "is_model_available", lambda: False)

    with client.websocket_connect("/api/pipeline/ws") as ws:
        ws.send_json({"input_mode": "text", "text": "कृपया मेरी मदद करो"})
        frames = []
        for _ in range(6):
            frames.append(ws.receive_json())
            if frames[-1].get("stage") == "complete":
                break
        complete = next(f for f in frames if f.get("stage") == "complete")
        assert complete["status"] == "done"
        assert complete["result"]["translation"]["translated_text"] == "Please help me"
        assert complete["result"]["glosses"] == "PLEASE ME HELP"
