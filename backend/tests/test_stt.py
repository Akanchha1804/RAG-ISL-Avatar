"""M1 STT: multilingual transcribe, guards, language override. No real models."""

import pytest


class _Seg:
    def __init__(self, text, avg_logprob=-0.1):
        self.text = text
        self.avg_logprob = avg_logprob


class _Info:
    def __init__(self, language="en", prob=0.95, duration=1.23):
        self.language = language
        self.language_probability = prob
        self.duration = duration


class _FakeModel:
    def __init__(self, calls, lang="en", texts=("hello world",), logprob=-0.1):
        self.calls = calls
        self.lang = lang
        self.texts = texts
        self.logprob = logprob

    def transcribe(self, path, **kwargs):
        self.calls.append({"path": path, "kwargs": kwargs})
        return ([_Seg(t, self.logprob) for t in self.texts], _Info(self.lang))


def _patch_model(monkeypatch, lang="en", texts=("hello world",), logprob=-0.1):
    import stt

    calls = []
    monkeypatch.setattr(stt, "_model", _FakeModel(calls, lang, texts, logprob))
    # get_model returns the fake without importing faster_whisper
    monkeypatch.setattr(stt, "get_model", lambda: stt._model)
    return calls


def test_transcribe_autodetect_by_default(monkeypatch):
    import stt

    calls = _patch_model(monkeypatch, lang="hi", texts=("  नमस्ते   आप कैसे हो ",))
    out = stt.transcribe("dummy.webm")
    assert out["transcript"] == "नमस्ते आप कैसे हो"  # whitespace collapsed, script kept
    assert out["language"] == "hi"
    assert out["duration"] == 1.23
    assert calls[0]["kwargs"].get("language") is None
    assert calls[0]["kwargs"].get("vad_filter") is True


def test_transcribe_language_override(monkeypatch):
    import stt

    calls = _patch_model(monkeypatch)
    stt.transcribe("dummy.wav", language="hi")
    assert calls[0]["kwargs"]["language"] == "hi"


def test_transcribe_reports_confidence(monkeypatch):
    import math
    import stt

    _patch_model(monkeypatch, texts=("hello",))
    out = stt.transcribe("dummy.wav")
    assert out["transcript_confidence"] == round(math.exp(-0.1), 3)


def test_transcribe_low_confidence_when_uncertain(monkeypatch):
    import stt

    _patch_model(monkeypatch, texts=("mumble",), logprob=-1.5)
    out = stt.transcribe("dummy.wav")
    assert out["transcript"] == "mumble"
    assert out["transcript_confidence"] < 0.5


def test_transcribe_vad_fallback(monkeypatch):
    """Odd containers that break Silero VAD retry once without VAD."""
    import stt

    class _VadBreaker:
        def transcribe(self, path, **kwargs):
            if kwargs.get("vad_filter"):
                raise RuntimeError("vad: onnx failed")
            return ([_Seg("hi",)], _Info("en"))

    monkeypatch.setattr(stt, "_model", _VadBreaker())
    monkeypatch.setattr(stt, "get_model", lambda: stt._model)
    out = stt.transcribe("dummy.ogg")
    assert out["transcript"] == "hi"


def test_transcribe_bytes_empty_rejected():
    import stt

    with pytest.raises(ValueError, match="Empty"):
        stt.transcribe_bytes(b"", "audio.webm")


def test_transcribe_bytes_oversize_rejected(monkeypatch):
    import stt

    monkeypatch.setattr(stt, "MAX_AUDIO_BYTES", 10)
    with pytest.raises(ValueError, match="too large"):
        stt.transcribe_bytes(b"x" * 11, "audio.wav")


def test_transcribe_bytes_path_traversal_safe(monkeypatch, tmp_path):
    import stt

    calls = _patch_model(monkeypatch)
    out = stt.transcribe_bytes(b"fake-audio", "../../evil.wav")
    assert out["transcript"] == "hello world"
    # traversal stripped: model never sees a path outside the temp dir
    assert "evil.wav" in str(calls[0]["path"])
    assert ".." not in str(calls[0]["path"])


def test_transcribe_bytes_bad_extension_defaults_wav(monkeypatch):
    import stt

    calls = _patch_model(monkeypatch)
    stt.transcribe_bytes(b"fake-audio", "noextension")
    assert str(calls[0]["path"]).endswith(".wav")


def test_transcribe_bytes_passes_language(monkeypatch):
    import stt

    calls = _patch_model(monkeypatch, lang="hi")
    out = stt.transcribe_bytes(b"fake-audio", "audio.webm", language="hi")
    assert out["language"] == "hi"
    assert calls[0]["kwargs"]["language"] == "hi"


def test_transcribe_endpoint_empty_400(client):
    resp = client.post("/api/transcribe", files={"file": ("a.webm", b"", "audio/webm")})
    assert resp.status_code == 400


def test_transcribe_endpoint_uses_stt(monkeypatch, client):
    import main

    async def fake_run(fn, *args):
        return {"transcript": "hello", "language": "en",
                "language_probability": 0.9, "duration": 1.0}

    monkeypatch.setattr(main, "_run_sync", fake_run)
    resp = client.post("/api/transcribe", files={"file": ("a.webm", b"123", "audio/webm")})
    assert resp.status_code == 200
    assert resp.json()["transcript"] == "hello"


def test_pipeline_ws_speech_language_passthrough(monkeypatch, client):
    """WS speech mode forwards optional language to stage_transcribe."""
    import base64
    import main

    seen = {}

    async def fake_stage(text, audio_bytes, filename="audio.wav", language=None):
        seen.update({"audio": audio_bytes, "language": language, "file": filename})
        return ("hello", {"transcript": "hello", "language": language or "en"})

    monkeypatch.setattr(main, "stage_transcribe", fake_stage)
    monkeypatch.setattr(main, "stage_retrieve", lambda *a, **k: [])
    monkeypatch.setattr(main, "stage_generate",
                         lambda *a, **k: {"glosses": "HELLO", "gloss_sequence": ["HELLO"],
                                          "method": "word_by_word", "matched_sentence": None,
                                          "similarity": None, "landmark_file": "", "landmark_url": ""})

    async def fake_animation(seq, sentence_hit=None):
        return {"clip_playlist": [], "resolved_tokens": [],
                "unresolved_tokens": list(seq)}

    monkeypatch.setattr(main, "stage_resolve_animation", fake_animation)

    with client.websocket_connect("/api/pipeline/ws") as ws:
        ws.send_json({"input_mode": "speech",
                      "audio_base64": base64.b64encode(b"abc").decode(),
                      "language": "hi"})
        frames = []
        for _ in range(6):
            frames.append(ws.receive_json())
        complete = next(f for f in frames if f.get("stage") == "complete")
        assert complete["status"] == "done"
        assert seen["language"] == "hi"
        assert seen["audio"] == b"abc"
