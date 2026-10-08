"""Phase 3: hierarchical retrieval, no-substitution, store fallback. No live DB."""

import pytest


def _entry(gloss, clip=True, landmark_clip=None):
    return {
        "gloss": gloss,
        "sign_id": "uid" if clip else None,
        "clip_id": "uid" if clip else f"corpus::{gloss.lower()}",
        "clip_url": "/clips/uid.mp4" if clip else None,
        "duration_ms": 800 if clip else None,
        "category": "test",
        "source": "cislr_vocabulary" if clip else "sentence_corpus",
        "landmark_file": "s.json" if not clip else None,
        "landmark_clip_url": f"/landmarks/{landmark_clip}" if landmark_clip else None,
    }


def _stub_store(monkeypatch, entries):
    import clip_store

    async def fake_sign(token):
        return entries.get(token)

    async def fake_index():
        return []

    monkeypatch.setattr(clip_store, "get_sign", fake_sign)
    monkeypatch.setattr(clip_store, "corpus_canonical_index", fake_index)

    async def _fake_name():
        return "stub"

    monkeypatch.setattr(clip_store, "store_name", _fake_name)


def test_sentence_level_covers_all_tokens(monkeypatch):
    import asyncio
    import retrieval

    entries = {"I": _entry("I"), "YOU": _entry("YOU"),
               "HELP": _entry("HELP", clip=False)}
    import clip_store
    _stub_store(monkeypatch, entries)

    out = asyncio.run(retrieval.resolve_playlist(
        ["I", "YOU", "HELP"],
        sentence_hit={"sentence": "can i help you",
                      "landmark_file": "h.json", "similarity": 1.0},
    ))
    assert out["retrieval_detail"]["level"] == "sentence"
    assert out["retrieval_detail"]["store"] == "stub"
    assert [e["gloss"] for e in out["clip_playlist"]] == ["I", "YOU", "HELP"]
    assert out["unsupported_tokens"] == []
    assert out["unresolved_tokens"] == []


def test_composed_marks_unsupported_never_substitutes(monkeypatch):
    import asyncio
    import retrieval

    entries = {"YOU": _entry("YOU"),
               "LET": _entry("LET", clip=False),  # corpus ref only: unplayable
               "ZZZ": None}  # unknown entirely
    import clip_store
    _stub_store(monkeypatch, entries)

    out = asyncio.run(retrieval.resolve_playlist(["YOU", "LET", "ZZZ"]))
    assert out["retrieval_detail"]["level"] == "composed"
    assert [e["gloss"] for e in out["clip_playlist"]] == ["YOU", "LET"]
    assert out["resolved_tokens"] == ["YOU", "LET"]
    assert out["unresolved_tokens"] == ["ZZZ"]
    # LET has no clip and no per-token landmark: reported, not replaced.
    assert out["unsupported_tokens"] == ["LET", "ZZZ"]
    # No-substitution invariant: input order preserved, nothing invented.
    accounted = out["resolved_tokens"] + out["unresolved_tokens"]
    assert accounted == ["YOU", "LET", "ZZZ"]


def test_phrase_matches_score_bigrams(monkeypatch):
    import asyncio
    import retrieval
    import clip_store

    async def fake_index():
        return [
            {"sentence": "help me", "canonical_tokens": ["ME", "HELP"],
             "landmark_file": "h.json"},
            {"sentence": "unrelated", "canonical_tokens": ["YOU", "GOOD"],
             "landmark_file": "g.json"},
        ]

    monkeypatch.setattr(clip_store, "corpus_canonical_index", fake_index)
    matches, coverage = asyncio.run(
        retrieval.phrase_matches(["PLEASE", "ME", "HELP"]))
    assert len(matches) == 1
    assert matches[0]["sentence"] == "help me"
    assert matches[0]["matched_tokens"] == ["HELP", "ME"]
    assert coverage == pytest.approx(2 / 3, abs=1e-3)


def test_clip_store_json_fallback_shapes():
    """Dead DB (conftest URL) -> JSON fallback with resolver-identical shapes."""
    import asyncio
    import animation_resolver
    import clip_store

    async def main():
        assert await clip_store.store_name() == "json_fallback"
        assert await clip_store.get_sign("HI") == animation_resolver.resolve_gloss_token("HI")
        assert await clip_store.get_sign("definitelynotagloss") is None
        hit = await clip_store.get_sentence("hello how are you")
        assert hit["landmark_file"] == "hello_landmarks.json"
        assert await clip_store.get_sentence("no such sentence") is None

    asyncio.run(main())


def test_translate_animation_carries_retrieval_detail(client):
    resp = client.post("/api/translate", json={"text": "hello how are you"})
    assert resp.status_code == 200
    anim = resp.json()["animation"]
    assert anim["retrieval_detail"]["level"] == "sentence"
    assert anim["retrieval_detail"]["store"] == "json_fallback"
    assert anim["unsupported_tokens"] == []
    assert anim["retrieval_detail"]["sentence_match"]["similarity"] == 1.0


def test_animate_reports_unsupported(client):
    resp = client.post("/api/animate", json={"gloss_sequence": ["HI", "ZZZNOTREAL"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval_detail"]["level"] == "composed"
    assert "ZZZNOTREAL" in body["unresolved_tokens"]
    assert "ZZZNOTREAL" in body["unsupported_tokens"]
    assert [e["gloss"] for e in body["clip_playlist"]] == ["HI"]
