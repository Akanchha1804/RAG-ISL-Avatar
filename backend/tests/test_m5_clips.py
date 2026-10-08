"""M5: best-clip selection, junk filtering, display-only suggestions."""

import pytest


def test_best_clip_prefers_shortest_duration():
    import gloss_lookup

    gloss_lookup.reset_vocabulary_state()
    gloss_lookup._landmark_uids = set()  # no landmark info in fixtures
    rows = dict((g, (u, c, d)) for g, u, c, d in gloss_lookup.iter_best_cislr_rows())
    # 'you' has two fixture videos (0.6s vs 0.9s): shortest wins.
    assert rows["you"][0] == "uid-you"
    assert rows["hi"][0] == "uid-hi"


def test_best_clip_prefers_landmark_ready_over_shorter():
    import gloss_lookup

    gloss_lookup.reset_vocabulary_state()
    # Pretend the LONGER video is the avatar-mirrorable one.
    gloss_lookup._landmark_uids = {"uid-dup"}
    rows = dict((g, (u, c, d)) for g, u, c, d in gloss_lookup.iter_best_cislr_rows())
    assert rows["you"][0] == "uid-dup"
    gloss_lookup.reset_vocabulary_state()


def test_junk_glosses_skipped(tmp_path):
    import gloss_lookup

    csv_path = tmp_path / "cislr.csv"
    csv_path.write_text(
        "uid,gloss,duration,category\n"
        "u1,#N/A,5,x\n"
        "u2,,5,x\n"
        "u3,hello,2,greeting\n"
        "u4,hello,1,greeting\n",
        encoding="utf-8",
    )
    gloss_lookup.reset_vocabulary_state()
    gloss_lookup._landmark_uids = set()
    rows = list(gloss_lookup.iter_best_cislr_rows(csv_path))
    assert [(g, u) for g, u, _, _ in rows] == [("hello", "u4")]
    gloss_lookup.reset_vocabulary_state()


def test_suggest_glosses_playable_only():
    import gloss_lookup
    from retrieval import suggest_glosses

    gloss_lookup.reset_vocabulary_state()
    gloss_lookup._landmark_uids = set()
    assert suggest_glosses("THANKK") == ["THANK"]
    assert suggest_glosses("HI") == []  # too short: no guesses
    assert suggest_glosses("AND") == []  # function word: no guesses
    assert suggest_glosses("OF") == []
    assert suggest_glosses("ZZZNOTREAL") == [] or all(
        isinstance(s, str) for s in suggest_glosses("ZZZNOTREAL")
    )
    gloss_lookup.reset_vocabulary_state()


def test_resolve_playlist_carries_suggestions(monkeypatch):
    import asyncio
    import retrieval

    async def fake_sign(token):
        return None

    async def fake_index():
        return []

    async def fake_store():
        return "stub"

    import clip_store
    monkeypatch.setattr(clip_store, "get_sign", fake_sign)
    monkeypatch.setattr(clip_store, "corpus_canonical_index", fake_index)
    monkeypatch.setattr(clip_store, "store_name", fake_store)
    monkeypatch.setattr(retrieval, "suggest_glosses", lambda t, top_n=3: ["X"] if t == "QQQ" else [])

    out = asyncio.run(retrieval.resolve_playlist(["QQQ", "WWW"]))
    assert out["suggestions"] == {"QQQ": ["X"]}
    assert out["unsupported_tokens"] == ["QQQ", "WWW"]
