import json
from pathlib import Path

from animation_resolver import (
    SOURCE_CISLR,
    SOURCE_CORPUS,
    normalize_gloss_sequence,
    reset_registry,
    resolve_gloss_sequence,
    resolve_gloss_token,
)

MAPPING = Path(__file__).resolve().parent / "fixtures" / "data" / "ISL_MediaPipe" / "sentence_mapping.json"


def setup_function(_):
    reset_registry()


def test_cislr_token_resolves_with_uid():
    entry = resolve_gloss_token("hi")
    assert entry is not None
    assert entry["source"] == SOURCE_CISLR
    assert entry["sign_id"] == "uid-hi"
    assert entry["clip_id"] == "uid-hi"
    assert entry["clip_url"] == "/clips/uid-hi.mp4"
    # no per-gloss landmark clips exist under the fixture DATA_DIR
    assert entry["landmark_clip_url"] is None
    assert entry["duration_ms"] == 800
    assert entry["category"] == "greeting"


def test_cislr_token_reports_landmark_clip_when_mapped(monkeypatch):
    import animation_resolver

    monkeypatch.setattr(
        animation_resolver,
        "load_gloss_landmarks",
        lambda: {"hi": {"status": "ready", "file": "gloss_hi.json"}},
    )
    entry = resolve_gloss_token("hi")
    assert entry["landmark_clip_url"] == "/landmarks/gloss_hi.json"


def test_cislr_token_ignores_unready_gloss_clip(monkeypatch):
    import animation_resolver

    monkeypatch.setattr(
        animation_resolver,
        "load_gloss_landmarks",
        lambda: {"hi": {"status": "no_landmarks", "file": ""}},
    )
    assert resolve_gloss_token("hi")["landmark_clip_url"] is None


def test_corpus_only_token_uses_landmark_reference():
    # "ARE_YOU" is in sentence_mapping glosses but not in cislr vocabulary
    entry = resolve_gloss_token("ARE_YOU")
    assert entry is not None
    assert entry["source"] == SOURCE_CORPUS
    assert entry["clip_id"] == "corpus::are_you"
    assert entry["landmark_file"] == "hello_landmarks.json"


def test_unknown_token_returns_none():
    assert resolve_gloss_token("definitelynotagloss") is None


def test_case_and_punctuation_insensitive():
    entry = resolve_gloss_token("  Hi! ")
    assert entry is not None
    assert entry["source"] == SOURCE_CISLR


def test_resolve_sequence_reports_unresolved():
    result = resolve_gloss_sequence(["HI", "NOPEGLOSS", "ARE_YOU"])
    assert result["resolved_tokens"] == ["HI", "ARE_YOU"]
    assert result["unresolved_tokens"] == ["NOPEGLOSS"]
    assert len(result["clip_playlist"]) == 2
    assert result["clip_playlist"][0]["gloss"] == "HI"
    assert result["clip_playlist"][0]["clip_url"] == "/clips/uid-hi.mp4"
    assert result["clip_playlist"][1]["landmark_file"] == "hello_landmarks.json"
    assert result["clip_playlist"][1].get("clip_url") is None


def test_resolve_sequence_never_raises_on_empty():
    assert resolve_gloss_sequence([]) == {
        "clip_playlist": [],
        "resolved_tokens": [],
        "unresolved_tokens": [],
    }


def test_resolve_sequence_skips_blank_tokens():
    result = resolve_gloss_sequence(["", "  ", "HI"])
    assert result["resolved_tokens"] == ["HI"]
    assert result["unresolved_tokens"] == []


def test_registry_built_from_fixture_mapping():
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    assert "hello how are you" in mapping


def test_normalize_maps_english_to_canonical_gloss():
    # "going" is not a gloss; it maps onto the "go" sign.
    normalized, dropped = normalize_gloss_sequence(["GOING"])
    assert normalized == ["GO"]
    assert dropped == []


def test_normalize_drops_unsigned_be_verbs():
    # ISL does not sign copular be-verbs ("am" is absent from the
    # fixture vocabulary, like the real one).
    normalized, dropped = normalize_gloss_sequence(["I", "AM", "HAPPY"])
    assert normalized == ["I", "HAPPY"]
    assert dropped == ["AM"]


def test_normalize_combines_thank_you_phrase():
    normalized, dropped = normalize_gloss_sequence(["THANK", "YOU"])
    assert normalized == ["THANK YOU"]
    assert dropped == []


def test_normalize_keeps_known_glosses_untouched():
    normalized, dropped = normalize_gloss_sequence(["HOW", "ARE", "YOU"])
    assert normalized == ["HOW", "ARE", "YOU"]
    assert dropped == []


def test_normalize_keeps_unresolvable_tokens_for_reporting():
    normalized, dropped = normalize_gloss_sequence(["ZZZ"])
    assert normalized == ["ZZZ"]
    assert dropped == []
