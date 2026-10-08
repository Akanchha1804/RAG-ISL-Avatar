"""M4 Phase 1: canonical ISL grammar rules. No models (vocab falls back fine)."""

from isl_grammar import canonicalize


def test_drops_articles_beverbs_modals():
    canon, dropped = canonicalize(["CAN", "I", "HELP", "YOU"])
    assert canon == ["I", "YOU", "HELP"]
    assert dropped == ["CAN"]


def test_sov_single_verb_final():
    assert canonicalize(["I", "HELP", "YOU"])[0] == ["I", "YOU", "HELP"]
    assert canonicalize(["YOU", "REPEAT", "PLEASE"])[0] == ["YOU", "PLEASE", "REPEAT"]
    assert canonicalize(["WE", "GO", "OUTSIDE"])[0] == ["WE", "OUTSIDE", "GO"]


def test_multi_verb_keeps_stable_order():
    seq = ["TURN", "ON", "LIGHT", "TURN", "OFF", "LIGHT"]
    assert canonicalize(seq)[0] == seq


def test_wh_final():
    assert canonicalize(["WHAT", "YOUR", "NAME"])[0] == ["YOUR", "NAME", "WHAT"]
    assert canonicalize(["WHEN", "TRAIN", "LEAVE"])[0] == ["TRAIN", "LEAVE", "WHEN"]


def test_neg_final_with_verb():
    assert canonicalize(["I", "DO", "NOT", "LIKE", "IT"])[0] == ["I", "IT", "LIKE", "NOT"]


def test_aux_do_dropped_when_verb_exists():
    canon, dropped = canonicalize(["WHAT", "DO", "YOU", "THINK"])
    assert "DO" in dropped
    assert canon == ["YOU", "THINK", "WHAT"]


def test_lone_do_kept_as_main_verb():
    canon, _ = canonicalize(["WHAT", "DO", "YOU", "DO"])
    assert canon == ["YOU", "DO", "WHAT"]


def test_time_fronted():
    assert canonicalize(["YOU", "FREE", "TODAY"])[0] == ["TODAY", "YOU", "FREE"]
    # GO_MAP needs no vocabulary (fixture vocab has 6 glosses on purpose).
    assert canonicalize(["HE", "GOING", "TODAY"])[0] == ["TODAY", "HE", "GO"]


def test_short_words_never_stemmed(monkeypatch):
    import gloss_lookup

    monkeypatch.setattr(
        gloss_lookup, "load_vocabulary",
        lambda: {"come": {"uid": "u", "category": "process", "duration": 1.0}},
    )
    assert canonicalize(["HE", "COMING", "TODAY"])[0] == ["TODAY", "HE", "COME"]


def test_contractions_expanded():
    assert canonicalize(["I'M", "HAPPY"])[0] == ["I", "HAPPY"]
    assert canonicalize(["DONOT", "WORRY"])[0] == ["WORRY", "NOT"]


def test_greeting_canonical_sign():
    assert canonicalize(["HELLO"])[0] == ["NAMASTE"]
    assert canonicalize(["HI", "HOW", "YOU"])[0] == ["NAMASTE", "YOU", "HOW"]


def test_thank_you_merges_and_is_opaque():
    canon, _ = canonicalize(["THANK", "YOU", "SO", "MUCH"])
    assert canon == ["THANK YOU", "SO", "MUCH"]


def test_underscore_units_opaque():
    assert canonicalize(["HI", "HOW", "ARE_YOU"])[0] == ["NAMASTE", "ARE_YOU", "HOW"]


def test_empty_after_drops():
    assert canonicalize(["IS"])[0] == []
    assert canonicalize([]) == ([], [])


def test_short_words_never_stemmed(monkeypatch):
    """Regression: stemmer once read IS -> I and HIS -> HI -> NAMASTE."""
    import gloss_lookup

    monkeypatch.setattr(
        gloss_lookup, "load_vocabulary",
        lambda: {"i": {}, "hi": {}, "come": {}, "book": {}},
    )
    canon, dropped = canonicalize(["WHAT", "IS", "YOUR", "NAME"])
    assert canon == ["YOUR", "NAME", "WHAT"]
    assert "IS" in dropped
    canon, _ = canonicalize(["HIS", "BOOK"])
    assert "NAMASTE" not in canon


def test_non_string_tokens_ignored():
    assert canonicalize(["YOU", None, "GO"])[0] == ["YOU", "GO"]


def test_ed_stemming(monkeypatch):
    import gloss_lookup

    monkeypatch.setattr(
        gloss_lookup, "load_vocabulary",
        lambda: {"stop": {}, "enjoy": {}, "plan": {}, "happen": {}},
    )
    assert canonicalize(["I", "STOPPED"])[0] == ["I", "STOP"]
    assert canonicalize(["I", "ENJOYED"])[0] == ["I", "ENJOY"]
    assert canonicalize(["WE", "PLANNED"])[0] == ["WE", "PLAN"]
    assert canonicalize(["WHAT", "HAPPENED"])[0] == ["HAPPEN", "WHAT"]


def test_have_family_kept_as_verb():
    assert canonicalize(["HAD", "YOUR", "FOOD"])[0] == ["YOUR", "FOOD", "HAVE"]
    assert canonicalize(["I", "HAVE", "CAR"])[0] == ["I", "CAR", "HAVE"]


def test_and_of_dropped_enumeration_ellipsis():
    # Corpus precedent: "go and sleep" -> GO SLEEP.
    assert canonicalize(["WATER", "AND", "PLANT"])[0] == ["WATER", "PLANT"]
    assert canonicalize(["TAKE", "CARE", "OF", "YOU"])[0] == ["TAKE", "CARE", "YOU"]


def test_disjunctions_kept():
    # OR/BUT change meaning: never dropped.
    assert canonicalize(["TEA", "OR", "COFFEE"])[0] == ["TEA", "OR", "COFFEE"]


def test_reflexive_maps_when_absent_drops_when_present():
    assert canonicalize(["PLANT", "AND", "MYSELF"])[0] == ["PLANT", "ME"]
    assert canonicalize(["TAKE", "CARE", "OF", "YOURSELF"])[0] == ["TAKE", "CARE", "YOU"]
    out, dropped = canonicalize(["YOU", "HURT", "YOURSELF"])
    assert out == ["YOU", "HURT"]
    assert "YOURSELF" in dropped


def test_corpus_snapshot_lock(monkeypatch):
    """Regression lock: all 101 corpus glosses canonicalize deterministically.

    Divergences from the English-ordered corpus glosses are INTENDED (M4);
    this test pins behavior so grammar edits show up as explicit diffs.
    Regenerate via C:/Users/SHYAMK~1/AppData/Local/Temp/opencode/m4_snapshot.py
    only when a rule change is deliberate, then review the diff.
    """
    import csv
    import json
    from pathlib import Path

    import gloss_lookup

    backend_dir = Path(__file__).resolve().parent.parent
    vocab = {}
    with open(backend_dir / "dataset.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            g = (row.get("gloss") or "").strip().lower()
            if g and g not in vocab:
                vocab[g] = {"uid": row.get("uid"), "category": row.get("category"),
                            "duration": row.get("duration")}
    monkeypatch.setattr(gloss_lookup, "_gloss_vocab", vocab)

    mapping = json.loads((backend_dir / "sentence_mapping.json").read_text(encoding="utf-8"))
    expected = json.loads(
        (backend_dir / "tests" / "fixtures" / "canonical_snapshot.json").read_text(encoding="utf-8")
    )
    assert set(expected) == set(mapping), "snapshot stale: mapping changed, regenerate"
    for sentence, entry in mapping.items():
        tokens = [t for t in entry.get("glosses", "").split(" ") if t]
        canon, dropped = canonicalize(tokens)
        assert expected[sentence] == {"canonical": canon, "dropped": dropped}, sentence
