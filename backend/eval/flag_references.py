"""Flag uncertain canonical refs for human verification (M4 Phase 2).

Reads backend/eval/references.csv + production vocab (backend/dataset.csv),
writes backend/eval/review_flags.csv with a `flags` column. Unflagged rows
are the high-confidence bulk; flagged rows need user verification before
the set becomes the BLEU reference set. Nothing here locks anything.

Flag meanings:
  MULTI_VERB      2+ verbs: order kept stable, may need manual ISL order
  NO_VERB_LONG    4+ tokens, no verb detected: possibly a missed verb
  STEMMED         suffix-stemming fired: verify the base form is intended
  UNKNOWN_TOKEN   content token outside the CISLR vocab (also an M5 signal)
  TYPO_CANDIDATE  token is 1-2 edits from a vocab word (corpus typos)
  OPAQUE_UNIT     underscore/multiword passthrough (ARE_YOU, THANK YOU)
  REPEATED_TOKEN  duplicated content token (faithful echo or error?)
  LONG            7+ tokens: complex clause, reorder confidence lower
  AUX_KEPT        lone DO/DOES/DID kept as pro-verb (do-support)
  PREP_STRANDED   preposition kept in place while the verb moved past it
  WH_NEG_COMBO    WH-word and negation co-occur: check final order
  REVIEW_HOLD     sentence listed in eval/review_hold.txt: deferred for
                  separate review, must not join the locked set
"""

import csv
import difflib
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

EVAL_DIR = BACKEND_DIR / "eval"

PREPOSITIONS = {
    "IN", "INTO", "ON", "OFF", "WITH", "FROM", "TO", "FOR", "BY", "ABOUT",
    "OF", "AT", "OVER", "UNDER",
}

LONG_WORDS = (
    "congratulations", "congratulatiions", "dilemma", "appreciate",
    "something", "anything",
)


def main() -> int:
    from isl_grammar import (
        AUX_DO, CORE_VERBS, NEG_WORDS, TIME_WORDS, WH_WORDS, _is_opaque,
        _stem_to_gloss,
    )

    vocab = {}
    with open(BACKEND_DIR / "dataset.csv", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            g = (row.get("gloss") or "").strip().lower()
            if g and g not in vocab:
                vocab[g] = True
    vocab_keys = set(vocab)

    with open(EVAL_DIR / "references.csv", encoding="utf-8") as f:
        refs = list(csv.DictReader(f))

    # Deferred items under separate review: always flagged REVIEW_HOLD so
    # they can never silently join the locked BLEU set. This is a review
    # workflow mechanism, not a grammar rule.
    hold = set()
    hold_file = EVAL_DIR / "review_hold.txt"
    if hold_file.exists():
        hold = {l.strip().lower() for l in
                hold_file.read_text(encoding="utf-8").splitlines() if l.strip()}

    out_rows = []
    for r in refs:
        # Keep THANK YOU as one unit so its parts are not misjudged.
        units = (r["corrected_gloss"] or "").replace("THANK YOU", "THANK_YOU").split()
        canon = (r["corrected_gloss"] or "").split()
        toks = [t for t in units if not _is_opaque(t)]
        flags: list[str] = []

        verbs = [t for t in toks if t in CORE_VERBS]
        if len(verbs) >= 2:
            flags.append("MULTI_VERB")
        if len(canon) >= 4 and not verbs:
            flags.append("NO_VERB_LONG")
        if len(canon) >= 7:
            flags.append("LONG")
        if any(_is_opaque(t) for t in canon):
            flags.append("OPAQUE_UNIT")
        if len(set(t for t in toks)) != len(toks):
            flags.append("REPEATED_TOKEN")
        if any(t in AUX_DO for t in canon):
            flags.append("AUX_KEPT")

        content = [t for t in toks if t not in WH_WORDS and t not in NEG_WORDS
                   and t not in TIME_WORDS]
        for t in content:
            low = t.lower()
            if low not in vocab_keys:
                flags.append(f"UNKNOWN_TOKEN:{t}")
                # Suggest only confident, long-word matches: short-word
                # difflib guesses (REALLY->RALLY) are noise.
                if len(t) >= 8:
                    close = difflib.get_close_matches(low, vocab_keys, n=1, cutoff=0.9)
                    if close and close[0] != low:
                        flags.append(f"TYPO_CANDIDATE:{t}->{close[0].upper()}")
        if _stem_to_gloss and any(
            t.lower() not in vocab_keys and _stem_to_gloss(t, vocab_keys)
            for t in content if len(t) > 3
        ):
            flags.append("STEMMED")
        if any(t in WH_WORDS for t in toks) and any(t in NEG_WORDS for t in toks):
            flags.append("WH_NEG_COMBO")
        if any(t in PREPOSITIONS for t in canon) and verbs and len(verbs) == 1:
            flags.append("PREP_STRANDED")
        if r["sentence"].strip().lower() in hold:
            flags.append("REVIEW_HOLD")

        out_rows.append({
            "sentence": r["sentence"],
            "corpus_gloss": r["corpus_gloss"],
            "canonical_gloss": r["corrected_gloss"],
            "dropped": r["dropped"],
            "flags": "|".join(dict.fromkeys(flags)),
        })

    with open(EVAL_DIR / "review_flags.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sentence", "corpus_gloss", "canonical_gloss", "dropped", "flags"])
        w.writeheader()
        w.writerows(out_rows)

    flagged = sum(1 for r in out_rows if r["flags"])
    print(f"reviewed {len(out_rows)} refs, flagged {flagged} -> review_flags.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
