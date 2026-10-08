"""
Canonical ISL gloss ordering (M4 Phase 1).

English tokens -> canonical ISL gloss sequence. This module owns LINGUISTIC
correctness only: it never consults clip availability. Retrieval (M5) adapts
to these tokens; these tokens are never adapted to retrieval.

Grammar v1 (hybrid, corpus-informed):
  - normalise: uppercase, strip punctuation, expand contractions
    (I'M -> I AM, DON'T/DONT/DONOT -> DO NOT, CANNOT/CANT -> CAN NOT, ...)
  - canonical signs: HELLO/HI/HEY -> NAMASTE (the ISL greeting sign),
    THANKS/THANK [+ YOU] -> THANK YOU (one sign), GOING/GONE/WENT -> GO,
    LIVING -> LIVE, plus safe suffix stemming only onto real CISLR glosses
  - drop function words: articles (A/AN/THE), be-verbs, modals
    (CAN/COULD/WOULD/SHOULD/MAY/MIGHT/MUST/SHALL/WILL), THAT, FOR, AND, OF,
    and DO/DOES/DID when they are auxiliaries (another verb exists;
    otherwise the last one is kept as the main verb). AND is dropped on
    corpus precedent ("go and sleep" -> GO SLEEP): ISL enumerates by
    juxtaposition. OR/BUT are kept: disjunctions change meaning.
    Reflexives map to their base pronoun (MYSELF -> ME) when absent,
    else dropped to avoid duplicates.
  - order (stable, applied in this sequence):
      1. time words fronted   (TODAY/YESTERDAY/TOMORROW/NOW/MORNING/NIGHT/EVENING)
      2. single main verb final (SOV); skipped when 0 or 2+ main verbs are
         present, so complex clauses keep a stable order instead of a wrong
         one. HAVE is auxiliary-like for the count but still goes final
         when it is the only verb ("YOUR FOOD HAVE").
      3. WH-words final       (WHAT/WHEN/WHERE/WHO/WHICH/WHY/HOW...)
      4. negation final       (NOT/NO/NEVER)
  - multi-word units (THANK YOU) and underscore units (ARE_YOU) are opaque:
    never selected as verb/WH/NEG/TIME, never split.

Verb detection is deliberately precision-biased (curated CORE_VERBS): a
missed verb keeps English order (status quo), while a false verb would
actively corrupt order. New verbs are added with a unit test.

Returns (canonical_tokens, dropped_tokens).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

_TOKEN_PUNCT = ".,!?;:'\""

ARTICLES = {"A", "AN", "THE"}
BE_VERBS = {"AM", "ARE", "IS", "WAS", "WERE", "BE", "BEING", "BEEN"}
MODALS = {"CAN", "COULD", "WOULD", "SHOULD", "MAY", "MIGHT", "MUST", "SHALL", "WILL"}
AUX_DO = {"DO", "DOES", "DID"}
DROPPED_OTHER = {"THAT", "FOR", "AND", "OF"}

WH_WORDS = {"WHAT", "WHEN", "WHERE", "WHO", "WHOM", "WHOSE", "WHICH", "WHY", "HOW"}
NEG_WORDS = {"NOT", "NO", "NEVER"}
TIME_WORDS = {"TODAY", "TOMORROW", "YESTERDAY", "NOW", "MORNING", "NIGHT", "EVENING"}

# Fused forms expanded before anything else (keys are post-uppercase tokens).
CONTRACTION_MAP = {
    "I'M": ("I", "AM"), "YOU'RE": ("YOU", "ARE"), "HE'S": ("HE", "IS"),
    "SHE'S": ("SHE", "IS"), "IT'S": ("IT", "IS"), "WE'RE": ("WE", "ARE"),
    "THEY'RE": ("THEY", "ARE"), "I'VE": ("I", "HAVE"), "YOU'VE": ("YOU", "HAVE"),
    "WE'VE": ("WE", "HAVE"), "THEY'VE": ("THEY", "HAVE"),
    "DON'T": ("DO", "NOT"), "DOESN'T": ("DOES", "NOT"), "DIDN'T": ("DID", "NOT"),
    "CAN'T": ("CAN", "NOT"), "WON'T": ("WILL", "NOT"), "ISN'T": ("IS", "NOT"),
    "AREN'T": ("ARE", "NOT"), "WASN'T": ("WAS", "NOT"), "WEREN'T": ("WERE", "NOT"),
    "HASN'T": ("HAS", "NOT"), "HAVEN'T": ("HAVE", "NOT"), "HADN'T": ("HAD", "NOT"),
    "COULDN'T": ("COULD", "NOT"), "WOULDN'T": ("WOULD", "NOT"),
    "SHOULDN'T": ("SHOULD", "NOT"),
    "I'LL": ("I", "WILL"), "YOU'LL": ("YOU", "WILL"),
    "DONT": ("DO", "NOT"), "DOESNT": ("DOES", "NOT"), "DIDNT": ("DID", "NOT"),
    "CANT": ("CAN", "NOT"), "WONT": ("WILL", "NOT"), "ISNT": ("IS", "NOT"),
    "ARENT": ("ARE", "NOT"), "DONOT": ("DO", "NOT"), "CANNOT": ("CAN", "NOT"),
}

# Canonical sign mappings (linguistic, not clip-driven).
GREETING_MAP = {"HELLO": "NAMASTE", "HI": "NAMASTE", "HEY": "NAMASTE"}
GO_MAP = {"GOING": "GO", "GONE": "GO", "WENT": "GO"}
LIVE_MAP = {"LIVING": "LIVE"}

# Reflexives resolve to their base pronoun (object form preserves argument
# role: "plants and myself" -> PLANT ME). Dropped when the pronoun already
# occurs, so self-directed actions never duplicate ("take care of yourself"
# with an overt YOU subject keeps one YOU).
REFLEXIVE_MAP = {
    "MYSELF": "ME", "YOURSELF": "YOU", "HIMSELF": "HIM", "HERSELF": "HER",
    "ITSELF": "IT", "OURSELVES": "US", "YOURSELVES": "YOU",
    "THEMSELVES": "THEM", "ONESELF": "ONE",
}

# Possessive/existential HAVE family: normalised to HAVE and kept as a real
# verb (ISL "I CAR HAVE"), never dropped. GOT is included: "got hurt" and
# "got to know" use it auxiliary-style, and HAVE carries the meaning.
HAVE_MAP = {"HAVE": "HAVE", "HAS": "HAVE", "HAD": "HAVE", "GOT": "HAVE"}

# Curated main verbs (precision-biased; see module docstring). Covers every
# verb in the 101-sentence corpus plus common ISL-demo verbs.
CORE_VERBS = {
    "HELP", "GO", "COME", "GIVE", "TAKE", "MAKE", "READ", "WRITE", "PLAY",
    "SPEAK", "TALK", "TELL", "ASK", "ANSWER", "UNDERSTAND", "KNOW", "THINK",
    "WANT", "NEED", "LIKE", "LOVE", "LIVE", "WORK", "STUDY", "LEARN", "TEACH",
    "BUY", "OPEN", "CLOSE", "TURN", "REPEAT", "TRY", "CALL", "SEND", "BRING",
    "CARRY", "WASH", "COOK", "CLEAN", "WEAR", "PUT", "KEEP", "LEAVE", "STAY",
    "WAIT", "MEET", "SEE", "EAT", "DRINK", "SLEEP", "SIT", "STAND", "WALK",
    "RUN", "CRY", "LAUGH", "SMILE", "HURT", "FEEL", "BECOME", "SAY", "LOOK",
    "WATCH", "LISTEN", "HEAR", "SMELL", "TOUCH", "HOLD", "CATCH", "THROW",
    "PULL", "PUSH", "LIFT", "DROP", "BREAK", "FIX", "BUILD", "DRIVE", "RIDE",
    "FLY", "SWIM", "DANCE", "SING", "SERVE", "POUR", "COMB", "CHAT", "TRUST",
    "DARE", "AGREE", "MEAN", "ENJOY", "APPRECIATE", "PLAN", "HAPPEN", "STOP",
    "CARE", "PREPARE", "ABUSE", "HIDE", "SHOW", "DRINK", "PROMISE", "SERVE",
    "SIT", "DO", "WORRY", "HAVE",
}


def _clean_token(token: str) -> str:
    return token.strip().strip(_TOKEN_PUNCT).upper()


def _is_opaque(token: str) -> bool:
    """Multi-word/underscore units are never reordered by role."""
    return " " in token or "_" in token


def _stem_to_gloss(key: str, vocab_keys: set) -> str | None:
    """Map an inflected word onto a base-form gloss in the vocabulary.

    Guarded by length: 2-3 letter words must never be stemmed ("IS" -> "I",
    "HIS" -> "HI" -> NAMASTE). Short function words are handled by the
    drop lists instead. Handles -ing, -s/-es and -ed/-d (STOPPED -> STOP,
    ENJOYED -> ENJOY, PLANNED -> PLAN), consonant doubling included.
    """
    if not key or len(key) <= 3:
        return None
    candidates: List[str] = []
    low = key.lower()
    if low.endswith("ing"):
        base = low[:-3]
        candidates += [base, base + "e"]
        if len(base) > 1 and base[-1] == base[-2]:
            candidates.append(base[:-1])
    if low.endswith("ed"):
        base = low[:-2]
        candidates.append(base)            # STOPPED -> stopp
        candidates.append(base + "e")      # LIVED -> live, AGREED -> agree
        if len(base) > 1 and base[-1] == base[-2]:
            candidates.append(base[:-1])   # STOPPED -> stop, PLANNED -> plan
    if low.endswith("s") and not low.endswith("ss"):
        candidates.append(low[:-1])
        if low.endswith("es"):
            candidates.append(low[:-2])
    for candidate in candidates:
        if candidate and candidate in vocab_keys:
            return candidate.upper()
    return None


def _load_vocab_keys() -> set:
    from gloss_lookup import load_vocabulary

    return set(load_vocabulary().keys())


def canonicalize(gloss_sequence: Iterable) -> Tuple[List[str], List[str]]:
    """Map tokens to canonical ISL gloss order. See module docstring."""
    vocab_keys = _load_vocab_keys()

    # 1. clean + expand contractions + drop closed-class function words.
    #    Unconditional drops happen BEFORE vocab/stem mapping so a short
    #    word can never be misread as a sign ("IS" stemmed to "I").
    #    DO/DOES/DID need verb context and are handled in step 4.
    expanded: List[str] = []
    dropped: List[str] = []
    for token in gloss_sequence:
        if not isinstance(token, str):
            continue
        clean = _clean_token(token)
        if not clean:
            continue
        if clean in CONTRACTION_MAP:
            expanded.extend(CONTRACTION_MAP[clean])
        else:
            expanded.append(clean)
    filtered: List[str] = []
    for token in expanded:
        if (
            token in ARTICLES
            or token in BE_VERBS
            or token in MODALS
            or token in DROPPED_OTHER
        ):
            dropped.append(token)
        else:
            filtered.append(token)
    expanded = filtered

    # 2. merge THANK YOU into one sign (before any role assignment)
    merged: List[str] = []
    i = 0
    while i < len(expanded):
        if (
            expanded[i] in ("THANK", "THANKS")
            and i + 1 < len(expanded)
            and expanded[i + 1] == "YOU"
        ):
            merged.append("THANK YOU")
            i += 2
        elif expanded[i] in ("THANK", "THANKS"):
            merged.append("THANK YOU")
            i += 1
        else:
            merged.append(expanded[i])
            i += 1

    # 3. canonical sign mapping + safe stemming
    mapped: List[str] = []
    for token in merged:
        if _is_opaque(token):
            mapped.append(token)
        elif token in GREETING_MAP:
            mapped.append(GREETING_MAP[token])
        elif token in GO_MAP:
            mapped.append(GO_MAP[token])
        elif token in LIVE_MAP:
            mapped.append(LIVE_MAP[token])
        elif token in HAVE_MAP:
            mapped.append(HAVE_MAP[token])
        elif token.lower() in vocab_keys:
            mapped.append(token)
        else:
            stemmed = _stem_to_gloss(token, vocab_keys)
            mapped.append(stemmed if stemmed else token)

    # 3b. reflexives -> base pronoun when absent, else dropped. Runs after
    # mapping so e.g. LIVING-derived tokens are already settled, and before
    # verb detection so ME/YOU count as arguments, never verbs.
    seen_args = {t for t in mapped if t not in REFLEXIVE_MAP}
    resolved: List[str] = []
    for token in mapped:
        if token in REFLEXIVE_MAP:
            base = REFLEXIVE_MAP[token]
            if base in seen_args:
                dropped.append(token)
            else:
                seen_args.add(base)
                resolved.append(base)
        else:
            resolved.append(token)
    mapped = resolved

    # 4. drops (aux-DO rule needs verb knowledge: DO/DOES/DID are dropped
    #    only when another verb exists, else the last one is the main verb).
    #    Unconditional drops already happened in step 1 and are in `dropped`.
    verbs_present = [
        t for t in mapped
        if not _is_opaque(t) and (t in CORE_VERBS)
    ]
    other_verbs = [t for t in verbs_present if t not in AUX_DO]
    kept: List[str] = []
    if other_verbs:
        for token in mapped:
            if token in AUX_DO:
                dropped.append(token)
            else:
                kept.append(token)
    else:
        do_seen = [t for t in mapped if t in AUX_DO]
        keep_do = do_seen[-1] if do_seen else None
        kept_do = False
        for token in mapped:
            if token in AUX_DO:
                if token == keep_do and not kept_do:
                    kept_do = True
                    kept.append(token)
                else:
                    dropped.append(token)
            else:
                kept.append(token)

    # 5. reorder: time-front, single-verb-final, WH-final, NEG-final
    times = [t for t in kept if not _is_opaque(t) and t in TIME_WORDS]
    rest = [t for t in kept if _is_opaque(t) or t not in TIME_WORDS]
    whs = [t for t in rest if not _is_opaque(t) and t in WH_WORDS]
    rest = [t for t in rest if _is_opaque(t) or t not in WH_WORDS]
    negs = [t for t in rest if not _is_opaque(t) and t in NEG_WORDS]
    rest = [t for t in rest if _is_opaque(t) or t not in NEG_WORDS]
    verbs = [t for t in rest if not _is_opaque(t) and t in CORE_VERBS]
    # HAVE is auxiliary-like for counting: it never blocks another verb's
    # final move ("I SOMEHOW HAVE ABOUT IT KNOW"), but as the only verb it
    # still goes final ("YOUR FOOD HAVE").
    main_verbs = [t for t in verbs if t != "HAVE"]
    if len(main_verbs) == 1:
        rest = [t for t in rest if t != main_verbs[0]] + main_verbs
    elif not main_verbs and "HAVE" in verbs:
        rest = [t for t in rest if t != "HAVE"] + ["HAVE"]

    return times + rest + whs + negs, dropped
