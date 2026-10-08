"""
Hierarchical clip retrieval (M5/Phase 3).

Translation (isl_grammar) is authoritative on WORDS; this module is
authoritative on ASSETS. It never edits, substitutes, or reorders gloss
tokens: every input token lands in exactly one of
  - clip_playlist  (a playable record exists), or
  - unsupported_tokens (no clip and no per-token landmark), or
  - unresolved_tokens  (unknown to the inventory entirely; a subset of
    unsupported in practice, kept as its own list for compatibility).

Levels:
  1. sentence  - direct_lookup/rag hit with a landmark file: the sentence
     clip covers the whole sequence (per-token sign records still attach
     clip metadata for reference).
  2. composed  - per-token sign records from the authoritative store;
     phrase_matches score canonical n-gram overlap against corpus glosses
     as retrieval evidence (playback still composes sign clips; no phrase
     assets exist until M6).
"""

from __future__ import annotations

import clip_store


def _bigrams(tokens: list) -> set:
    return set(zip(tokens, tokens[1:])) if len(tokens) >= 2 else set()


async def phrase_matches(canonical_tokens: list, top_n: int = 3) -> tuple[list, float]:
    """Score input against corpus canonical glosses (evidence, not playback).

    Returns (matches, coverage): matches are up to top_n
    {sentence, matched_tokens, bigram_hits} ordered by bigram hits then
    token coverage; coverage is the fraction of input tokens appearing in
    any matched corpus token set.
    """
    index = await clip_store.corpus_canonical_index()
    toks = [t for t in canonical_tokens if isinstance(t, str) and t.strip()]
    if not toks:
        return [], 0.0
    in_bigrams = _bigrams(toks)
    in_set = set(toks)
    scored = []
    for entry in index:
        entry_toks = entry.get("canonical_tokens") or []
        shared_bigrams = sorted(
            [" ".join(b) for b in (in_bigrams & _bigrams(entry_toks))]
        )
        if not shared_bigrams:
            continue
        overlap = sorted(in_set & set(entry_toks))
        scored.append({
            "sentence": entry.get("sentence", ""),
            "matched_tokens": overlap,
            "bigram_hits": shared_bigrams,
            "landmark_file": entry.get("landmark_file", ""),
        })
    scored.sort(key=lambda m: (len(m["bigram_hits"]), len(m["matched_tokens"])),
                reverse=True)
    matches = scored[:top_n]
    covered: set = set()
    for m in matches:
        covered.update(m["matched_tokens"])
    coverage = round(len(covered) / len(in_set), 3) if in_set else 0.0
    return matches, coverage


def _playable(entry: dict | None) -> bool:
    """A token is playable when a clip or per-token landmark exists.

    A sentence-level landmark_file reference alone does NOT count: it
    points at a whole other sentence's motion, not this sign.
    """
    if not entry:
        return False
    return bool(entry.get("clip_url") or entry.get("landmark_clip_url"))


def suggest_glosses(token: str, top_n: int = 3) -> list:
    """Nearest PLAYABLE vocab glosses for an unknown token (display only).

    Suggestions are information only: the pipeline never substitutes them
    (no-substitution invariant). Filters to signs the avatar can actually
    play so we never suggest another dead end.
    """
    from difflib import get_close_matches

    key = (token or "").strip().lower()
    if len(key) < 3:
        return []
    # Function words get no suggestions: guessing SAND for AND is noise,
    # and grammar drops these before retrieval ever sees them.
    from isl_grammar import (ARTICLES, AUX_DO, BE_VERBS, CONTRACTION_MAP,
                             DROPPED_OTHER, MODALS, REFLEXIVE_MAP)

    if (key.upper() in ARTICLES or key.upper() in BE_VERBS
            or key.upper() in MODALS or key.upper() in AUX_DO
            or key.upper() in DROPPED_OTHER or key.upper() in REFLEXIVE_MAP
            or key.upper() in CONTRACTION_MAP):
        return []
    from animation_resolver import resolve_gloss_token
    from gloss_lookup import load_vocabulary

    vocab_keys = set(load_vocabulary().keys())
    out = []
    for candidate in get_close_matches(key, vocab_keys, n=10, cutoff=0.6):
        if candidate == key:
            continue
        entry = resolve_gloss_token(candidate)
        if _playable(entry):
            out.append(candidate.upper())
        if len(out) >= top_n:
            break
    return out


async def resolve_playlist(
    canonical_tokens: list,
    sentence_hit: dict | None = None,
) -> dict:
    """Resolve canonical tokens to a clip playlist (hierarchical).

    sentence_hit: {sentence, landmark_file, similarity} from the
    direct_lookup/rag path, or None. Returns clip_playlist (input order),
    resolved/unresolved/unsupported tokens, and retrieval_detail
    {level, store, sentence_match, phrase_matches, coverage}.
    """
    store = await clip_store.store_name()
    tokens = [t for t in (canonical_tokens or []) if isinstance(t, str) and t.strip()]

    if sentence_hit and sentence_hit.get("landmark_file"):
        level = "sentence"
        phrase_list: list = []
        coverage = 1.0
    else:
        level = "composed"
        phrase_list, coverage = await phrase_matches(tokens)

    clip_playlist: list = []
    resolved_tokens: list = []
    unresolved_tokens: list = []
    unsupported_tokens: list = []

    for token in tokens:
        entry = await clip_store.get_sign(token)
        if entry is None:
            unresolved_tokens.append(token)
            # At sentence level the sentence clip covers every token; only
            # the composed fallback reports per-token asset gaps.
            if level == "composed":
                unsupported_tokens.append(token)
            continue
        resolved_tokens.append(token)
        clip_playlist.append(entry)
        if level == "composed" and not _playable(entry):
            unsupported_tokens.append(token)

    suggestions = {}
    for token in unresolved_tokens:
        options = suggest_glosses(token)
        if options:
            suggestions[token] = options

    return {
        "clip_playlist": clip_playlist,
        "resolved_tokens": resolved_tokens,
        "unresolved_tokens": unresolved_tokens,
        "unsupported_tokens": unsupported_tokens,
        "suggestions": suggestions,
        "retrieval_detail": {
            "level": level,
            "store": store,
            "sentence_match": sentence_hit,
            "phrase_matches": phrase_list,
            "coverage": coverage,
        },
    }
