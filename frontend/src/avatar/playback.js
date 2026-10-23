// Pure playback logic, kept out of the WebGL component so it can be tested
// without a renderer. The numeric constants that are shared with Unity are
// documented in Docs/Description/10_Avatar_Retargeting_Spec.md.

// ISL non-manual markers driven from the gloss sequence: faces carry
// question/negation grammar that hands alone cannot express. WH-words ->
// brow raise (VRM 'surprised'); negation -> brow furrow (VRM 'angry' at a
// low, non-emotional weight). Display-only grammar channel; never affects
// glosses.
export const WH_GLOSSES = new Set([
  'WHAT', 'WHEN', 'WHERE', 'WHO', 'WHOM', 'WHOSE', 'WHICH', 'WHY', 'HOW',
]);
export const NEG_GLOSSES = new Set([
  'NOT', 'NO', 'NEVER', 'DONOT', 'DONT', 'CANNOT',
]);

export function expressionTargetsFor(glossSequence) {
  const toks = (Array.isArray(glossSequence)
    ? glossSequence
    : typeof glossSequence === 'string'
      ? glossSequence.split(/\s+/)
      : []
  ).map((g) => String(g).toUpperCase());
  return {
    surprised: toks.some((t) => WH_GLOSSES.has(t)) ? 0.55 : 0,
    angry: toks.some((t) => NEG_GLOSSES.has(t)) ? 0.35 : 0,
  };
}

// How far the hands travel from the current clip's last applied frame to the
// next clip's first frame, in normalised image space. Worst side wins. Used
// to decide whether a gloss switch can crossfade directly or must pass
// through the arms-down idle.
export function clipSwitchJump(lastWrist, nextFrame) {
  if (!lastWrist || !nextFrame) return Infinity;
  const next = {};
  for (const hand of nextFrame.hands || []) {
    if (!hand.landmarks || hand.landmarks.length !== 21) continue;
    const w = hand.landmarks[0];
    const side = w.x > 0.5 ? 'left' : 'right';
    if (!next[side]) next[side] = w;
  }
  let jump = 0;
  for (const side of ['left', 'right']) {
    const a = lastWrist[side];
    const b = next[side];
    if (a && b) jump = Math.max(jump, Math.hypot(a.x - b.x, a.y - b.y));
  }
  return jump;
}

// Clips the avatar can actually play: a resolved gloss only reaches the
// playlist with a per-gloss landmark clip. A resolved gloss without one is
// not signable, so callers must not treat "resolved" as "signed".
export function animatedClips(animation) {
  return Array.isArray(animation?.clip_playlist)
    ? animation.clip_playlist.filter((c) => c && c.landmark_clip_url)
    : [];
}

// What the result panel reports. Sentence-level playback signs the whole
// sentence with one clip, so the composed counter only applies when the
// per-gloss playlist drives the avatar.
export function signedCoverage(glossSequence, animation, landmarkUrl) {
  const total = Array.isArray(glossSequence) ? glossSequence.length : 0;
  const composed = !landmarkUrl && Array.isArray(animation?.clip_playlist);
  const signed = composed ? animatedClips(animation).length : 0;
  const missing = Array.isArray(animation?.unsupported_tokens)
    ? animation.unsupported_tokens
    : [];
  return { composed, total, signed, missing };
}
