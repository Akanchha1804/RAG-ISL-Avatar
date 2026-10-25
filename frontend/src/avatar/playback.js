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

// Playback speed for clip frames. The avatar's rigid hands read signs faster
// than the signer who recorded them, so the demo runs slightly under real
// time. 1.0 = the clip's recorded rate; 0.85 = 15% slower.
export const PLAYBACK_RATE = 0.85;

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

// Clip frames arrive at 25/30 Hz but the avatar renders at 60+ Hz. Snapping
// the retargeter to the clip rate and holding the pose between frames reads
// as a stutter; this samples the landmark stream at the render rate instead.
// Hands are matched by nearest wrist (the detection order and the tracker's
// handedness label are not stable), and a pair further apart than
// HAND_MATCH_MAX is not blended - that is a real cut, not motion.
const HAND_MATCH_MAX = 0.25;

const mix = (a, b, t) => a + (b - a) * t;

export function interpolateFrame(a, b, alpha) {
  if (!a) return b || null;
  if (!b || !Array.isArray(a.hands) || !Array.isArray(b.hands)) return a;
  const t = Math.max(0, Math.min(1, alpha));
  if (t === 0 || a.hands.length === 0 || b.hands.length === 0) return a;
  const used = new Set();
  const hands = a.hands.map((hand) => {
    const w0 = hand && hand.landmarks && hand.landmarks[0];
    if (!w0) return hand;
    let best = -1;
    let bestD = HAND_MATCH_MAX;
    for (let i = 0; i < b.hands.length; i += 1) {
      if (used.has(i)) continue;
      const other = b.hands[i];
      const w1 = other && other.landmarks && other.landmarks[0];
      if (!w1) continue;
      const d = Math.hypot(w0.x - w1.x, w0.y - w1.y);
      if (d <= bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best < 0) return hand;
    used.add(best);
    const other = b.hands[best];
    const landmarks = (hand.landmarks || []).map((p, i) => {
      const q = other.landmarks && other.landmarks[i];
      if (!p || !q) return p;
      return { ...p, x: mix(p.x, q.x, t), y: mix(p.y, q.y, t), z: mix(p.z, q.z, t) };
    });
    return { ...hand, landmarks };
  });
  const timestamp = typeof a.timestamp_ms === 'number' && typeof b.timestamp_ms === 'number'
    ? mix(a.timestamp_ms, b.timestamp_ms, t)
    : a.timestamp_ms;
  return { ...a, timestamp_ms: timestamp, hands };
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
