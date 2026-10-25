import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
import { apiUrl, resolveUrl } from '../config';
import {
  clipSwitchJump,
  expressionTargetsFor,
  interpolateFrame,
  PLAYBACK_RATE,
} from '../avatar/playback';

// Test/debug hooks (`HOOKS.__vrm`, `__vrmAdvance`, `__vrmPen`, `__vrmShot`,
// ...) let the retargeting be asserted from a browser: scene state, cross
// checks, stepped playback and screenshots. They are attached only in dev
// builds, so a production bundle exposes nothing on `window` and skips the
// per-frame debug object entirely.
const HOOKS = import.meta.env.DEV ? globalThis : Object.create(null);

// MediaPipe hand landmark indices (wrist, then 4 joints per finger).
const FINGERS = [
  { key: 'Index', lm: [5, 6, 7, 8] },
  { key: 'Middle', lm: [9, 10, 11, 12] },
  { key: 'Ring', lm: [13, 14, 15, 16] },
  { key: 'Little', lm: [17, 18, 19, 20] },
];

// VRM1 thumb chain (metacarpal/proximal/distal) driven by landmarks 1->2/2->3/3->4.
const THUMB_BONES = ['Metacarpal', 'Proximal', 'Distal'];
const THUMB_SEGS = [[1, 2], [2, 3], [3, 4]];

const SMOOTHNESS = 15;

function disposeObject(root) {
  try {
    root.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : o.material ? [o.material] : [];
      mats.forEach((m) => {
        Object.values(m).forEach((v) => {
          if (v && v.isTexture) v.dispose();
        });
        m.dispose();
      });
    });
  } catch {
    /* ignore */
  }
}

function lmVec(lm, out) {
  // Same calibration as Unity HandPoseMapper: invert Y and Z.
  out.set(lm.x, -lm.y, -lm.z);
  return out;
}

function basisFromPalm(p0, p5, across) {
  // `across` must be the knuckle line (index MCP -> pinky MCP), NOT
  // wrist -> middle MCP. Those two are near-parallel on a flat hand
  // (2-19 deg apart in the source data), so their cross product is
  // numerical noise and the wrist roll came out random frame to frame.
  const x = new THREE.Vector3().subVectors(p5, p0);
  if (x.lengthSq() < 1e-10 || across.lengthSq() < 1e-10) return null;
  x.normalize();
  const z = new THREE.Vector3().crossVectors(x, across);
  if (z.lengthSq() < 1e-10) return null;
  z.normalize();
  const y = new THREE.Vector3().crossVectors(z, x).normalize();
  x.crossVectors(y, z).normalize();
  return { x, y, z };
}

function quatFromBasis(b) {
  const m = new THREE.Matrix4().makeBasis(b.x, b.y, b.z);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

// ---- joint-safety helpers (unnatural-bend fixes) ----

// Clamp a flexion rotation (restDir -> targetDir) to maxRad. Returns the
// clamped delta quaternion. Degenerate axes pass through unclamped and are
// counted by the caller.
function clampFlexion(restDir, targetDir, maxRad, out) {
  const angle = restDir.angleTo(targetDir);
  if (!(angle > maxRad)) return null;
  const axis = new THREE.Vector3().crossVectors(restDir, targetDir);
  if (axis.lengthSq() < 1e-10) return null;
  axis.normalize();
  return out.setFromAxisAngle(axis, maxRad);
}

// Anatomical flexion limits (radians) by joint class.
const FLEX_LIMITS = {
  Proximal: (100 * Math.PI) / 180,
  Intermediate: (110 * Math.PI) / 180,
  Distal: (80 * Math.PI) / 180,
  ThumbDefault: (90 * Math.PI) / 180,
};
const ELBOW_MAX_RAD = (150 * Math.PI) / 180;

// Push a point outside a vertical torso capsule (hips->neck axis, radius r).
// Returns true when a push was applied (counted as a torso violation).
function clampOutsideCapsule(point, hipsPos, neckPos, radius, out) {
  _capA.subVectors(neckPos, hipsPos);
  const lenSq = _capA.lengthSq();
  if (lenSq < 1e-8) return false;
  const t = Math.min(1, Math.max(0, _capB.subVectors(point, hipsPos).dot(_capA) / lenSq));
  _capC.copy(hipsPos).addScaledVector(_capA, t);
  _capD.subVectors(point, _capC);
  const d = _capD.length();
  if (d >= radius || d < 1e-9) return d < 1e-9;
  out.copy(_capC).addScaledVector(_capD.normalize(), radius);
  return true;
}
const _capA = new THREE.Vector3();
const _capB = new THREE.Vector3();
const _capC = new THREE.Vector3();
const _capD = new THREE.Vector3();

// Torso LENGTH is a poor proxy for torso DEPTH. 0.24 * |hips->neck| gives
// this avatar a 0.113 m radius, but its chest actually reaches 0.175 m from
// the spine axis at y 1.40-1.50 - precisely the height the hands are driven
// to - so the clamp below happily parked the wrists ~3 cm INSIDE the bust and
// the torso mesh swallowed the palms ("the hands join and part of them
// disappears"). The clamp is also a single clamped POINT, and a 16 cm hand is
// not a point. Both are fixed by measuring the real geometry once at load.
const TORSO_PROFILE_BANDS = 24;
// The chest surface is sampled in a narrow slab around the spine axis: that
// is where a namaste hand ends up, and it avoids the shoulders/arms, which
// sit at the same heights out at |x| > 0.15.
const TORSO_MIDLINE_HALF = 0.06;
const HAND_CLEARANCE = 0.005;
// A hand is ~16 cm long, so the surface that matters is the furthest-forward
// torso point anywhere across that vertical span. Sampling only the wrist
// height both misses the bust apex and lets single-band dropouts undo the
// guard.
const HAND_SPAN_UP = 0.17;
const HAND_SPAN_DOWN = 0.04;
const HAND_SPAN_SAMPLES = 9;
// A hand box can project far along Z while edge-on; cap the standoff so a
// degenerate orientation can never fling the arm forward.
const HAND_BACK_MAX = 0.12;
// Mesh that hangs past the last finger joint.
const FINGER_TIP_MARGIN = 0.015;
// Sampled hand vertices kept for the penetration probe.
const PEN_POINT_MAX = 900;
const _handM = new THREE.Matrix4();
const _handV = new THREE.Vector3();
const _bodyV = new THREE.Vector3();
const _bodyV2 = new THREE.Vector3();

// One pass over the skinned meshes measures everything the collision code
// needs: the torso's forward surface per height band, each hand's vertex box
// in its own bone frame (so the guard can account for the hand's real volume),
// and the hand's thickness (the physical floor for the contact assist).
// Returns null when the rig lacks what it needs; the capsule then remains.
const measureBody = (vrm) => {
  const rawBone = (name) => {
    try {
      return vrm.humanoid.getRawBoneNode(name);
    } catch {
      return null;
    }
  };
  const hipsNode = rawBone('hips');
  const neckNode = rawBone('neck') || rawBone('head');
  if (!hipsNode || !neckNode) return null;
  vrm.scene.updateMatrixWorld(true);
  const hips = new THREE.Vector3();
  const neck = new THREE.Vector3();
  hipsNode.getWorldPosition(hips);
  neckNode.getWorldPosition(neck);
  const axisLen = neck.distanceTo(hips);
  if (!(axisLen > 1e-3)) return null;
  const handBones = {};
  const handInv = {};
  // Hand vertices kept for the penetration probe: every 8th vertex owned by
  // the hand or one of its finger/thumb bones, stored in ITS OWN bone's bind
  // frame, so the probe reproduces the posed mesh surface exactly.
  const penPts = { left: [], right: [] };
  const boneInv = new Map();
  for (const side of ['left', 'right']) {
    const hand = rawBone(side + 'Hand');
    if (!hand) continue;
    // Finger/thumb bones are children of the hand in every VRM rig, so a
    // subtree walk gives the full hand vertex domain.
    const set = new Set([hand]);
    hand.traverse((o) => {
      if (o.isBone) set.add(o);
    });
    handBones[side] = set;
    handInv[side] = new THREE.Matrix4().copy(hand.matrixWorld).invert();
  }
  // Bone-filtered torso set. Needed for the LATERAL half-width, where the
  // arms must be excluded (they occupy the same heights out at the sides),
  // and as a second source for the front depth. The front is ALSO measured
  // geometrically (midline slab) because the bust/clothing vertices are owned
  // by secondary J_Sec_* bones, so a bone filter alone leaves holes.
  const torsoBones = new Set();
  for (const name of ['hips', 'spine', 'chest', 'upperChest']) {
    const b = rawBone(name);
    if (b) torsoBones.add(b);
  }
  const frontZ = new Float32Array(TORSO_PROFILE_BANDS).fill(-Infinity);
  const halfX = new Float32Array(TORSO_PROFILE_BANDS);
  const boxMin = {};
  const boxMax = {};
  const boxN = {};
  const handBox = {};
  for (const side of ['left', 'right']) {
    if (!handBones[side]) continue;
    boxMin[side] = new THREE.Vector3(Infinity, Infinity, Infinity);
    boxMax[side] = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    boxN[side] = 0;
  }
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh || !o.skeleton || !o.geometry) return;
    const attr = o.geometry.attributes.position;
    if (!attr || !attr.count) return;
    const si = o.geometry.attributes.skinIndex;
    const sw = o.geometry.attributes.skinWeight;
    const bones = o.skeleton.bones;
    let hasHand = false;
    let hasTorso = false;
    for (const b of bones) {
      if (torsoBones.has(b)) hasTorso = true;
      if ((handBones.left && handBones.left.has(b))
        || (handBones.right && handBones.right.has(b))) hasHand = true;
    }
    for (let i = 0; i < attr.count; i++) {
      _bodyV.fromBufferAttribute(attr, i).applyMatrix4(o.matrixWorld);
      const t = (_bodyV.y - hips.y) / axisLen;
      const band = (t >= 0 && t < 1)
        ? Math.min(TORSO_PROFILE_BANDS - 1, Math.floor(t * TORSO_PROFILE_BANDS))
        : -1;
      if (band >= 0) {
        const axisX = hips.x + t * (neck.x - hips.x);
        const dx = Math.abs(_bodyV.x - axisX);
        if (dx < TORSO_MIDLINE_HALF && _bodyV.z > frontZ[band]) frontZ[band] = _bodyV.z;
      }
      if (!si || !sw) continue;
      // Dominant skin joint: the bone that actually owns this vertex.
      let ji = -1;
      let best = -1;
      for (let k = 0; k < 4; k++) {
        const w = k === 0 ? sw.getX(i) : k === 1 ? sw.getY(i) : k === 2 ? sw.getZ(i) : sw.getW(i);
        if (w > best) {
          best = w;
          ji = k === 0 ? si.getX(i) : k === 1 ? si.getY(i) : k === 2 ? si.getZ(i) : si.getW(i);
        }
      }
      const bm = ji >= 0 ? bones[ji] : null;
      if (!bm) continue;
      if (hasTorso && band >= 0 && torsoBones.has(bm)) {
        const axisX = hips.x + t * (neck.x - hips.x);
        const dx = Math.abs(_bodyV.x - axisX);
        if (dx > halfX[band]) halfX[band] = dx;
        if (_bodyV.z > frontZ[band]) frontZ[band] = _bodyV.z;
      }
      if (!hasHand) continue;
      for (const side of ['left', 'right']) {
        if (!handBones[side] || !handBones[side].has(bm)) continue;
        _bodyV2.copy(_bodyV).applyMatrix4(handInv[side]);
        boxMin[side].min(_bodyV2);
        boxMax[side].max(_bodyV2);
        boxN[side] += 1;
        if ((i % 8) === 0 && penPts[side].length < PEN_POINT_MAX) {
          let inv = boneInv.get(bm);
          if (!inv) {
            inv = new THREE.Matrix4().copy(bm.matrixWorld).invert();
            boneInv.set(bm, inv);
          }
          _bodyV2.copy(_bodyV).applyMatrix4(inv);
          penPts[side].push({
            b: bm,
            x: +_bodyV2.x.toFixed(4),
            y: +_bodyV2.y.toFixed(4),
            z: +_bodyV2.z.toFixed(4),
          });
        }
      }
    }
  });
  // Bands the torso mesh never reaches (neck gap, etc.) would otherwise
  // interpolate against -Infinity; carry the nearest real surface across.
  let seen = -Infinity;
  for (let i = 0; i < TORSO_PROFILE_BANDS; i++) {
    if (frontZ[i] === -Infinity) frontZ[i] = seen;
    else seen = frontZ[i];
  }
  seen = -Infinity;
  for (let i = TORSO_PROFILE_BANDS - 1; i >= 0; i--) {
    if (frontZ[i] === -Infinity) frontZ[i] = seen;
    else seen = frontZ[i];
  }
  if (frontZ[0] === -Infinity) return null;
  for (const side of ['left', 'right']) {
    if (!boxN[side]) continue;
    handBox[side] = {
      c: boxMin[side].clone().add(boxMax[side]).multiplyScalar(0.5),
      h: boxMax[side].clone().sub(boxMin[side]).multiplyScalar(0.5),
    };
  }
  return { profile: { hipsY: hips.y, axisLen, hipsX: hips.x, neckX: neck.x, frontZ, halfX }, handBox, penPts };
};

// Forward torso surface (world Z) at a world height, or null when the point
// is off the torso, where the capsule still owns the constraint.
// Scratch for torsoSurfaceAt: called twice a frame, so no allocation.
const _surf = { front: 0, halfX: 0, axisX: 0, span: 0 };
// The frontmost torso point and the torso half-width across the hand's whole
// vertical span, plus the spine axis' lateral position at the wrist height.
// Returns null when the wrist is off the torso (legs/head), where the
// capsule owns the constraint.
const torsoSurfaceAt = (prof, y) => {
  if (!prof || !prof.frontZ || !prof.frontZ.length) return null;
  const n = prof.frontZ.length;
  if (n < 2) return null;
  let front = -Infinity;
  let halfX = 0;
  let tMax = -1;
  for (let s = 0; s < HAND_SPAN_SAMPLES; s++) {
    const k = s / (HAND_SPAN_SAMPLES - 1);
    const yy = y - HAND_SPAN_DOWN + k * (HAND_SPAN_DOWN + HAND_SPAN_UP);
    const t = (yy - prof.hipsY) / prof.axisLen;
    if (!(t >= 0 && t <= 1)) continue;
    const f = t * (n - 1);
    const i = Math.min(n - 2, Math.floor(f));
    const j = f - i;
    const fz = prof.frontZ[i] * (1 - j) + prof.frontZ[i + 1] * j;
    if (fz > front) front = fz;
    const hx = prof.halfX[i] * (1 - j) + prof.halfX[i + 1] * j;
    if (hx > halfX) halfX = hx;
    if (t > tMax) tMax = t;
  }
  if (front === -Infinity) return null;
  const t0 = Math.min(1, Math.max(0, (y - prof.hipsY) / prof.axisLen));
  _surf.front = front;
  _surf.halfX = halfX;
  _surf.axisX = prof.hipsX + t0 * (prof.neckX - prof.hipsX);
  _surf.span = tMax;
  return _surf;
};

// How far the hand's real mesh reaches BEHIND the wrist (world -Z) right now.
// Built from the posed palm vertices and the posed finger joints, so it tracks
// the actual hand: a rest-pose bounding box cannot, because its splayed
// fingers over-estimate the across extent by ~2x and shove the hand a hand's
// width clear of the chest.
const handBackExtent = (st, side) => {
  const hand = st.bones && st.bones[side + 'Hand'];
  if (!hand) return 0;
  hand.node.updateWorldMatrix(true, true);
  _handM.copy(hand.node.matrixWorld);
  hand.node.getWorldPosition(_handV);
  const wristZ = _handV.z;
  let back = 0;
  const pts = st.penPts && st.penPts[side];
  if (pts) {
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      // Palm vertices are rigid to the hand bone, so this is exact.
      if (p.b !== hand.node) continue;
      _handV.set(p.x, p.y, p.z).applyMatrix4(_handM);
      const d = wristZ - _handV.z;
      if (d > back) back = d;
    }
  }
  hand.node.traverse((o) => {
    if (!o.isBone || o === hand.node) return;
    o.getWorldPosition(_handV);
    const d = wristZ - _handV.z + FINGER_TIP_MARGIN;
    if (d > back) back = d;
  });
  return Math.max(0, Math.min(back, HAND_BACK_MAX));
};

// Deepest palm-vertex intrusion into the measured torso surface, in metres.
// Mirrors the `__vrmPen` probe so the render loop can act on the same
// measurement instead of trusting the pre-pose estimate. Returns 0 when the
// hand is clear (or when there is no measured profile yet).
const handPenetration = (st, side) => {
  const prof = st.torsoProfile;
  const pts = st.penPts && st.penPts[side];
  if (!prof || !prof.frontZ || !pts || !pts.length) return 0;
  const n = prof.frontZ.length;
  let deepest = 0;
  for (let i = 0; i < pts.length; i += 1) {
    const p = pts[i];
    _handV.set(p.x, p.y, p.z).applyMatrix4(p.b.matrixWorld);
    const t = (_handV.y - prof.hipsY) / prof.axisLen;
    if (t < 0 || t > 1) continue;
    const f = t * (n - 1);
    const bi = Math.min(n - 2, Math.max(0, Math.floor(f)));
    const bk = f - bi;
    const axisX = prof.hipsX + t * (prof.neckX - prof.hipsX);
    const hx = prof.halfX[bi] * (1 - bk) + prof.halfX[bi + 1] * bk;
    if (Math.abs(_handV.x - axisX) > hx) continue;
    const fz = prof.frontZ[bi] * (1 - bk) + prof.frontZ[bi + 1] * bk;
    const d = fz - _handV.z;
    if (d > deepest) deepest = d;
  }
  return deepest;
};

// Signing-space calibration (M6 fix): source videos frame the signer
// differently (waist-level vs chest-level), so absolute image coords
// reproduce the wrong height on the avatar. Height (Y) and depth (Z) are
// therefore calibrated per clip from the wrist trajectory bbox. Min-span
// guarded: a static hold maps mid-box rather than amplifying jitter into
// full-box swings.
//
// The LATERAL axis (X) must NOT be calibrated per clip. Stretching a clip's
// own x-range across the box turns a compact pose into a full-width one: the
// namaste clip's wrists sit in a ~0.14-wide band near the centre, which the
// per-clip map blew out to +-0.275 m so the hands could never meet (closest
// approach 0.36 m). X uses a FIXED reference span and the image centre so
// absolute hand separation is preserved.
const CALIB_X_SPAN = 0.35;
// Vertical: same trap as the lateral axis, one dimension over. A compact
// hold (the namaste's wrists span only ~0.175 image-y) was being stretched
// across the whole 0.7 m box, i.e. ~4x gain, which parked the joined hands
// at chin height. Floor the span so gain never exceeds ~2x; wide clips
// (>=0.35 image-y) keep their own span and are untouched.
const CALIB_MIN_SPAN_Y = 0.35;
const CALIB_MIN_SPAN_Z = 0.05;

// Lateral gain: normalized x -> metres of lateral offset from the body midline.
const LAT_GAIN = 0.55;
// Contact assist: when the source holds the two hands close but not quite
// touching (namaste reads ~0.14 m apart after calibration), compress the
// pair's lateral separation toward their own midpoint so the palms actually
// meet. Ramped from CLASP_THRESHOLD (no effect) to CLASP_FULL (max), so
// shoulder-width and wider poses are left untouched.
const CLASP_THRESHOLD = 0.3;
const CLASP_FULL = 0.13;
const CLASP_MIN_SCALE = 0.35;

// Hand-to-arm assignment. The tracker's per-frame handedness label is not
// trustworthy on the shipped gloss clips: it flips mid-clip (140 of 184
// clips, 536 flips) and one physical hand is reported twice often enough to
// matter (114 of 184 clips). Grouping BY THE LABEL therefore drove the wrong
// arm across the body on a flip, and two detections of one hand fought over
// a single arm - the overlapping hands and the sign hopping between arms in
// composed playback. Detections are now clustered, then assigned to an arm
// by continuity with that arm's previous wrist; lateral position is only the
// fallback for a hand with no history yet (first frame of a clip).
const HAND_MERGE_DIST = 0.05;
const SIDE_SWITCH_HYST = 0.06;

// Idle arm pose. The model's rest pose is a T-pose, so a side with no
// landmarks (or the neutral a clip switch blends through) left at rest reads
// as a broken mannequin with one arm straight out. buildNeutralPose drives
// the same two-bone IK to a wrist target beside the body once at load and
// keeps the resulting quaternions. DROP/FWD are fractions of the arm length;
// LATERAL is the wrist's outward offset from the shoulder.
const IDLE_LATERAL = 0.09;
const IDLE_DROP = 0.78;
const IDLE_FWD = 0.16;
// Load-time convergence budget for the idle solve (per side, at 60 Hz).
const IDLE_SETTLE_STEPS = 240;

// A hand that disappears for a frame or two is a tracker dropout, not the
// sign ending: 164 such holes sit inside the shipped clips (median 2 frames,
// p90 7) and every one of them used to drop that arm to the idle and snap it
// back, which read as the arm pumping mid-sign. Hold the last pose for this
// long first; only a hand that stays missing really is out of the clip.
const HAND_GAP_HOLD = 0.5;

// Clip-switch blending. A composed sentence used to pass through the
// arms-down idle between every gloss (signer resetting after each word),
// which reads as a list of words rather than a sentence. Consecutive glosses
// from one signer are coarticulated: crossfade them directly when the hands
// do not jump. Measured over the curated sentences, 88% of gloss-to-gloss
// boundaries move a wrist less than this in normalised image space.
const SWITCH_JUMP_MAX = 0.4;

function calibrateSigningSpace(dataset) {
  const xs = [];
  const ys = [];
  const zs = [];
  for (const frame of (dataset && dataset.frames) || []) {
    for (const hand of frame.hands || []) {
      const w = hand.landmarks && hand.landmarks[0];
      if (!w) continue;
      xs.push(w.x);
      ys.push(w.y);
      zs.push(w.z);
    }
  }
  if (xs.length === 0) return null;
  // Percentile bbox: edge-of-frame mistracks must not define the range.
  const pct = (arr, q) => {
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(q * s.length))];
  };
  const minY = pct(ys, 0.02);
  const maxY = pct(ys, 0.98);
  const minZ = pct(zs, 0.02);
  const maxZ = pct(zs, 0.98);
  const fit = (lo, hi, minSpan) => {
    const span = Math.max(hi - lo, minSpan);
    const mid = (lo + hi) / 2;
    return [mid - span / 2, span];
  };
  // Lateral: fixed span centred on the image, so a compact pose stays compact.
  const x0 = 0.5 - CALIB_X_SPAN / 2;
  const spanX = CALIB_X_SPAN;
  const [y0, spanY] = fit(minY, maxY, CALIB_MIN_SPAN_Y);
  const [z0, spanZ] = fit(minZ, maxZ, CALIB_MIN_SPAN_Z);
  return { minX: x0, minY: y0, minZ: z0, spanX, spanY, spanZ };
}

function normalizeWrist(calib, w) {
  if (!calib) return { x: w.x, y: w.y, z: w.z };
  const clamp = (v) => Math.min(1.2, Math.max(-0.2, v));
  return {
    x: clamp((w.x - calib.minX) / calib.spanX),
    y: clamp((w.y - calib.minY) / calib.spanY),
    z: clamp((w.z - calib.minZ) / calib.spanZ),
  };
}

const _tmpQ = new THREE.Quaternion();
const _tmpV = new THREE.Vector3();
const _tmpV2 = new THREE.Vector3();

export default function VrmAvatar({ landmarkUrl, playlistUrls, glossSequence, onError }) {
  const mountRef = useRef(null);
  const onErrorRef = useRef(null);
  // Keep the latest onError without re-running the setup effect.
  useEffect(() => {
    onErrorRef.current = onError;
  });
  // Rendered status is derived (never set inside effects): the model
  // load flag and the landmark dataset change only in async callbacks.
  // landmarkData carries its URL so a stale dataset never renders as current.
  const [modelReady, setModelReady] = useState(false);
  const [modelError, setModelError] = useState(null);
  const [landmarkData, setLandmarkData] = useState(null);
  // Stable playlist identity: the URL array is rebuilt every render.
  const listKey = Array.isArray(playlistUrls) ? playlistUrls.filter(Boolean).join('|') : '';
  const signing = landmarkUrl && landmarkData && landmarkData.url === landmarkUrl;
  const playlistActive = !landmarkUrl && listKey && landmarkData && landmarkData.url === listKey;
  const status = modelError
    || (!landmarkUrl && !listKey
      ? (modelReady ? 'Avatar ready' : 'Loading avatar…')
      : (signing ? `Recorded sentence playback · ${landmarkData.frames} frames`
        : (playlistActive ? `Experimental gloss playback · ${landmarkData.clips} clips`
          : (modelReady ? 'Loading animation…' : 'Loading avatar…'))));

  const stateRef = useRef({
    renderer: null,
    scene: null,
    camera: null,
    vrm: null,
    bones: {}, // name -> { node, restLocal, restLocalDir }
    avBasis: {}, // left/right -> { x, y, z } rest palm basis (world)
    dataset: null,
    playlist: null,
    playIndex: 0,
    frameIndex: 0,
    frameTimer: 0,
    loadSeq: 0,
    modelRoot: null,
    // When the sequence ends/clears, ease every driven bone back to
    // its rest orientation instead of freezing mid-gesture.
    resetting: false,
    loadId: 0,
    disposed: false,
    exprTarget: { surprised: 0, angry: 0 },
    exprCurrent: { surprised: 0, angry: 0 },
    // Joint-safety violation counters (headless proof the fixes engage).
    viol: { curl: 0, elbow: 0, torso: 0, skipped: 0, dup: 0, wcap: 0, sideHold: 0 },
  });

  // Non-manual targets follow the gloss sequence (derived, never set in
  // the render loop). Keyed on the joined string so the effect only runs
  // when the sequence actually changes.
  const glossKey = Array.isArray(glossSequence) ? glossSequence.join('|') : '';
  useEffect(() => {
    // Recorded sentence clips keep a neutral face: the corpus has hand
    // landmarks only, so a synthetic brow marker would be invented on top of
    // the signer's recording. Composed gloss playback keeps the timed NMM.
    stateRef.current.exprTarget = landmarkUrl
      ? { surprised: 0, angry: 0 }
      : expressionTargetsFor(glossKey ? glossKey.split('|') : []);
  }, [glossKey, landmarkUrl]);

  // ---- one-time three.js setup + VRM load ----
  useEffect(() => {
    const st = stateRef.current;
    st.disposed = false;
    const mount = mountRef.current;
    if (!mount) return undefined;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    mount.appendChild(renderer.domElement);
    st.renderer = renderer;

    const scene = new THREE.Scene();
    // A little lighter than the old near-black: the avatar's black top needs
    // to separate from the backdrop, and the white hands need a readable
    // silhouette on a projector.
    scene.background = new THREE.Color('#1e293b');
    st.scene = scene;

    const camera = new THREE.PerspectiveCamera(30, 1.5, 0.1, 20);
    camera.position.set(0, 1.25, 2.3);
    camera.lookAt(0, 1.0, 0);
    st.camera = camera;

    // Size the canvas from the live layout (never a one-time guess) and
    // re-fit the signing-space framing whenever layout settles or changes.
    // Measuring once at setup runs before CSS/layout settles and can freeze
    // a tiny canvas with a far-away camera.
    const fitViewport = () => {
      if (!mount || !st.renderer || !st.camera) return;
      const w = mount.clientWidth || 720;
      const h = mount.clientHeight || 480;
      st.renderer.setSize(w, h);
      st.camera.aspect = w / h;
      const fb = st.frameBox;
      if (fb && fb.sizeY > 1e-6) {
        const halfFov = THREE.MathUtils.degToRad(st.camera.fov * 0.5);
        const spanY = fb.sizeY * 0.62;
        // Rest pose is a T-pose (~full arm span wide); fit it so the hands
        // are not amputated at the frame edges at idle.
        const spanX = fb.sizeY * 0.85;
        const dist = Math.max(
          spanY / (2 * Math.tan(halfFov)),
          spanX / (2 * Math.tan(halfFov) * st.camera.aspect),
        ) * 1.08;
        const targetY = fb.centerY + fb.sizeY * 0.5 - spanY * 0.5;
        st.frameInfo = { sizeY: fb.sizeY, targetY, dist };
        st.camera.position.set(fb.centerX, targetY, fb.centerZ + dist);
        st.camera.lookAt(fb.centerX, targetY, fb.centerZ);
      }
      st.camera.updateProjectionMatrix();
    };
    fitViewport();
    const ro = (typeof ResizeObserver !== 'undefined')
      ? new ResizeObserver(() => fitViewport())
      : null;
    if (ro) ro.observe(mount);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x475569, 1.5));
    const dir = new THREE.DirectionalLight(0xffffff, 2.0);
    dir.position.set(1.5, 2.5, 2.5);
    scene.add(dir);
    const fill = new THREE.DirectionalLight(0xbcd2ff, 0.7);
    fill.position.set(-2, 1.5, 1.5);
    scene.add(fill);
    // Rim light from behind: separates the arms/hair from the backdrop
    // without changing the animated pose or the avatar material.
    const rim = new THREE.DirectionalLight(0x93c5fd, 1.2);
    rim.position.set(-1.5, 2.4, -2.5);
    scene.add(rim);

    const clock = new THREE.Clock();
    let rafId = 0;
    // Load token: StrictMode (and HMR) re-runs this effect while a
    // previous VRM load is still in flight. Only the latest load may
    // attach its scene; stale loads are discarded so the model can
    // never be added twice (which previously left a frozen duplicate
    // avatar on screen that ignored all animation).
    const loadToken = ++st.loadSeq;

    const fail = (msg) => {
      if (st.disposed) return;
      setModelError(msg);
      if (onErrorRef.current) onErrorRef.current(msg);
    };

    const captureRestPose = (vrm) => {
      vrm.scene.updateMatrixWorld(true);
      const bones = {};
      const need = ['UpperArm', 'LowerArm', 'Hand', 'UpperLeg', 'LowerLeg', 'Foot', ...FINGERS.flatMap((f) => ['Proximal', 'Intermediate', 'Distal'].map((j) => f.key + j)), ...THUMB_BONES.map((j) => 'Thumb' + j)];
      // NOTE: RAW nodes, not normalized ones. The visible meshes are
      // skinned to the raw scene nodes; the normalized rig is a parallel
      // structure whose pose never reaches the mesh on its own. Driving
      // raw requires autoUpdateHumanBones=false (set at load), otherwise
      // the transfer overwrites our pose every frame.
      const rawNode = (name) => {
        try {
          return vrm.humanoid.getRawBoneNode(name);
        } catch {
          return null;
        }
      };
      for (const side of ['left', 'right']) {
        for (const part of need) {
          const name = side + part;
          const node = rawNode(name);
          if (!node) continue;
          const restLocal = node.quaternion.clone();
          const restWorld = new THREE.Quaternion();
          node.getWorldQuaternion(restWorld);
          // Reference direction for flexion mapping: child direction when
          // it is a real bone continuation; otherwise (tipless Distal
          // bones, nub end-sites) the incoming segment direction, so the
          // fingertip keeps following its parent instead of going rigid.
          // Bones with no usable reference are left null and SKIPPED at
          // drive time (a visibly wrong joint is worse than a still one).
          let restLocalDir = null;
          const child = node.children && node.children[0];
          if (child && node.parent) {
            const bonePos = new THREE.Vector3();
            const childPos = new THREE.Vector3();
            node.getWorldPosition(bonePos);
            child.getWorldPosition(childPos);
            const parentQ = new THREE.Quaternion();
            node.parent.getWorldQuaternion(parentQ);
            restLocalDir = childPos
              .sub(bonePos)
              .applyQuaternion(parentQ.invert());
            if (restLocalDir.lengthSq() > 1e-8) restLocalDir.normalize();
            else restLocalDir = null;
          }
          if (!restLocalDir && node.parent) {
            const bonePos = new THREE.Vector3();
            const parentPos = new THREE.Vector3();
            node.getWorldPosition(bonePos);
            node.parent.getWorldPosition(parentPos);
            const parentQ = new THREE.Quaternion();
            node.parent.getWorldQuaternion(parentQ);
            restLocalDir = bonePos
              .sub(parentPos)
              .applyQuaternion(parentQ.invert());
            if (restLocalDir.lengthSq() > 1e-8) restLocalDir.normalize();
            else restLocalDir = null;
          }
          bones[name] = { node, restLocal, restLocalDir, restWorld };
        }
      }
      // Midline torso bones (whole-body probe test: rotating hips must
      // move everything above the legs, unmistakable on screen).
      for (const name of ['hips', 'spine', 'chest', 'neck', 'head']) {
        const node = rawNode(name);
        if (!node) continue;
        bones[name] = { node, restLocal: node.quaternion.clone(), restLocalDir: null };
      }
      st.bones = bones;

      // Avatar rest palm basis per side (mirrors Unity BuildSideBasis),
      // then mirror-verified: each side's palm normal (z) must point AWAY
      // from the body midline. A mirrored model/rig would otherwise drive
      // one hand's fingers twisted backward (invisible in telemetry, which
      // only proves motion exists). Correction is a 180-degree roll about
      // the finger axis (keeps a proper rotation, no reflection).
      const avBasis = {};
      st.palmMirrored = {};
      const midRef = new THREE.Vector3();
      {
        const midNode = rawNode('chest') || rawNode('spine') || rawNode('neck');
        if (midNode) midNode.getWorldPosition(midRef);
      }
      for (const side of ['left', 'right']) {
        const hand = bones[side + 'Hand'];
        const index = bones[side + 'IndexProximal'];
        const little = bones[side + 'LittleProximal'];
        if (!hand || !index || !little) continue;
        const o = new THREE.Vector3();
        const pi = new THREE.Vector3();
        const pl = new THREE.Vector3();
        hand.node.getWorldPosition(o);
        index.node.getWorldPosition(pi);
        little.node.getWorldPosition(pl);
        // Knuckle line (index MCP -> little MCP), matching basisFromPalm.
        // wrist -> middle MCP is near-parallel to the fingers in this rest
        // pose, which made the palm normal noise.
        const across = pl.clone().sub(pi);
        const toIndex = pi.clone().sub(o);
        if (toIndex.lengthSq() < 1e-8 || across.lengthSq() < 1e-8) continue;
        const x = toIndex.clone().normalize();
        const z = new THREE.Vector3().crossVectors(toIndex, across);
        if (z.lengthSq() < 1e-8) continue;
        z.normalize();
        const y = new THREE.Vector3().crossVectors(z, x).normalize();
        x.crossVectors(y, z).normalize();
        const outward = o.clone().sub(midRef);
        outward.y = 0;
        if (outward.lengthSq() < 1e-8) outward.set(side === 'left' ? 1 : -1, 0, 0);
        outward.normalize();
        // Only correct a CLEARLY inward palm normal. In the imported rest
        // pose the raw cross product is nearly perpendicular to `outward`
        // (dot ~ 0), where a bare `< 0` test flips on numeric noise and
        // silently twists one hand 180 degrees (Unity parity).
        const inward = z.dot(outward);
        if (inward < -0.35) {
          y.negate();
          z.negate();
          st.palmMirrored[side] = true;
        }
        avBasis[side] = { x, y, z };
      }
      st.avBasis = avBasis;

      // Thumb audit: which thumb bones exist AND have a usable reference.
      // Reported in telemetry; a rigid thumb is otherwise silent.
      st.thumbOk = 0;
      for (const side of ['left', 'right']) {
        for (const j of THUMB_BONES) {
          const entry = bones[side + 'Thumb' + j];
          if (entry && entry.node && entry.restLocalDir) st.thumbOk += 1;
        }
      }

      // Arm rig per side for two-bone IK: the landmark files carry hand
      // points only, so without arms the hands would sign down at the
      // hips. The wrist target below carries each hand up into the
      // signing space in front of the chest; elbows point down/out.
      let chestPos = null;
      for (const n of ['chest', 'spine', 'neck']) {
        const node = rawNode(n);
        if (node) {
          chestPos = new THREE.Vector3();
          node.getWorldPosition(chestPos);
          break;
        }
      }
      // Torso capsule endpoints for collision (wrist/elbow clamp).
      // Static rest pose: the torso never animates in this pipeline.
      const hipsNode = rawNode('hips');
      const neckNode = rawNode('neck') || rawNode('head');
      if (hipsNode && neckNode) {
        st.torsoHips = new THREE.Vector3();
        st.torsoNeck = new THREE.Vector3();
        hipsNode.getWorldPosition(st.torsoHips);
        neckNode.getWorldPosition(st.torsoNeck);
        const torsoLen = st.torsoNeck.distanceTo(st.torsoHips);
        st.torsoRadius = Math.max(0.1, torsoLen * 0.24);
      } else {
        st.torsoHips = null;
        st.torsoNeck = null;
        st.torsoRadius = 0.15;
      }
      const arm = {};
      for (const side of ['left', 'right']) {
        const upper = bones[side + 'UpperArm'];
        const lower = bones[side + 'LowerArm'];
        const hand = bones[side + 'Hand'];
        if (!upper || !lower || !hand) continue;
        const S = new THREE.Vector3();
        const E = new THREE.Vector3();
        const W = new THREE.Vector3();
        upper.node.getWorldPosition(S);
        lower.node.getWorldPosition(E);
        hand.node.getWorldPosition(W);
        const restWQU = new THREE.Quaternion();
        const restWQL = new THREE.Quaternion();
        upper.node.getWorldQuaternion(restWQU);
        lower.node.getWorldQuaternion(restWQL);
        const restDirU = E.clone().sub(S);
        const restDirL = W.clone().sub(E);
        if (restDirU.lengthSq() < 1e-8 || restDirL.lengthSq() < 1e-8) continue;
        restDirU.normalize();
        restDirL.normalize();
        if (!chestPos) chestPos = S.clone().add(new THREE.Vector3(0, 0.3, 0));
      // Elbow hint: outward from chest, down, slightly FORWARD. A
      // backward-biased pole parks elbows behind the torso plane, where
      // the torso mesh swallows the upper arms (arms must read in front).
      const outward = S.clone().sub(chestPos);
      outward.y = 0;
      if (outward.lengthSq() < 1e-8) outward.set(side === 'left' ? 1 : -1, 0, 0);
      outward.normalize();
      const pole = outward.multiplyScalar(0.45).add(new THREE.Vector3(0, -1, 0.2)).normalize();
        arm[side] = {
          upper, lower,
          S: S.clone(),
          lenU: E.distanceTo(S),
          lenL: W.distanceTo(E),
          restWQU, restWQL, restDirU, restDirL, pole,
        };
      }
      st.arm = arm;
      st.chestPos = chestPos;
      return Object.keys(bones).length;
    };

    // NOTE: dt comes from the frame loop's single clock.getDelta() call.
    // mul damps noisy end joints (fingertips jitter most): 1 = full rate.
    const smoothWithDt = (node, target, dt, mul = 1) => {
      const t = (1 - Math.exp(-SMOOTHNESS * Math.max(dt, 0))) * mul;
      node.quaternion.slerp(target, Math.min(t, 1));
    };

    // Crossfade across clip switches / loop wraps. The transition passes
    // through a near-neutral pose (first half: current -> rest, second
    // half: rest -> live) like a signer resetting between signs. A direct
    // joint-space slerp between distant poses swings hands THROUGH the
    // face and out to wingspans, which reads as creepy.
    const BLEND_SWITCH_S = 0.35;
    const BLEND_LOOP_S = 0.15;
    const beginBlend = (viaRest = true) => {
      if (st.probe || !st.vrm) return;
      const pose = {};
      for (const key of Object.keys(st.bones)) {
        const entry = st.bones[key];
        if (entry && entry.node) pose[key] = entry.node.quaternion.clone();
      }
      st.blend = { t: 0, pose, viaRest };
    };
    // Neutral pose for a bone: the arms-down idle for the arm bones (built
    // once at load, see buildNeutralPose), the model rest pose for everything
    // else. Blending a clip switch to the rest pose popped the arms out to a
    // full T-pose between every gloss of a composed sentence.
    const neutralFor = (key, entry) => (
      (st.neutral && st.neutral[key]) || entry.restLocal
    );

    // Distance the hands travel into the next clip is measured by
    // clipSwitchJump() in ../avatar/playback (tested without a renderer).
    const updateBlend = (dt) => {
      const b = st.blend;
      if (!b || st.probe) return;
      const dur = b.viaRest === false ? BLEND_LOOP_S : BLEND_SWITCH_S;
      b.t += Math.max(dt, 0) / dur;
      const x = Math.min(b.t, 1);
      if (x >= 1) {
        st.blend = null;
        return;
      }
      if (b.viaRest === false) {
        // Loop wrap: short direct ease, no neutral dip.
        const k = x * x * (3 - 2 * x);
        for (const key of Object.keys(b.pose)) {
          const entry = st.bones[key];
          if (!entry || !entry.node) continue;
          entry.node.quaternion.slerpQuaternions(
            b.pose[key], entry.node.quaternion.clone(), k);
        }
        return;
      }
      // Clip switch: first half -> neutral, second half neutral -> live.
      const half = x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
      for (const key of Object.keys(b.pose)) {
        const entry = st.bones[key];
        if (!entry || !entry.node || !entry.restLocal) continue;
        const neutral = neutralFor(key, entry);
        if (x >= 1) continue;
        if (x < 0.5) {
          entry.node.quaternion.slerpQuaternions(
            b.pose[key], neutral, half);
        } else {
          entry.node.quaternion.slerpQuaternions(
            neutral, entry.node.quaternion.clone(), half);
        }
      }
      if (x >= 1) st.blend = null;
    };

    const _flexQ = new THREE.Quaternion();
    const applyFingerBone = (entry, segDirWorld, lmB, avB, dt, maxFlex, damp = 1,
      handDelta = null) => {
      if (!entry || !entry.restLocalDir) return;
      const { node, restLocal, restLocalDir } = entry;
      if (segDirWorld.lengthSq() < 1e-12) return;
      const inLm = _tmpV
        .set(
          segDirWorld.dot(lmB.x),
          segDirWorld.dot(lmB.y),
          segDirWorld.dot(lmB.z),
        )
        .normalize();
      const targetWorld = _tmpV2
        .set(0, 0, 0)
        .addScaledVector(avB.x, inLm.x)
        .addScaledVector(avB.y, inLm.y)
        .addScaledVector(avB.z, inLm.z);
      // `avB` is the hand basis captured in the REST world frame, but the
      // wrist has since been rolled by the palm mapping. Without carrying
      // that rotation across, the finger bones are aimed in the rest frame
      // and simply do not follow the hand: a rolled wrist left the fingers
      // hanging ~90 deg off the new palm axis (measurable as a huge
      // finger-joint bend with a straight source finger).
      if (handDelta) targetWorld.applyQuaternion(handDelta);
      node.parent.updateWorldMatrix(true, false);
      const parentQ = node.parent.getWorldQuaternion(_tmpQ);
      const targetLocal = targetWorld.applyQuaternion(parentQ.invert()).normalize();
      if (targetLocal.lengthSq() < 1e-8) return;
      // Anatomical clamp: a segment can never fold past its joint limit
      // from rest, no matter what the tracker reports.
      let delta;
      if (restLocalDir.angleTo(targetLocal) > maxFlex) {
        st.viol.curl += 1;
        const clamped = clampFlexion(restLocalDir, targetLocal, maxFlex, _flexQ);
        if (!clamped) {
          smoothWithDt(node, restLocal, dt, damp);
          return;
        }
        delta = clamped.multiply(restLocal);
      } else {
        delta = new THREE.Quaternion().setFromUnitVectors(restLocalDir, targetLocal).multiply(restLocal);
      }
      smoothWithDt(node, delta, dt, damp);
    };

    // Two-bone IK: carry the wrist to the landmark wrist trajectory
    // mapped into a signing box in front of the chest. Joint-safe:
    // adaptive pole (no flips on cross-body targets), torso-capsule
    // collision (no hands inside the chest), elbow hinge clamp (no
    // hyperextension), twist-locked upper arm (no free roll).
    const applyArmAt = (side, target, dt, snap = false, allowBend = true) => {
      const A = st.arm && st.arm[side];
      const C = st.chestPos;
      if (!A || !C) return;
      // Torso collision: the signing box overlaps the chest volume, so
      // clamp the wrist target outside the torso capsule (+3cm standoff).
      if (st.torsoHips && st.torsoNeck) {
        if (clampOutsideCapsule(target, st.torsoHips, st.torsoNeck,
          st.torsoRadius + 0.03, target)) st.viol.torso += 1;
      }
      const toT = target.clone().sub(A.S);
      let dist = toT.length();
      let maxReach = A.lenU + A.lenL - 0.02;
      // A hand target low over the torso (belly/chest) is often at the edge
      // of this avatar's short reach. A straight arm then lies against the
      // body and the elbow/upper arm disappear into it. Keep a real elbow
      // bend for those targets; the hand stays in front of the torso.
      if (allowBend && st.torsoProfile) {
        const surfR = torsoSurfaceAt(st.torsoProfile, target.y);
        // Only hands well inside the torso silhouette need the extra elbow
        // bend. A hand at the side of the waist is clear and must keep its
        // natural reach (this also protects the built neutral/idle pose).
        if (surfR && Math.abs(target.x - surfR.axisX) < surfR.halfX - 0.02) {
          maxReach = Math.min(maxReach, (A.lenU + A.lenL) * 0.75);
        }
      }
      if (dist > maxReach) {
        dist = maxReach;
        target.copy(A.S).addScaledVector(toT.normalize(), dist);
      }
      if (dist < 1e-6) return;
      // Chest-surface guard, applied AFTER the reach clamp so pulling the arm
      // back in cannot undo it. The capsule above constrains the wrist POINT
      // with a radius this avatar's bust exceeds by 3 cm; this constrains the
      // whole hand box against the measured surface, so the palm stays in
      // front of the chest instead of sinking into it. `toT` is refreshed
      // because the pole below derives from it.
      if (st.torsoProfile) {
        const surf = torsoSurfaceAt(st.torsoProfile, target.y);
        // Only over the torso: a hand out at the side is nowhere near the
        // chest surface and must not be shoved forward.
        if (surf && Math.abs(target.x - surf.axisX) < surf.halfX + 0.02) {
          const back = handBackExtent(st, side);
          const need = surf.front + HAND_CLEARANCE + back;
          if (target.z < need) {
            target.z = need;
            st.viol.chest = (st.viol.chest || 0) + 1;
            // `toT` feeds the pole AND `dist` feeds the shoulder angle, so
            // both must be refreshed here: a stale `dist` aims the upper arm
            // for the old target and the wrist lands short of the new one.
            toT.copy(target).sub(A.S);
            dist = toT.length();
          }
        }
      }
      st.armTgt = { side, t: target.toArray() };
      const dirST = toT.normalize();
      const cosA = Math.min(1, Math.max(-1,
        (A.lenU * A.lenU + dist * dist - A.lenL * A.lenL) / (2 * A.lenU * dist)));
      const angA = Math.acos(cosA);
      // Adaptive pole: bias downward as the target crosses the midline so
      // the elbow folds naturally instead of flaring or flipping.
      const sideSign = side === 'left' ? 1 : -1;
      const cross = Math.min(1, Math.max(0, -sideSign * (target.x - C.x) / 0.3));
      let pole = A.pole.clone()
        .addScaledVector(_DOWN, cross * 0.8);
      // A low hand target (belly/chest) otherwise parks the elbow behind the
      // bust: the torso swallows the upper arm and the forearm crosses the
      // body. For low targets, use a strongly outward pole so the elbow
      // stays at the side and the forearm reads in front.
      const low = allowBend
        ? Math.min(1, Math.max(0, (C.y + 0.12 - target.y) / 0.3))
        : 0;
      if (low > 0.05) {
        const out = A.S.clone().sub(C);
        out.y = 0;
        if (out.lengthSq() > 1e-8) {
          const sidePole = out.normalize().multiplyScalar(1)
            .add(new THREE.Vector3(0, -0.25, 0.15)).normalize();
          pole.normalize().lerp(sidePole, low).normalize();
        }
      }
      const perp = pole.clone().addScaledVector(dirST, -pole.dot(dirST));
      if (perp.lengthSq() < 1e-8) perp.set(0, -1, 0);
      perp.normalize();
      const upperDir = dirST.clone().multiplyScalar(Math.cos(angA))
        .addScaledVector(perp, Math.sin(angA)).normalize();
      // Forward guard: the upper arm must never point clearly behind the
      // torso plane (VRM faces +Z) or the mesh disappears into the chest.
      // Threshold sits just behind neutral so relaxed hanging arms (z≈0)
      // pass through untouched and only true backward aims are fixed.
      if (upperDir.z < -0.03) {
        upperDir.z = -0.03;
        upperDir.normalize();
        st.viol.elbow += 1;
      }
      const elbowPos = A.S.clone().addScaledVector(upperDir, A.lenU);
      // Elbow collision: keep the joint out of the torso too.
      if (st.torsoHips && st.torsoNeck) {
        if (clampOutsideCapsule(elbowPos, st.torsoHips, st.torsoNeck,
          st.torsoRadius, elbowPos)) st.viol.torso += 1;
      }
      let foreDir = target.clone().sub(elbowPos).normalize();
      // Elbow hinge clamp (max 150 deg): hyperextended elbows read as broken.
      const flexAng = upperDir.angleTo(foreDir);
      if (flexAng > ELBOW_MAX_RAD) {
        st.viol.elbow += 1;
        const hinge = new THREE.Vector3().crossVectors(upperDir, foreDir);
        if (hinge.lengthSq() > 1e-10) {
          hinge.normalize();
          foreDir = upperDir.clone()
            .applyAxisAngle(hinge, ELBOW_MAX_RAD).normalize();
        }
      }

      A.upper.node.parent.updateWorldMatrix(true, false);
      const parentQU = new THREE.Quaternion();
      A.upper.node.parent.getWorldQuaternion(parentQU);
      // Twist-locked swing: build rest/target frames around the pole-side
      // axis so upper-arm roll is determined, not free. Falls back to the
      // minimal swing when the pole side degenerates.
      const restPoleSide = A.pole.clone()
        .addScaledVector(A.restDirU, -A.pole.dot(A.restDirU));
      const poleSide = pole.clone()
        .addScaledVector(upperDir, -pole.dot(upperDir));
      let qU;
      if (restPoleSide.lengthSq() > 1e-8 && poleSide.lengthSq() > 1e-8) {
        restPoleSide.normalize();
        poleSide.normalize();
        const restSide2 = new THREE.Vector3()
          .crossVectors(A.restDirU, restPoleSide).normalize();
        const targSide2 = new THREE.Vector3()
          .crossVectors(upperDir, poleSide).normalize();
        const qRest = quatFromBasis(
          { x: A.restDirU, y: restPoleSide, z: restSide2 });
        const qTarg = quatFromBasis(
          { x: upperDir, y: poleSide, z: targSide2 });
        qU = qTarg.multiply(qRest.invert()).multiply(A.restWQU);
      } else {
        qU = new THREE.Quaternion()
          .setFromUnitVectors(A.restDirU, upperDir).multiply(A.restWQU);
      }
      const localU = parentQU.invert().multiply(qU);
      if (snap) A.upper.node.quaternion.copy(localU);
      else smoothWithDt(A.upper.node, localU, dt);

      A.lower.node.parent.updateWorldMatrix(true, false);
      const parentQL = new THREE.Quaternion();
      A.lower.node.parent.getWorldQuaternion(parentQL);
      const qL = new THREE.Quaternion()
        .setFromUnitVectors(A.restDirL, foreDir).multiply(A.restWQL);
      const localL = parentQL.invert().multiply(qL);
      if (snap) A.lower.node.quaternion.copy(localL);
      else smoothWithDt(A.lower.node, localL, dt);
    };

    // Map a landmark wrist through the signing box and hand it to the IK.
    const applyArm = (side, wristRaw, dt, calib) => {
      const C = st.chestPos;
      if (!C) return;
      const w = normalizeWrist(calib, wristRaw);
      // Lateral gain is narrower than vertical: the avatar's arms are
      // short relative to a full-box sweep, which read as stiff full
      // wingspans. ±0.275m lateral keeps elbows bent on wide signs.
      const clasp = (st.claspAdj && st.claspAdj[side]) || 0;
      const target = new THREE.Vector3(
        C.x + (w.x - 0.5) * LAT_GAIN + clasp,
        C.y + 0.02 + (0.5 - w.y) * 0.7,
        Math.max(C.z + 0.3 - w.z * 0.4, C.z + 0.06),
      );
      applyArmAt(side, target, dt);
    };

    // Where a signer's hand rests between signs: hanging just outside and
    // below the shoulder. Expressed against this arm's own length so the
    // idle stays sane if the model is ever swapped.
    const idleWristTarget = (side) => {
      const A = st.arm && st.arm[side];
      if (!A) return null;
      const L = A.lenU + A.lenL;
      const s = side === 'left' ? 1 : -1;
      return new THREE.Vector3(
        A.S.x + s * IDLE_LATERAL * L,
        A.S.y - IDLE_DROP * L,
        A.S.z + IDLE_FWD * L,
      );
    };

    // Relax one arm to the idle pose: used for a side with no landmarks this
    // frame. Freezing it wherever the previous clip left it stranded a hand
    // in mid-air - and, when the labels churned, on top of the signing hand.
    const relaxArm = (side, dt) => {
      const A = st.arm && st.arm[side];
      const idle = st.idle && st.idle[side];
      if (!A || !idle) return;
      smoothWithDt(A.upper.node, idle.upper, dt);
      smoothWithDt(A.lower.node, idle.lower, dt);
      const hand = st.bones[side + 'Hand'];
      if (hand) smoothWithDt(hand.node, hand.restLocal, dt, 0.6);
      for (const f of FINGERS) {
        for (const j of ['Proximal', 'Intermediate', 'Distal']) {
          const e = st.bones[side + f.key + j];
          if (e) smoothWithDt(e.node, e.restLocal, dt, 0.4);
        }
      }
      for (const j of THUMB_BONES) {
        const e = st.bones[side + 'Thumb' + j];
        if (e) smoothWithDt(e.node, e.restLocal, dt, 0.4);
      }
    };

    // Ease a side down only once it has been missing for longer than a
    // dropout: short holes hold the last pose, so the arm neither pumps
    // between the sign and the idle nor drops mid-sign.
    const settleArm = (side, dt) => {
      st.gap = st.gap || { left: 0, right: 0 };
      st.gap[side] = (st.gap[side] || 0) + dt;
      if (st.gap[side] > HAND_GAP_HOLD) relaxArm(side, dt);
    };

    // A side that has landmarks this frame is being driven: clear its gap
    // clock so the next dropout starts from zero.
    const driveArm = (side) => {
      st.gap = st.gap || { left: 0, right: 0 };
      st.gap[side] = 0;
    };

    // Solve the arms-down idle once, through the same IK the signs use, so it
    // obeys the capsule/chest clamps and the reach limit instead of fighting
    // them. Iterating the smoothed IK to convergence is the cheapest way to
    // get a pose that cannot disagree with the live one.
    const buildNeutralPose = () => {
      if (!st.arm || !st.chestPos) return;
      const idle = {};
      const neutral = {};
      for (const side of ['left', 'right']) {
        const A = st.arm[side];
        const target = idleWristTarget(side);
        if (!A || !target) continue;
        for (let i = 0; i < IDLE_SETTLE_STEPS; i += 1) {
          // The idle target is deliberately close to the torso side; the
          // signing-arm bend heuristic must not drag it up onto the chest.
          applyArmAt(side, target, 1 / 60, false, false);
        }
        idle[side] = {
          upper: A.upper.node.quaternion.clone(),
          lower: A.lower.node.quaternion.clone(),
        };
        neutral[side + 'UpperArm'] = idle[side].upper;
        neutral[side + 'LowerArm'] = idle[side].lower;
      }
      st.idle = idle;
      st.neutral = neutral;
    };

    // A new clip starts from its own first frame, not from wherever the
    // previous clip left the arms. Holding the old pose for HAND_GAP_HOLD is
    // correct for a tracking dropout mid-clip, but on a clip switch it leaves
    // the old signing hand on the chest for half a second. Snap both arms to
    // the built idle pose and clear the tracking history instead.
    const snapArmsToIdle = () => {
      if (!st.idle || !st.arm) return false;
      for (const side of ['left', 'right']) {
        const A = st.arm[side];
        const idle = st.idle[side];
        if (!A || !idle) continue;
        A.upper.node.quaternion.copy(idle.upper);
        A.lower.node.quaternion.copy(idle.lower);
        const hand = st.bones[side + 'Hand'];
        if (hand) hand.node.quaternion.copy(hand.restLocal);
        for (const f of FINGERS) {
          for (const j of ['Proximal', 'Intermediate', 'Distal']) {
            const e = st.bones[side + f.key + j];
            if (e) e.node.quaternion.copy(e.restLocal);
          }
        }
        for (const j of THUMB_BONES) {
          const e = st.bones[side + 'Thumb' + j];
          if (e) e.node.quaternion.copy(e.restLocal);
        }
      }
      st.gap = { left: HAND_GAP_HOLD + 1, right: HAND_GAP_HOLD + 1 };
      st.lastWrist = {};
      st.lastWristQ = {};
      st.lastSingleSide = null;
      st.claspAdj = null;
      st.claspInfo = null;
      if (st.vrm) st.vrm.scene.updateMatrixWorld(true);
      return true;
    };

    const _DOWN = new THREE.Vector3(0, -1, 0);

    const applyHand = (hand, isLeft, dt, calib) => {
      const lms = hand.landmarks;
      if (!lms || lms.length !== 21) return;
      const side = isLeft ? 'left' : 'right';
      // Arms first so the hands ride up into signing space.
      applyArm(side, lms[0], dt, calib);
      const avB = st.avBasis[side];
      if (!avB) return;
      const P = (i) => lmVec(lms[i], new THREE.Vector3());
      const p0 = P(0);
      const p5 = P(5);
      const p17 = P(17);
      const across = new THREE.Vector3().subVectors(p17, p5);
      let lmB = basisFromPalm(p0, p5, across);
      // Degenerate frame (collinear finger roots): keep the LAST good
      // palm frame instead of freezing the wrist while the arm keeps
      // moving (detached/flailing look) — but only while the arm is still
      // near where that basis was captured. A stale basis on a traveled
      // arm kinks the wrist sharply; then skip the frame instead.
      // Falls back to skip when no good frame has ever been seen.
      st.lastLmB = st.lastLmB || {};
      const tgtNow = (st.armTgt && st.armTgt.side === side) ? st.armTgt.t : null;
      if (lmB) {
        st.lastLmB[side] = {
          b: { x: lmB.x.clone(), y: lmB.y.clone(), z: lmB.z.clone() },
          at: tgtNow ? [...tgtNow] : null,
        };
      } else if (st.lastLmB[side]) {
        const rec = st.lastLmB[side];
        if (rec.at && tgtNow) {
          const moved = Math.hypot(
            tgtNow[0] - rec.at[0], tgtNow[1] - rec.at[1], tgtNow[2] - rec.at[2]);
          if (moved > 0.15) return;
        }
        lmB = rec.b;
      } else {
        return;
      }
      // Clasp palm roll: a landmark basis can only ever say "flat to the
      // camera", so a palms-together pose leaves the wrists un-pronated and
      // both palms pointing at the viewer. When the hands are clasped,
      // override the palm normal so each palm faces the other hand.
      let lmBWrist = lmB;
      if (st.claspAdj) {
        // Chirality note (measured on the imported rest pose): the basis z
        // is the BACK-of-hand normal on the left and the PALM normal on the
        // right, so one target direction turns both palms inward.
        const tgt = new THREE.Vector3(1, 0, 0);
        tgt.addScaledVector(lmB.x, -tgt.dot(lmB.x));
        if (tgt.lengthSq() > 0.05) {
          tgt.normalize();
          lmBWrist = {
            x: lmB.x.clone(),
            y: new THREE.Vector3().crossVectors(tgt, lmB.x).normalize(),
            z: tgt,
          };
        }
      }
      const qAv = quatFromBasis(avB);
      const qLm = quatFromBasis(lmBWrist);
      const palmDelta = qLm.multiply(qAv.invert());

      const handEntry = st.bones[side + 'Hand'];
      let handDelta = null;
      if (handEntry && handEntry.node.parent) {
        // palmDelta is a WORLD-space rotation: rotate the hand's rest
        // WORLD orientation, then express the result in the CURRENT parent
        // (lower-arm) space. Applying it to the local quaternion - as
        // before - conjugated the wrist by the arm's rotation, so once the
        // arm was posed by IK the palm pointed the wrong way and the hands
        // could not face each other. Parent world matrix is refreshed first.
        handEntry.node.parent.updateWorldMatrix(true, false);
        const parentQ = new THREE.Quaternion();
        handEntry.node.parent.getWorldQuaternion(parentQ);
        const target = parentQ
          .invert()
          .multiply(palmDelta.clone().multiply(handEntry.restWorld ?? handEntry.restLocal));
        // Angular-velocity cap: tracker basis flips read as wrist snaps
        // (a >~7 rad/s single-frame jump is never human). Clamp the step
        // instead of jumping; normal signing passes through untouched.
        st.lastWristQ = st.lastWristQ || {};
        const prevQ = st.lastWristQ[side];
        const maxStep = 7 * Math.max(dt, 1e-3);
        let goal = target;
        if (prevQ) {
          const step = prevQ.angleTo(target);
          if (step > maxStep) {
            st.viol.wcap = (st.viol.wcap || 0) + 1;
            st.wristCap = st.wristCap || {};
            st.wristCap[side] = (st.wristCap[side] || 0) + 1;
            goal = prevQ.clone().rotateTowards(target, maxStep);
          }
        }
        st.lastWristQ[side] = goal.clone();
        smoothWithDt(handEntry.node, goal, dt);
        // The hand's ACTUAL world rotation change this frame, i.e. after the
        // velocity cap and smoothing, so the fingers stay welded to the hand.
        handEntry.node.updateWorldMatrix(true, false);
        const handWorld = handEntry.node.getWorldQuaternion(new THREE.Quaternion());
        const restWorldQ = handEntry.restWorld ?? handEntry.restLocal;
        handDelta = restWorldQ
          ? handWorld.multiply(restWorldQ.clone().invert())
          : null;
      }

      const seg = (a, b) => {
        const va = P(a);
        const vb = P(b);
        return vb.sub(va);
      };
      for (const f of FINGERS) {
        const joints = ['Proximal', 'Intermediate', 'Distal'];
        for (let j = 0; j < 3; j++) {
          const entry = st.bones[side + f.key + joints[j]];
          // Distal tips jitter most in tracking: smooth them harder.
          applyFingerBone(entry, seg(f.lm[j], f.lm[j + 1]), lmB, avB, dt,
            FLEX_LIMITS[joints[j]], joints[j] === 'Distal' ? 0.45 : 1, handDelta);
        }
      }
      for (let j = 0; j < 3; j++) {
        const entry = st.bones[side + 'Thumb' + THUMB_BONES[j]];
        const [a, b] = THUMB_SEGS[j];
        // Thumb saddle joint over-rotates: damp the metacarpal, soften tips.
        const damp = j === 0 ? 0.6 : (j === 2 ? 0.45 : 0.8);
        applyFingerBone(entry, seg(a, b), lmB, avB, dt, FLEX_LIMITS.ThumbDefault, damp, handDelta);
      }

      // The forward guard runs before the wrist/palm is re-oriented, so a
      // hand that turns its palm into the chest can still finish a few cm
      // inside it. Measure the posed palm now and nudge the arm forward,
      // re-solving immediately rather than waiting for the next frame.
      if (st.torsoProfile && st.armTgt && st.armTgt.side === side) {
        const hand = st.bones[side + 'Hand'];
        for (let attempt = 0; attempt < 3; attempt += 1) {
          if (hand) hand.node.updateWorldMatrix(true, true);
          let need = handPenetration(st, side);
          // The sampled palm points can miss a fingertip. Also require the
          // hand's measured rear extent to clear the chest surface, the same
          // conservative test `__vrmBody` reports.
          if (hand) {
            const wrist = hand.node.getWorldPosition(_handV);
            // handBackExtent() reuses _handV, so snapshot the wrist first.
            const wx = wrist.x;
            const wy = wrist.y;
            const wz = wrist.z;
            const surf = torsoSurfaceAt(st.torsoProfile, wy);
            if (surf && Math.abs(wx - surf.axisX) < surf.halfX + 0.02) {
              const back = handBackExtent(st, side);
              const deficit = surf.front + HAND_CLEARANCE - (wz - back);
              if (deficit > need) need = deficit;
            }
          }
          if (need <= HAND_CLEARANCE) break;
          const t = new THREE.Vector3().fromArray(st.armTgt.t);
          t.z += need + HAND_CLEARANCE;
          applyArmAt(side, t, dt, true);
          st.viol.chest = (st.viol.chest || 0) + 1;
        }
      }
    };

    const applyFrame = (frame, dt, calib) => {
      if (!frame || !Array.isArray(frame.hands)) return;
      // Cluster the frame's detections first: the tracker reports one
      // physical hand twice often enough to matter (114 of 184 shipped
      // clips), and two copies drive two arms that then occupy the same
      // space.
      const clusters = [];
      for (const hand of frame.hands) {
        if (!hand || !Array.isArray(hand.landmarks) || hand.landmarks.length !== 21) continue;
        const w = hand.landmarks[0];
        let near = -1;
        for (let i = 0; i < clusters.length; i += 1) {
          const p = clusters[i].landmarks[0];
          if (Math.hypot(w.x - p.x, w.y - p.y) < HAND_MERGE_DIST) { near = i; break; }
        }
        if (near < 0) {
          clusters.push(hand);
        } else {
          // One hand seen twice: keep the more confident detection.
          st.viol.dup += 1;
          if ((hand.handedness_score || 0) > (clusters[near].handedness_score || 0)) {
            clusters[near] = hand;
          }
        }
      }
      // Assign clusters to arms by CONTINUITY - the arm whose previous wrist
      // is nearest keeps the hand - never by the per-frame handedness label,
      // which flips mid-clip and would swap the signing arm mid-sign. Lateral
      // position is only the fallback for an arm with no history yet.
      st.lastWrist = st.lastWrist || {};
      const wristCost = (hand, side) => {
        const prev = st.lastWrist[side];
        if (!prev) return 0.5;
        const w = hand.landmarks[0];
        return Math.hypot(w.x - prev.x, w.y - prev.y);
      };
      const assign = [];
      if (clusters.length === 0) {
        // Sparse frame: no usable detection. Hold both arms (a short dropout)
        // or ease them down (a hand that is really gone), and never fall
        // through to the pairing code.
        st.claspAdj = null;
        st.claspInfo = null;
        settleArm('left', dt);
        settleArm('right', dt);
        st.lastAssign = {
          clusters: 0,
          sides: [],
          relaxed: ['left', 'right'].filter((sd) => (st.gap[sd] || 0) > HAND_GAP_HOLD),
        };
        return;
      }
      if (clusters.length === 1) {
        const hand = clusters[0];
        const cl = wristCost(hand, 'left');
        const cr = wristCost(hand, 'right');
        let side;
        if (cl === cr) {
          // No history on either arm: image-left is the signer's right hand,
          // which is the avatar's right arm.
          side = hand.landmarks[0].x > 0.5 ? 'left' : 'right';
        } else {
          side = cl < cr ? 'left' : 'right';
          const other = side === 'left' ? 'right' : 'left';
          if (st.lastSingleSide === other
            && Math.max(cl, cr) - Math.min(cl, cr) < SIDE_SWITCH_HYST) {
            side = other;
            st.viol.sideHold += 1;
          }
        }
        st.lastSingleSide = side;
        assign.push([hand, side]);
      } else {
        // Two or more distinct hands: keep the two most confident and pick
        // the pairing (or its swap) with the lower total cost. A tie means
        // neither arm has history, and the lateral rule breaks it.
        if (clusters.length > 2) st.viol.dup += clusters.length - 2;
        const top = clusters
          .slice()
          .sort((a, b) => (b.handedness_score || 0) - (a.handedness_score || 0))
          .slice(0, 2)
          .sort((a, b) => a.landmarks[0].x - b.landmarks[0].x);
        const [leftMost, rightMost] = top;
        const straight = wristCost(leftMost, 'right') + wristCost(rightMost, 'left');
        const crossed = wristCost(leftMost, 'left') + wristCost(rightMost, 'right');
        if (straight <= crossed) {
          assign.push([leftMost, 'right'], [rightMost, 'left']);
        } else {
          assign.push([leftMost, 'left'], [rightMost, 'right']);
        }
        st.lastSingleSide = null;
      }
      // Contact assist: if the source holds the two hands close, pull the
      // pair together (toward their own midpoint, not the body) so the palms
      // meet. Applied as a per-side lateral delta used inside applyArm.
      st.claspAdj = null;
      st.claspInfo = null;
      const wrist = {};
      for (const [hand, side] of assign) wrist[side] = hand.landmarks[0];
      if (calib && wrist.left && wrist.right) {
        const wl = normalizeWrist(calib, wrist.left);
        const wr = normalizeWrist(calib, wrist.right);
        const latL = (wl.x - 0.5) * LAT_GAIN;
        const latR = (wr.x - 0.5) * LAT_GAIN;
        const gap = Math.abs(latL - latR);
        if (gap > 1e-4 && gap < CLASP_THRESHOLD) {
          const t = Math.min(
            1, (CLASP_THRESHOLD - gap) / (CLASP_THRESHOLD - CLASP_FULL));
          let scale = 1 - t * (1 - CLASP_MIN_SCALE);
          // Palms are solid: CLASP_MIN_SCALE alone could drive the pair to
          // 1.9 cm apart when the hand is ~5 cm thick, letting the two meshes
          // intersect. Clamp the compressed separation to the measured hand
          // thickness so the palms can touch but never pass through, and
          // never widen a pair that is already tighter than that.
          const minSep = st.handMinSep || 0;
          if (minSep > 0 && gap * scale < minSep) scale = Math.min(1, minSep / gap);
          st.claspInfo = {
            latL: +latL.toFixed(4), latR: +latR.toFixed(4), gap: +gap.toFixed(4),
            t: +t.toFixed(3), scale: +scale.toFixed(3), minSep: +minSep.toFixed(4),
            out: +(gap * scale).toFixed(4),
          };
          const mid = (latL + latR) / 2;
          st.claspAdj = {
            left: (mid - latL) * (1 - scale),
            right: (mid - latR) * (1 - scale),
          };
        }
      }
      for (const [hand, side] of assign) {
        const w0 = hand.landmarks[0];
        st.lastWrist[side] = { x: w0.x, y: w0.y, z: w0.z };
        try {
          applyHand(hand, side === 'left', dt, calib);
        } catch {
          /* never let one bad frame kill playback */
        }
      }
      // A side with nothing this frame holds its last pose through a tracker
      // dropout (HAND_GAP_HOLD) and only then relaxes to the arms-down idle.
      for (const side of ['left', 'right']) {
        if (wrist[side]) driveArm(side);
        else settleArm(side, dt);
      }
      st.lastAssign = {
        clusters: clusters.length,
        sides: assign.map((pair) => pair[1]),
        relaxed: ['left', 'right'].filter(
          (sd) => !wrist[sd] && (st.gap[sd] || 0) > HAND_GAP_HOLD),
      };
    };

    const tick = () => {
      if (st.disposed) return;
    rafId = requestAnimationFrame(tick);

      const dt = Math.min(clock.getDelta(), 0.1);
      st.ticks = (st.ticks || 0) + 1;
      // Single sentence dataset wins; otherwise the per-gloss playlist
      // plays clip after clip (smoothing carries the pose across cuts).
      // A test probe freezes playback so the probed bone holds still.
      let ds = null;
      let isPlaylist = false;
      if (!st.probe) {
      if (st.dataset && st.dataset.frames && st.dataset.frames.length > 0 && st.vrm) {
        ds = st.dataset;
      } else if (st.playlist && st.playlist.length > 0 && st.vrm) {
        isPlaylist = true;
        const cur = st.playlist[st.playIndex % st.playlist.length];
        if (cur && cur.frames && cur.frames.length > 0) ds = cur;
      }
      }
      if (st.resetArms && snapArmsToIdle()) st.resetArms = false;
      if (ds) {
        const fps = ds.fps > 0 ? ds.fps : 25;
        st.frameTimer += dt * PLAYBACK_RATE;
        const frameDuration = 1 / fps;
        while (st.frameTimer >= frameDuration) {
          st.frameTimer -= frameDuration;
          st.frameIndex += 1;
          if (st.frameIndex >= ds.frames.length) {
            st.frameIndex = 0;
            if (isPlaylist) st.playIndex = (st.playIndex + 1) % st.playlist.length;
            // Mid-sentence gloss switch: crossfade directly (coarticulation)
            // when the hands do not jump. Only a sentence boundary or a big
            // jump passes through neutral, so the arms cannot sweep through
            // the torso. A single clip's loop wrap always blends directly.
            const wrapped = isPlaylist && st.playIndex === 0;
            const nextDs = isPlaylist
              ? st.playlist[st.playIndex % st.playlist.length]
              : ds;
            const jump = clipSwitchJump(
              st.lastWrist, nextDs && nextDs.frames && nextDs.frames[0]);
            beginBlend(isPlaylist
              ? (wrapped || jump > SWITCH_JUMP_MAX)
              : false);
          }
        }
        // Solve the retargeter every rendered frame. The clip only has a new
        // frame at 25/30 Hz; holding the previous pose until then makes the
        // motion strobe on a 60 Hz display. Sample between the two frames so
        // the targets move continuously, and smooth at the render rate too.
        const frame = ds.frames[st.frameIndex];
        const next = st.frameIndex + 1 < ds.frames.length
          ? ds.frames[st.frameIndex + 1]
          : null;
        const alpha = frameDuration > 0 ? st.frameTimer / frameDuration : 0;
        const solveStart = import.meta.env.DEV ? performance.now() : 0;
        applyFrame(interpolateFrame(frame, next, alpha), dt, ds.calib || null);
        updateBlend(dt);
        if (import.meta.env.DEV) {
          st.solveMs = (st.solveMs || 0) * 0.9
            + (performance.now() - solveStart) * 0.1;
        }
      } else if (!st.probe && st.resetting && st.vrm) {
        // No sequence: relax back to the neutral pose (arms down, not the
        // model's T-pose rest).
        const t = 1 - Math.exp(-8 * Math.max(dt, 0));
        let maxAngle = 0;
        for (const key of Object.keys(st.bones)) {
          const entry = st.bones[key];
          if (!entry) continue;
          const neutral = neutralFor(key, entry);
          entry.node.quaternion.slerp(neutral, t);
          const a = entry.node.quaternion.angleTo(neutral);
          if (a > maxAngle) maxAngle = a;
        }
        if (maxAngle < 0.01) st.resetting = false;
      }
      // Non-manuals follow the gloss that is playing, not the whole sentence:
      // a WH sign carries its marker while it is being signed and releases
      // afterwards. Sentence-level clips keep the whole-sequence target.
      if (isPlaylist && st.playlist) {
        const cur = st.playlist[st.playIndex % st.playlist.length];
        st.exprTarget = expressionTargetsFor([cur && cur.gloss]);
      }
      try {
        if (st.vrm) st.vrm.update(dt);
      } catch {
        /* ignore spring-bone errors */
      }
      // Non-manuals: ease the face toward the sentence targets while a
      // sequence plays, release at rest. Guarded: preset support varies
      // by model, and a missing manager must never break the loop.
      // Plus a periodic blink (unstaring eyes are a top uncanniness cue).
      try {
        const mgr = st.vrm && st.vrm.expressionManager;
        if (mgr) {
          const active = !!ds;
          const k = 1 - Math.exp(-6 * Math.max(dt, 0));
          for (const name of ['surprised', 'angry']) {
            const goal = active ? (st.exprTarget[name] || 0) : 0;
            st.exprCurrent[name] += (goal - st.exprCurrent[name]) * k;
            mgr.setValue(name, Math.max(0, Math.min(1, st.exprCurrent[name])));
          }
          st.blinkT = (st.blinkT || 2.5) - dt;
          if (st.blinkT <= -0.12) st.blinkT = 2.5 + Math.random() * 3;
          const blinkW = st.blinkT <= 0 ? Math.max(0, 1 + st.blinkT / 0.12) : 0;
          try {
            mgr.setValue('blink', blinkW);
          } catch {
            /* preset missing on this model */
          }
          st.blink = blinkW;
        }
      } catch {
        /* expressions are cosmetic; ignore model gaps */
      }
      if (st.probe) {
        try {
          st.probe.entry.node.quaternion.copy(st.probe.q);
          st.probe.entry.node.updateWorldMatrix(true, false);
        } catch {
          /* ignore */
        }
      }
      try {
        renderer.render(scene, camera);
      } catch (e) {
        st.renderErr = String((e && e.message) || e);
      }
      if (import.meta.env.DEV) {
        const lw = st.bones['leftHand'];
        const rw = st.bones['rightHand'];
        // wrist orientations + one finger joint prove the hands move
        const rf = st.bones['rightIndexProximal'];
        HOOKS.__vrm = {
          loaded: !!st.vrm,
          bones: Object.keys(st.bones).length,
          frames: ds && ds.frames ? ds.frames.length : 0,
          index: st.frameIndex,
          lw: lw ? lw.node.quaternion.toArray() : null,
          rw: rw ? rw.node.quaternion.toArray() : null,
          rf: rf ? rf.node.quaternion.toArray() : null,
          ticks: st.ticks || 0,
          solveMs: st.solveMs ? +st.solveMs.toFixed(2) : 0,
          loadId: st.loadId || 0,
          blend: st.blend ? +st.blend.t.toFixed(2) : -1,
          claspAdj: st.claspAdj || null,
          claspInfo: st.claspInfo || null,
          avBasisZ: st.avBasis ? {
            left: st.avBasis.left
              ? st.avBasis.left.z.toArray().map((v) => +v.toFixed(3)) : null,
            right: st.avBasis.right
              ? st.avBasis.right.z.toArray().map((v) => +v.toFixed(3)) : null,
          } : null,
          expr: st.exprCurrent
            ? [+(st.exprCurrent.surprised || 0).toFixed(2),
               +(st.exprCurrent.angry || 0).toFixed(2)]
            : null,
          viol: st.viol || null,
          neutral: st.neutral
            ? { bones: Object.keys(st.neutral).length, err: st.neutralErr || null }
            : null,
          idle: st.idle ? {
            upper: {
              left: st.idle.left
                ? st.idle.left.upper.toArray().map((v) => +v.toFixed(4)) : null,
              right: st.idle.right
                ? st.idle.right.upper.toArray().map((v) => +v.toFixed(4)) : null,
            },
          } : null,
          assign: st.lastAssign || null,
          thumb: st.thumbOk ?? null,
          palmMirrored: st.palmMirrored || null,
          blink: st.blink ?? null,
          renderErr: st.renderErr || null,
          clips: st.playlist ? st.playlist.length : 0,
          playIndex: st.playIndex || 0,
          frame: st.frameInfo || null,
          arm: {
            left: st.arm && st.arm.left ? st.arm.left.upper.node.quaternion.toArray() : null,
            right: st.arm && st.arm.right ? st.arm.right.upper.node.quaternion.toArray() : null,
            chest: st.chestPos ? st.chestPos.toArray() : null,
            tgt: st.armTgt || null,
            shoulder: st.arm && st.arm.left ? st.arm.left.S.toArray() : null,
            lens: st.arm && st.arm.left
              ? [+st.arm.left.lenU.toFixed(4), +st.arm.left.lenL.toFixed(4)]
              : null,
          },
        };
      }
    };

    (async () => {
      try {
        const loader = new GLTFLoader();
        loader.register((parser) => new VRMLoaderPlugin(parser));
        const gltf = await loader.loadAsync(apiUrl('/avatar/model.vrm'));
        if (st.disposed || loadToken !== st.loadSeq) return;
        const vrm = gltf.userData.vrm;
        if (!vrm) throw new Error('No VRM in model file');
        vrm.scene.traverse((o) => {
          o.frustumCulled = false;
        });
        // Drop any previously attached model before attaching the new
        // one, so exactly one avatar exists in the scene.
        if (st.modelRoot && st.modelRoot.parent === scene) {
          scene.remove(st.modelRoot);
          disposeObject(st.modelRoot);
        }
        scene.add(vrm.scene);
        st.modelRoot = vrm.scene;
        st.vrm = vrm;
        // We drive RAW bone nodes directly; the normalized->raw transfer
        // would overwrite our pose on every vrm.update, so disable it.
        try {
          vrm.humanoid.autoUpdateHumanBones = false;
        } catch {
          /* ignore */
        }
        // Frame head-to-hips with margin (stored; fitViewport applies it
        // now and again on every layout change, so the framing can never
        // freeze from a pre-CSS measurement).
        try {
          const box = new THREE.Box3().setFromObject(vrm.scene);
          const center = box.getCenter(new THREE.Vector3());
          const size = box.getSize(new THREE.Vector3());
          st.frameBox = {
            centerX: center.x, centerY: center.y, centerZ: center.z, sizeY: size.y,
          };
          fitViewport();
        } catch (e) {
          st.frameInfo = { err: String((e && e.message) || e) };
          /* keep the default framing */
        }
        const n = captureRestPose(vrm);
        // Collision geometry measured from the mesh (see measureBody). Until
        // this lands, the arm clamp falls back to the old capsule.
        try {
          const body = measureBody(vrm);
          if (body) {
            st.torsoProfile = body.profile;
            st.handBox = body.handBox;
            st.penPts = body.penPts;
            let thick = 0;
            for (const side of ['left', 'right']) {
              const b = body.handBox[side];
              if (!b) continue;
              // A hand is a flat slab: its smallest half-extent is half the
              // palm thickness, which is the distance that matters when the
              // palms face each other.
              thick = Math.max(thick, 2 * Math.min(b.h.x, b.h.y, b.h.z));
            }
            st.handMinSep = thick > 0 ? thick + HAND_CLEARANCE : 0;
          }
        } catch (e) {
          st.bodyErr = String((e && e.message) || e);
        }
        // Arms-down idle, built after the body is measured so it obeys the
        // same collision clamps as the live pose.
        try {
          buildNeutralPose();
        } catch (e) {
          st.neutralErr = String((e && e.message) || e);
        }
        setModelReady(true);
        if (typeof window !== 'undefined') {
          HOOKS.__vrm = { loaded: true, bones: n, frames: 0, index: 0, lw: null, rw: null };
        }
      } catch (e) {
        fail(`Avatar failed to load: ${e && e.message ? e.message : e}`);
      }
    })();

    rafId = requestAnimationFrame(tick);

    // Test probe: pose one bone directly (playback freezes while a
    // probe is set, so the probed bone holds still for inspection).
    if (typeof window !== 'undefined') {
      HOOKS.__vrmProbe = (name, x, y, z, w) => {
        const entry = st.bones[name];
        if (!entry) return `no-bone:${name}:have=${Object.keys(st.bones).length}`;
        entry.node.quaternion.set(x, y, z, w);
        entry.node.updateWorldMatrix(true, false);
        st.probe = { entry, q: new THREE.Quaternion(x, y, z, w) };
        return `ok:${name}:now=${entry.node.quaternion.toArray().map((v) => Number(v).toFixed(4)).join(',')}`;
      };
      HOOKS.__vrmUnprobe = () => {
        st.probe = null;
        // Re-arm the ease-back so the avatar relaxes to rest instead of
        // freezing in the probed pose.
        st.resetting = true;
        return 'ok';
      };
      // Scene probes: verify screenshots reflect live rendering.
      HOOKS.__vrmBg = (hex) => {
        st.scene.background.set(hex);
        return 'ok';
      };
      HOOKS.__vrmHide = () => {
        if (st.vrm) st.vrm.scene.visible = false;
        return 'ok';
      };
      HOOKS.__vrmShow = () => {
        if (st.vrm) st.vrm.scene.visible = true;
        return 'ok';
      };
      // Deterministic stepper. An occluded tab has requestAnimationFrame
      // suspended, which freezes playback and makes playback impossible to
      // verify headlessly. Drive the frame loop by hand with a synthetic
      // clock instead: same code path, fixed dt. Debug/verification only.
      const liveClip = () => {
        if (st.dataset && st.dataset.frames && st.dataset.frames.length) return st.dataset;
        if (st.playlist && st.playlist.length) {
          const cur = st.playlist[st.playIndex % st.playlist.length];
          if (cur && cur.frames && cur.frames.length) return cur;
        }
        return null;
      };
      HOOKS.__vrmAdvance = (steps = 20, dtMs = 33) => {
        const realNow = performance.now.bind(performance);
        let t = realNow();
        try {
          performance.now = () => (t += dtMs);
        } catch {
          /* ignore */
        }
        for (let i = 0; i < Math.max(1, steps); i++) tick();
        try {
          performance.now = realNow;
          clock.oldTime = realNow();
        } catch {
          /* ignore */
        }
        return {
          ticks: st.ticks || 0,
          index: st.frameIndex,
          playing: !!liveClip(),
          avatar: !!st.vrm,
        };
      };
      // One-shot: advance to a target clip frame, then grab the canvas.
      // The PNG is parked on `HOOKS.__vrmShot` (download it separately so
      // the return value stays small).
      HOOKS.__vrmCaptureAt = (target = 22, maxSteps = 600) => {
        let guard = 0;
        while (guard < maxSteps) {
          if (!liveClip()) break;
          if (st.frameIndex >= target) break;
          HOOKS.__vrmAdvance(1, 33);
          guard += 1;
        }
        const c = st.renderer && st.renderer.domElement;
        HOOKS.__vrmShot = c ? c.toDataURL('image/png') : null;
        HOOKS.__vrmShotIdx = st.frameIndex;
        return {
          index: st.frameIndex,
          guard,
          palm: typeof HOOKS.__vrmPalm === 'function' ? HOOKS.__vrmPalm() : null,
          shot: !!HOOKS.__vrmShot,
        };
      };
      HOOKS.__vrmDownloadShot = (name) => {
        if (!HOOKS.__vrmShot) return 'no-shot';
        const a = document.createElement('a');
        a.href = HOOKS.__vrmShot;
        a.download = name || ('shot_' + HOOKS.__vrmShotIdx + '.png');
        document.body.appendChild(a);
        a.click();
        a.remove();
        return 'ok:' + a.download;
      };
      // Load one landmark clip directly, for visual checks and for replaying a
      // single file without going through the translation pipeline. Dev-only.
      HOOKS.__vrmPlay = async (url) => {
        const res = await fetch(url, { cache: 'reload' });
        const data = await res.json();
        data.calib = calibrateSigningSpace(data);
        st.dataset = data;
        st.playlist = null;
        st.blend = null;
        st.playIndex = 0;
        st.frameIndex = 0;
        st.frameTimer = 0;
        st.lastWrist = {};
        st.lastWristQ = {};
        st.lastSingleSide = null;
        st.gap = { left: HAND_GAP_HOLD + 1, right: HAND_GAP_HOLD + 1 };
        st.resetArms = true;
        return { clip: String(url).split('/').pop(), frames: (data.frames || []).length, fps: data.fps };
      };
      // Palm-orientation probe: does the wrist actually reach its target?
      HOOKS.__vrmPalm = () => {
        const out = {};
        for (const side of ['left', 'right']) {
          const hand = st.bones[side + 'Hand'];
          const index = st.bones[side + 'IndexProximal'];
          const little = st.bones[side + 'LittleProximal'];
          if (!hand || !index || !little) { out[side] = 'missing'; continue; }
          const o = new THREE.Vector3();
          const pi = new THREE.Vector3();
          const pl = new THREE.Vector3();
          hand.node.getWorldPosition(o);
          index.node.getWorldPosition(pi);
          little.node.getWorldPosition(pl);
          const x = pi.clone().sub(o);
          const m = pl.clone().sub(pi);
          let z = null;
          if (x.lengthSq() > 1e-8 && m.lengthSq() > 1e-8) {
            z = new THREE.Vector3().crossVectors(x, m).normalize();
          }
          const rec = st.lastLmB && st.lastLmB[side];
          const ii = st.bones[side + 'IndexIntermediate'];
          const fseg = ii ? ii.node.getWorldPosition(new THREE.Vector3()).sub(pi) : null;
          out[side] = {
            pos: o.toArray().map((v) => +v.toFixed(3)),
            normal: z ? z.toArray().map((v) => +v.toFixed(2)) : null,
            finger: x.clone().normalize().toArray().map((v) => +v.toFixed(2)),
            fingerT: rec ? rec.b.x.toArray().map((v) => +v.toFixed(2)) : null,
            // How far the first finger joint is bent away from the hand's
            // finger axis. Source data says ~15-25 deg for this clip; a
            // near-90 deg reading means the finger bones are being driven
            // somewhere the landmarks never asked for.
            bend: (fseg && x.lengthSq() > 1e-8)
              ? +THREE.MathUtils.radToDeg(fseg.clone().normalize()
                  .angleTo(x.clone().normalize())).toFixed(1) : null,
            errDeg: (z && rec) ? +THREE.MathUtils.radToDeg(z.angleTo(rec.b.z)).toFixed(1) : null,
            fErr: (rec && x.lengthSq() > 1e-8)
              ? +THREE.MathUtils.radToDeg(x.clone().normalize().angleTo(rec.b.x)).toFixed(1) : null,
            cap: (st.wristCap && st.wristCap[side]) || 0,
            clasp: !!st.claspAdj,
          };
        }
        return JSON.stringify(out);
      };
      // Chest-penetration probe: the whole point of the torso-profile guard is
      // that no part of a hand sits inside the torso, so report the measured
      // surface, how far back the hand box reaches, and the resulting
      // intrusion (negative = clear of the body).
      HOOKS.__vrmBody = () => {
        const prof = st.torsoProfile;
        const out = {};
        for (const side of ['left', 'right']) {
          const hand = st.bones[side + 'Hand'];
          if (!hand) { out[side] = 'missing'; continue; }
          const o = hand.node.getWorldPosition(new THREE.Vector3());
          const back = handBackExtent(st, side);
          const surf = prof ? torsoSurfaceAt(prof, o.y) : null;
          const rear = o.z - back;
          const overTorso = surf
            ? Math.abs(o.x - surf.axisX) < surf.halfX + 0.02 : false;
          const joint = st.arm && st.arm[side];
          out[side] = {
            wrist: o.toArray().map((v) => +v.toFixed(3)),
            elbow: joint ? joint.lower.node.getWorldPosition(new THREE.Vector3())
              .toArray().map((v) => +v.toFixed(3)) : null,
            shoulder: joint ? joint.S.toArray().map((v) => +v.toFixed(3)) : null,
            back: +back.toFixed(4),
            overTorso,
            chestFrontZ: surf ? +surf.front.toFixed(3) : null,
            torsoHalfX: surf ? +surf.halfX.toFixed(3) : null,
            handRearZ: +rear.toFixed(3),
            // > 0 means the back of the hand is behind the chest surface,
            // i.e. the torso mesh is drawing over the hand. Negative = clear.
            intrusion: (surf && overTorso) ? +(surf.front - rear).toFixed(4) : null,
            minSep: st.handMinSep ?? null,
          };
        }
        const lh = st.bones.leftHand && st.bones.leftHand.node;
        const rh = st.bones.rightHand && st.bones.rightHand.node;
        if (lh && rh) {
          const a = lh.getWorldPosition(new THREE.Vector3());
          const b = rh.getWorldPosition(new THREE.Vector3());
          out.handGapX = +Math.abs(a.x - b.x).toFixed(4);
        }
        out.profile = prof ? {
          hipsY: +prof.hipsY.toFixed(3),
          axisLen: +prof.axisLen.toFixed(3),
          frontZ: Array.from(prof.frontZ).map((v) => +v.toFixed(3)),
          halfX: Array.from(prof.halfX).map((v) => +v.toFixed(3)),
        } : null;
        out.err = st.bodyErr || null;
        return JSON.stringify(out);
      };
      // Ground truth for "the hand sinks into the chest": transform the
      // actual palm vertices (rigidly bound to the hand bone) into world
      // space and test each one against the measured torso surface. `inside`
      // is how many palm vertices the torso mesh is drawn over; `deepest` is
      // the worst intrusion in metres.
      HOOKS.__vrmPen = () => {
        const prof = st.torsoProfile;
        if (!st.penPts || !prof) return JSON.stringify({ err: 'no-profile' });
        const out = {};
        const v = new THREE.Vector3();
        for (const side of ['left', 'right']) {
          const pts = st.penPts[side];
          if (!pts || !pts.length) { out[side] = 'n/a'; continue; }
          const n = prof.frontZ.length;
          let inside = 0;
          let deepest = 0;
          let badPt = null;
          for (let i = 0; i < pts.length; i++) {
            const p = pts[i];
            v.set(p.x, p.y, p.z).applyMatrix4(p.b.matrixWorld);
            const t = (v.y - prof.hipsY) / prof.axisLen;
            if (t < 0 || t > 1) continue;
            const f = t * (n - 1);
            const bi = Math.min(n - 2, Math.max(0, Math.floor(f)));
            const bk = f - bi;
            const hx = prof.halfX[bi] * (1 - bk) + prof.halfX[bi + 1] * bk;
            const axisX = prof.hipsX + t * (prof.neckX - prof.hipsX);
            if (Math.abs(v.x - axisX) > hx) continue;
            const fz = prof.frontZ[bi] * (1 - bk) + prof.frontZ[bi + 1] * bk;
            if (v.z < fz) {
              inside += 1;
              const d = fz - v.z;
              if (d > deepest) {
                deepest = d;
                badPt = [+v.x.toFixed(3), +v.y.toFixed(3), +v.z.toFixed(3)];
              }
            }
          }
          out[side] = { pts: pts.length, inside, deepest: +deepest.toFixed(4), badPt };
        }
        return JSON.stringify(out);
      };
      // Binding audit: is the bone we animate the bone the mesh uses?
      HOOKS.__vrmBoneInfo = (name) => {
        if (!st.vrm) return 'no-vrm';
        let raw = null;
        try {
          raw = st.vrm.humanoid.getRawBoneNode(name);
        } catch (e) {
          return `raw-err:${e && e.message}`;
        }
        const entry = st.bones[name];
        const norm = entry ? entry.node : null;
        const skinned = [];
        st.vrm.scene.traverse((o) => {
          if (o.isSkinnedMesh) skinned.push(o);
        });
        const uses = skinned.map((m) => ({
          mesh: m.name,
          bones: m.skeleton ? m.skeleton.bones.length : -1,
          hasRaw: !!(m.skeleton && raw && m.skeleton.bones.includes(raw)),
        }));
        let rest = null;
        let worldDir = null;
        try {
          if (entry) {
            rest = entry.restLocal.toArray().map((v) => Number(v).toFixed(4));
            const p0 = new THREE.Vector3();
            entry.node.getWorldPosition(p0);
            const c = entry.node.children && entry.node.children[0];
            if (c) {
              const p1 = new THREE.Vector3();
              c.getWorldPosition(p1);
              worldDir = p1.sub(p0).normalize().toArray().map((v) => Number(v).toFixed(3));
            }
          }
        } catch {
          /* ignore */
        }
        return JSON.stringify({
          rawIsNorm: raw === norm,
          autoUpdate: st.vrm.humanoid.autoUpdateHumanBones,
          rest,
          worldDir,
          skinnedMeshes: uses.length,
          withRaw: uses.filter((u) => u.hasRaw).map((u) => u.mesh),
          uses,
        });
      };
      // Batch playback audit: replay clips through the REAL retargeter and
      // report the invariants the assignment/idle fixes must hold, so a whole
      // corpus can be checked without watching it. Returns one row per clip:
      //   dup        duplicate detections the clustering had to merge
      //   sideHold   frames where hysteresis kept the arm on the hand
      //   sides      which arm(s) the clip drove (single-hand clips: exactly 1)
      //   singleFrames / frames
      //   minGap     closest the two hands came (m); compare with handMinSep
      //   relaxedDevDeg  worst upper-arm angle from the idle while relaxed
      //   travel     wrist travel (m) - near zero means a static clip
      HOOKS.__vrmAudit = async (urls) => {
        const rows = [];
        const keep = st.dataset;
        const dt = 1 / 25;
        for (const url of urls) {
          let data;
          try {
            const res = await fetch(url, { cache: 'reload' });
            data = await res.json();
          } catch (e) {
            rows.push({ clip: String(url).split('/').pop(), error: String((e && e.message) || e) });
            continue;
          }
          const frames = data.frames || [];
          const calib = calibrateSigningSpace(data);
          st.dataset = data;
          st.playlist = null;
          st.blend = null;
          st.lastWrist = {};
          st.lastSingleSide = null;
          st.gap = { left: 0, right: 0 };
          const dup0 = st.viol.dup;
          const hold0 = st.viol.sideHold;
          let minGap = Infinity;
          let relaxedDev = 0;
          let travel = 0;
          let pairs = 0;
          let multiFrames = 0;
          let singleFrames = 0;
          let swaps = 0;
          let relaxEvents = 0;
          let holdFrames = 0;
          let maxPen = 0;
          let penFrame = null;
          const wasRelaxed = { left: false, right: false };
          let prevSide = null;
          const relaxRun = { left: 0, right: 0 };
          const sidesSeen = new Set();
          const prevPos = {};
          for (const frame of frames) {
            applyFrame(frame, dt, calib);
            if (st.modelRoot) st.modelRoot.updateMatrixWorld(true);
            const pen = Math.max(
              handPenetration(st, 'left'),
              handPenetration(st, 'right'),
            );
            if (pen > maxPen) { maxPen = pen; penFrame = st.frameIndex; }
            const a = st.lastAssign || { sides: [], relaxed: [] };
            if ((a.clusters | 0) >= 2) multiFrames += 1;
            if (a.sides && a.sides.length === 1) {
              singleFrames += 1;
              sidesSeen.add(a.sides[0]);
              if (prevSide && prevSide !== a.sides[0]) swaps += 1;
              prevSide = a.sides[0];
            }
            const pos = {};
            for (const side of ['left', 'right']) {
              const e = st.bones[side + 'Hand'];
              if (!e) continue;
              const m = e.node.matrixWorld.elements;
              pos[side] = { x: m[12], y: m[13], z: m[14] };
            }
            if (a.sides && a.sides.length === 2 && pos.left && pos.right) {
              // Only compare two hands that are BOTH signing this frame: the
              // signing hand passing the resting hand is not a collision.
              pairs += 1;
              const g = Math.hypot(
                pos.left.x - pos.right.x,
                pos.left.y - pos.right.y,
                pos.left.z - pos.right.z,
              );
              if (g < minGap) minGap = g;
            }
            for (const side of ['left', 'right']) {
              const relaxed = (a.relaxed || []).includes(side);
              if (relaxed && !wasRelaxed[side]) relaxEvents += 1;
              wasRelaxed[side] = relaxed;
              // Absent but still inside HAND_GAP_HOLD: the pose is held.
              if (!relaxed && !(a.sides || []).includes(side)) holdFrames += 1;
              if (!relaxed) {
                relaxRun[side] = 0;
                continue;
              }
              relaxRun[side] += 1;
              // Only judge the ease once it has had time to converge; the
              // first frames after a hand leaves are expected to be far.
              if (relaxRun[side] < 5) continue;
              const A = st.arm && st.arm[side];
              const idle = st.idle && st.idle[side];
              if (!A || !idle) continue;
              const deg = A.upper.node.quaternion.angleTo(idle.upper) * (180 / Math.PI);
              if (deg > relaxedDev) relaxedDev = deg;
            }
            for (const side of a.sides || []) {
              const p = pos[side];
              if (!p) continue;
              if (prevPos[side]) {
                travel += Math.hypot(
                  p.x - prevPos[side].x,
                  p.y - prevPos[side].y,
                  p.z - prevPos[side].z,
                );
              }
              prevPos[side] = p;
            }
          }
          rows.push({
            clip: String(url).split('/').pop(),
            frames: frames.length,
            singleFrames,
            multiFrames,
            sides: Array.from(sidesSeen).join('+'),
            swaps,
            relaxEvents,
            holdFrames,
            pairs,
            minGap: Number.isFinite(minGap) ? +minGap.toFixed(4) : null,
            relaxedDevDeg: +relaxedDev.toFixed(1),
            travel: +travel.toFixed(3),
            maxPen: +maxPen.toFixed(4),
            penFrame,
            dup: st.viol.dup - dup0,
            sideHold: st.viol.sideHold - hold0,
          });
        }
        st.dataset = keep;
        st.blend = null;
        // Accumulate across batches, then report only what violates a gate.
        if (rows.length) HOOKS.__vrmAuditRows = (HOOKS.__vrmAuditRows || []).concat(rows);
        const all = HOOKS.__vrmAuditRows || [];
        const problems = all.filter((r) => {
          if (r.error) return true;
          // A hand may only hop arms if the clip really uses two hands; in a
          // single-handed clip the same hand must keep the same arm.
          const singleShare = 1 - r.multiFrames / Math.max(1, r.frames);
          if (singleShare >= 0.9 && r.swaps > 0) return true;
          if (r.relaxedDevDeg > 15) return true;
          if (r.pairs >= 5 && r.minGap != null && r.minGap < 0.045) return true;
          // Dropping an arm to rest mid-sign more than the one resting hand
          // allows means a gap outlasted HAND_GAP_HOLD.
          if (r.relaxEvents > 2) return true;
          return false;
        });
        const travel = all.map((r) => r.travel).sort((x, y) => x - y);
        return {
          batch: rows.length,
          total: all.length,
          problems: problems.map((r) => ({
            clip: r.clip, sides: r.sides, swaps: r.swaps, dup: r.dup,
            minGap: r.minGap, dev: r.relaxedDevDeg, travel: r.travel,
            relax: r.relaxEvents,
            singleShare: +(1 - r.multiFrames / Math.max(1, r.frames)).toFixed(2),
          })),
          twoHandedClips: all.filter((r) => r.multiFrames >= r.frames * 0.1).length,
          staticClips: all.filter((r) => r.travel < 0.05).length,
          maxRelaxedDev: all.reduce((m, r) => Math.max(m, r.relaxedDevDeg), 0),
          relaxEvents: all.reduce((sum, r) => sum + r.relaxEvents, 0),
          holdFrames: all.reduce((sum, r) => sum + r.holdFrames, 0),
          travelP05: travel.length ? travel[Math.floor(travel.length * 0.05)] : null,
          travelMedian: travel.length ? travel[Math.floor(travel.length / 2)] : null,
        };
      };
    }

    const onResize = () => {
      fitViewport();
    };
    window.addEventListener('resize', onResize);

    return () => {
      st.disposed = true;
      st.loadSeq += 1;
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', onResize);
      if (ro) ro.disconnect();
      disposeObject(scene);
      try {
        renderer.dispose();
      } catch {
        /* ignore */
      }
      try {
        if (renderer.domElement && renderer.domElement.parentNode === mount) {
          mount.removeChild(renderer.domElement);
        }
      } catch {
        /* ignore */
      }
      st.vrm = null;
      st.bones = {};
      st.dataset = null;
    };
  }, []);

  // ---- landmark dataset loading ----
  // Single sentence file, or a playlist of per-gloss landmark clips
  // played in sequence (novel sentences with no sentence file).
  useEffect(() => {
    const st = stateRef.current;
    const id = ++st.loadId;
    const url = landmarkUrl;
    const urls = listKey ? listKey.split('|') : [];
    const clearPlayback = () => {
      st.dataset = null;
      st.playlist = null;
      st.playIndex = 0;
      st.frameIndex = 0;
      st.frameTimer = 0;
      st.blend = null;
      st.resetting = true;
    };
    if (!url && urls.length === 0) {
      clearPlayback();
      return undefined;
    }
    const controller = new AbortController();
    const loadJson = (u) => fetch(resolveUrl(u), { signal: controller.signal }).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${u}`);
      return res.json();
    });
    if (url) {
      loadJson(url)
        .then((data) => {
          if (id !== st.loadId) return;
          if (!Array.isArray(data.frames) || data.frames.length === 0) {
            throw new Error('landmark file contains no frames');
          }
          st.dataset = data;
          st.playlist = null;
          st.playIndex = 0;
          st.frameIndex = 0;
          st.frameTimer = 0;
          st.resetting = false;
          st.resetArms = true;
          st.lastWrist = {};
          st.lastWristQ = {};
          st.lastSingleSide = null;
          st.gap = { left: HAND_GAP_HOLD + 1, right: HAND_GAP_HOLD + 1 };
          try {
            data.calib = calibrateSigningSpace(data);
          } catch {
            data.calib = null;
          }
          setLandmarkData({ url, frames: data.frames.length, clips: 0 });
        })
        .catch((err) => {
          if (err && err.name === 'AbortError') return;
          if (id !== st.loadId) return;
          clearPlayback();
        });
    } else {
      Promise.all(urls.map((u) => loadJson(u)))
        .then((all) => {
          if (id !== st.loadId) return;
          const valid = all.filter((d) => Array.isArray(d.frames) && d.frames.length > 0);
          if (valid.length === 0) throw new Error('no playable clips');
          st.dataset = null;
          st.playlist = valid;
          st.playIndex = 0;
          st.frameIndex = 0;
          st.frameTimer = 0;
          st.resetting = false;
          st.resetArms = true;
          st.lastWrist = {};
          st.lastWristQ = {};
          st.lastSingleSide = null;
          st.gap = { left: HAND_GAP_HOLD + 1, right: HAND_GAP_HOLD + 1 };
          for (const clip of valid) {
            try {
              clip.calib = calibrateSigningSpace(clip);
            } catch {
              clip.calib = null;
            }
          }
          const total = valid.reduce((n, d) => n + d.frames.length, 0);
          setLandmarkData({ url: listKey, frames: total, clips: valid.length });
        })
        .catch((err) => {
          if (err && err.name === 'AbortError') return;
          if (id !== st.loadId) return;
          clearPlayback();
        });
    }
    return () => controller.abort();
  }, [landmarkUrl, listKey]);

  return (
    <div className="flex w-full flex-col items-center gap-2">
      <div
        ref={mountRef}
        className="border border-white/10 rounded-lg bg-slate-900 w-full max-w-[720px] h-[480px] overflow-hidden"
      />
      <span className="text-xs text-slate-400">{status}</span>
    </div>
  );
}

// WebGL playback is built once in a mount effect, so React Fast Refresh cannot
// swap the animation loop in place. A hot update can leave the old loop running
// against a half-updated module graph - a missing constant turns the frame
// timer into NaN and the avatar freezes. Remounting this component is cheap, so
// take a full page reload instead. Dev only; production has no import.meta.hot.
if (import.meta.hot) {
  import.meta.hot.accept(() => window.location.reload());
}
