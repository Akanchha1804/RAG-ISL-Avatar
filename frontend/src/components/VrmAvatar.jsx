import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';
import { apiUrl } from '../config';

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

function basisFromPalm(p0, p5, p9) {
  const x = new THREE.Vector3().subVectors(p5, p0);
  const m = new THREE.Vector3().subVectors(p9, p0);
  if (x.lengthSq() < 1e-10 || m.lengthSq() < 1e-10) return null;
  x.normalize();
  const z = new THREE.Vector3().crossVectors(x, m);
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

// Per-clip signing-space calibration (M6 fix): source videos frame the
// signer differently (waist-level vs chest-level), so absolute image coords
// reproduce the wrong height on the avatar. Each clip's wrist trajectory
// bbox is mapped into the signing box instead. Min-span guarded: a static
// hold maps mid-box rather than amplifying jitter into full-box swings.
const CALIB_MIN_SPAN_XY = 0.12;
const CALIB_MIN_SPAN_Z = 0.05;

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
  const minX = pct(xs, 0.02);
  const maxX = pct(xs, 0.98);
  const minY = pct(ys, 0.02);
  const maxY = pct(ys, 0.98);
  const minZ = pct(zs, 0.02);
  const maxZ = pct(zs, 0.98);
  const fit = (lo, hi, minSpan) => {
    const span = Math.max(hi - lo, minSpan);
    const mid = (lo + hi) / 2;
    return [mid - span / 2, span];
  };
  const [x0, spanX] = fit(minX, maxX, CALIB_MIN_SPAN_XY);
  const [y0, spanY] = fit(minY, maxY, CALIB_MIN_SPAN_XY);
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

// ISL non-manual markers driven from the gloss sequence (M6 fix): faces
// carry question/negation grammar that hands alone cannot express.
// WH-words -> brow raise (VRM 'surprised'); negation -> brow furrow
// (VRM 'angry' at a low, non-emotional weight). Eased in while signing,
// released at rest. Display-only grammar channel; never affects glosses.
const WH_GLOSSES = new Set(['WHAT', 'WHEN', 'WHERE', 'WHO', 'WHOM', 'WHOSE', 'WHICH', 'WHY', 'HOW']);
const NEG_GLOSSES = new Set(['NOT', 'NO', 'NEVER', 'DONOT', 'DONT', 'CANNOT']);

function expressionTargetsFor(glossSequence) {
  const toks = Array.isArray(glossSequence)
    ? glossSequence.map((g) => String(g).toUpperCase())
    : [];
  return {
    surprised: toks.some((t) => WH_GLOSSES.has(t)) ? 0.55 : 0,
    angry: toks.some((t) => NEG_GLOSSES.has(t)) ? 0.35 : 0,
  };
}

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
      : (signing ? `Signing ${landmarkData.frames} frames`
        : (playlistActive ? `Signing ${landmarkData.clips} clips`
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
    viol: { curl: 0, elbow: 0, torso: 0, skipped: 0, dup: 0, wcap: 0 },
  });

  // Non-manual targets follow the gloss sequence (derived, never set in
  // the render loop). Keyed on the joined string so the effect only runs
  // when the sequence actually changes.
  const glossKey = Array.isArray(glossSequence) ? glossSequence.join('|') : '';
  useEffect(() => {
    stateRef.current.exprTarget = expressionTargetsFor(
      glossKey ? glossKey.split('|') : []);
  }, [glossKey]);

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
    scene.background = new THREE.Color('#0f172a');
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
          bones[name] = { node, restLocal, restLocalDir };
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
        const middle = bones[side + 'MiddleProximal'];
        if (!hand || !index || !middle) continue;
        const o = new THREE.Vector3();
        const pi = new THREE.Vector3();
        const pm = new THREE.Vector3();
        hand.node.getWorldPosition(o);
        index.node.getWorldPosition(pi);
        middle.node.getWorldPosition(pm);
        const toIndex = pi.sub(o);
        const toMiddle = pm.sub(o.clone());
        if (toIndex.lengthSq() < 1e-8 || toMiddle.lengthSq() < 1e-8) continue;
        const x = toIndex.clone().normalize();
        const z = new THREE.Vector3().crossVectors(toIndex, toMiddle);
        if (z.lengthSq() < 1e-8) continue;
        z.normalize();
        const y = new THREE.Vector3().crossVectors(z, x).normalize();
        x.crossVectors(y, z).normalize();
        const outward = o.clone().sub(midRef);
        outward.y = 0;
        if (outward.lengthSq() < 1e-8) outward.set(side === 'left' ? 1 : -1, 0, 0);
        outward.normalize();
        if (z.dot(outward) < 0) {
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
      // Clip switch: first half -> rest, second half rest -> live.
      const half = x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2;
      for (const key of Object.keys(b.pose)) {
        const entry = st.bones[key];
        if (!entry || !entry.node || !entry.restLocal) continue;
        if (x >= 1) continue;
        if (x < 0.5) {
          entry.node.quaternion.slerpQuaternions(
            b.pose[key], entry.restLocal, half);
        } else {
          entry.node.quaternion.slerpQuaternions(
            entry.restLocal, entry.node.quaternion.clone(), half);
        }
      }
      if (x >= 1) st.blend = null;
    };

    const _flexQ = new THREE.Quaternion();
    const applyFingerBone = (entry, segDirWorld, lmB, avB, dt, maxFlex, damp = 1) => {
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
    const applyArm = (side, wristRaw, dt, calib) => {
      const A = st.arm && st.arm[side];
      const C = st.chestPos;
      if (!A || !C) return;
      const w = normalizeWrist(calib, wristRaw);
      // Lateral gain is narrower than vertical: the avatar's arms are
      // short relative to a full-box sweep, which read as stiff full
      // wingspans. ±0.275m lateral keeps elbows bent on wide signs.
      const target = new THREE.Vector3(
        C.x + (w.x - 0.5) * 0.55,
        C.y + 0.02 + (0.5 - w.y) * 0.7,
        Math.max(C.z + 0.3 - w.z * 0.4, C.z + 0.06),
      );
      // Torso collision: the signing box overlaps the chest volume, so
      // clamp the wrist target outside the torso capsule (+3cm standoff).
      if (st.torsoHips && st.torsoNeck) {
        if (clampOutsideCapsule(target, st.torsoHips, st.torsoNeck,
          st.torsoRadius + 0.03, target)) st.viol.torso += 1;
      }
      const toT = target.clone().sub(A.S);
      let dist = toT.length();
      const maxReach = A.lenU + A.lenL - 0.02;
      if (dist > maxReach) {
        dist = maxReach;
        target.copy(A.S).addScaledVector(toT.normalize(), dist);
      }
      if (dist < 1e-6) return;
      st.armTgt = { side, t: target.toArray() };
      const dirST = toT.normalize();
      const cosA = Math.min(1, Math.max(-1,
        (A.lenU * A.lenU + dist * dist - A.lenL * A.lenL) / (2 * A.lenU * dist)));
      const angA = Math.acos(cosA);
      // Adaptive pole: bias downward as the target crosses the midline so
      // the elbow folds naturally instead of flaring or flipping.
      const sideSign = side === 'left' ? 1 : -1;
      const cross = Math.min(1, Math.max(0, -sideSign * (target.x - C.x) / 0.3));
      const pole = A.pole.clone()
        .addScaledVector(_DOWN, cross * 0.8).normalize();
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
      smoothWithDt(A.upper.node, parentQU.invert().multiply(qU), dt);

      A.lower.node.parent.updateWorldMatrix(true, false);
      const parentQL = new THREE.Quaternion();
      A.lower.node.parent.getWorldQuaternion(parentQL);
      const qL = new THREE.Quaternion()
        .setFromUnitVectors(A.restDirL, foreDir).multiply(A.restWQL);
      smoothWithDt(A.lower.node, parentQL.invert().multiply(qL), dt);
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
      const p9 = P(9);
      let lmB = basisFromPalm(p0, p5, p9);
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
      const qAv = quatFromBasis(avB);
      const qLm = quatFromBasis(lmB);
      const palmDelta = qLm.multiply(qAv.invert());

      const handEntry = st.bones[side + 'Hand'];
      if (handEntry) {
        const target = palmDelta.multiply(handEntry.restLocal.clone());
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
            goal = prevQ.clone().rotateTowards(target, maxStep);
          }
        }
        st.lastWristQ[side] = goal.clone();
        smoothWithDt(handEntry.node, goal, dt);
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
            FLEX_LIMITS[joints[j]], joints[j] === 'Distal' ? 0.45 : 1);
        }
      }
      for (let j = 0; j < 3; j++) {
        const entry = st.bones[side + 'Thumb' + THUMB_BONES[j]];
        const [a, b] = THUMB_SEGS[j];
        // Thumb saddle joint over-rotates: damp the metacarpal, soften tips.
        const damp = j === 0 ? 0.6 : (j === 2 ? 0.45 : 0.8);
        applyFingerBone(entry, seg(a, b), lmB, avB, dt, FLEX_LIMITS.ThumbDefault, damp);
      }
    };

    const applyFrame = (frame, dt, calib) => {
      if (!frame || !Array.isArray(frame.hands)) return;
      // At most ONE hand per side per frame: tracker mislabels put two
      // same-side hands in a frame (190/288 clips affected), and driving
      // both double-drives one arm (last-wins flicker + starved side).
      // Keep the wrist nearest that side's previous wrist; count the rest.
      st.lastWrist = st.lastWrist || {};
      const bySide = { left: [], right: [] };
      for (const hand of frame.hands) {
        if (!hand || !Array.isArray(hand.landmarks) || hand.landmarks.length !== 21) continue;
        const h = String(hand.handedness || '').toLowerCase();
        if (!h.includes('left') && !h.includes('right')) continue;
        (h.includes('left') ? bySide.left : bySide.right).push(hand);
      }
      for (const side of ['left', 'right']) {
        const cands = bySide[side];
        if (cands.length === 0) continue;
        let pick = cands[0];
        if (cands.length > 1) {
          st.viol.dup = (st.viol.dup || 0) + (cands.length - 1);
          const prev = st.lastWrist[side];
          if (prev) {
            let best = Infinity;
            for (const c of cands) {
              const w = c.landmarks[0];
              const d = (w.x - prev.x) ** 2 + (w.y - prev.y) ** 2 + (w.z - prev.z) ** 2;
              if (d < best) { best = d; pick = c; }
            }
          }
        }
        const w0 = pick.landmarks[0];
        st.lastWrist[side] = { x: w0.x, y: w0.y, z: w0.z };
        try {
          applyHand(pick, side === 'left', dt, calib);
        } catch {
          /* never let one bad frame kill playback */
        }
      }
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
      if (ds) {
        const fps = ds.fps > 0 ? ds.fps : 25;
        st.frameTimer += dt;
        const frameDuration = 1 / fps;
        let advanced = false;
        while (st.frameTimer >= frameDuration) {
          st.frameTimer -= frameDuration;
          st.frameIndex += 1;
          if (st.frameIndex >= ds.frames.length) {
            st.frameIndex = 0;
            if (isPlaylist) st.playIndex = (st.playIndex + 1) % st.playlist.length;
            // Clip SWITCH passes through neutral (signer resetting between
            // signs). Loop WRAP of one clip blends directly end->start: a
            // neutral dip every loop reads as stuttering mid-performance.
            beginBlend(isPlaylist);
          }
          advanced = true;
        }
        if (advanced || st.frameTimer === dt) {
          applyFrame(ds.frames[st.frameIndex], dt, ds.calib || null);
        }
        updateBlend(dt);
      } else if (!st.probe && st.resetting && st.vrm) {
        // No sequence: relax back to the rest pose.
        const t = 1 - Math.exp(-8 * Math.max(dt, 0));
        let maxAngle = 0;
        for (const key of Object.keys(st.bones)) {
          const entry = st.bones[key];
          if (!entry) continue;
          entry.node.quaternion.slerp(entry.restLocal, t);
          const a = entry.node.quaternion.angleTo(entry.restLocal);
          if (a > maxAngle) maxAngle = a;
        }
        if (maxAngle < 0.01) st.resetting = false;
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
      if (typeof window !== 'undefined') {
        const lw = st.bones['leftHand'];
        const rw = st.bones['rightHand'];
        // wrist orientations + one finger joint prove the hands move
        const rf = st.bones['rightIndexProximal'];
        window.__vrm = {
          loaded: !!st.vrm,
          bones: Object.keys(st.bones).length,
          frames: ds && ds.frames ? ds.frames.length : 0,
          index: st.frameIndex,
          lw: lw ? lw.node.quaternion.toArray() : null,
          rw: rw ? rw.node.quaternion.toArray() : null,
          rf: rf ? rf.node.quaternion.toArray() : null,
          ticks: st.ticks || 0,
          loadId: st.loadId || 0,
          blend: st.blend ? +st.blend.t.toFixed(2) : -1,
          expr: st.exprCurrent
            ? [+(st.exprCurrent.surprised || 0).toFixed(2),
               +(st.exprCurrent.angry || 0).toFixed(2)]
            : null,
          viol: st.viol || null,
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
        setModelReady(true);
        if (typeof window !== 'undefined') {
          window.__vrm = { loaded: true, bones: n, frames: 0, index: 0, lw: null, rw: null };
        }
      } catch (e) {
        fail(`Avatar failed to load: ${e && e.message ? e.message : e}`);
      }
    })();

    rafId = requestAnimationFrame(tick);

    // Test probe: pose one bone directly (playback freezes while a
    // probe is set, so the probed bone holds still for inspection).
    if (typeof window !== 'undefined') {
      window.__vrmProbe = (name, x, y, z, w) => {
        const entry = st.bones[name];
        if (!entry) return `no-bone:${name}:have=${Object.keys(st.bones).length}`;
        entry.node.quaternion.set(x, y, z, w);
        entry.node.updateWorldMatrix(true, false);
        st.probe = { entry, q: new THREE.Quaternion(x, y, z, w) };
        return `ok:${name}:now=${entry.node.quaternion.toArray().map((v) => Number(v).toFixed(4)).join(',')}`;
      };
      window.__vrmUnprobe = () => {
        st.probe = null;
        // Re-arm the ease-back so the avatar relaxes to rest instead of
        // freezing in the probed pose.
        st.resetting = true;
        return 'ok';
      };
      // Scene probes: verify screenshots reflect live rendering.
      window.__vrmBg = (hex) => {
        st.scene.background.set(hex);
        return 'ok';
      };
      window.__vrmHide = () => {
        if (st.vrm) st.vrm.scene.visible = false;
        return 'ok';
      };
      window.__vrmShow = () => {
        if (st.vrm) st.vrm.scene.visible = true;
        return 'ok';
      };
      // Binding audit: is the bone we animate the bone the mesh uses?
      window.__vrmBoneInfo = (name) => {
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
    const loadJson = (u) => fetch(u, { signal: controller.signal }).then((res) => {
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
