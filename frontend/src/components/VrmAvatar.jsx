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

const _tmpQ = new THREE.Quaternion();
const _tmpV = new THREE.Vector3();
const _tmpV2 = new THREE.Vector3();

export default function VrmAvatar({ landmarkUrl, playlistUrls, onError }) {
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
  });

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

      // Avatar rest palm basis per side (mirrors Unity BuildSideBasis).
      const avBasis = {};
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
        avBasis[side] = { x, y, z };
      }
      st.avBasis = avBasis;

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
        const outward = S.clone().sub(chestPos);
        outward.y = 0;
        if (outward.lengthSq() < 1e-8) outward.set(side === 'left' ? 1 : -1, 0, 0);
        outward.normalize();
        const pole = outward.multiplyScalar(0.45).add(new THREE.Vector3(0, -1, -0.15)).normalize();
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
    const smoothWithDt = (node, target, dt) => {
      const t = 1 - Math.exp(-SMOOTHNESS * Math.max(dt, 0));
      node.quaternion.slerp(target, t);
    };

    const applyFingerBone = (entry, segDirWorld, lmB, avB, dt) => {
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
      const delta = new THREE.Quaternion().setFromUnitVectors(restLocalDir, targetLocal);
      smoothWithDt(node, delta.multiply(restLocal), dt);
    };

    // Two-bone IK: carry the wrist to the landmark wrist trajectory
    // mapped into a signing box in front of the chest. Directions come
    // from the raw image coords (origin top-left, z toward viewer).
    const applyArm = (side, wristRaw, dt) => {
      const A = st.arm && st.arm[side];
      const C = st.chestPos;
      if (!A || !C) return;
      const target = new THREE.Vector3(
        C.x + (wristRaw.x - 0.5) * 0.7,
        C.y + 0.02 + (0.5 - wristRaw.y) * 0.7,
        Math.max(C.z + 0.3 - wristRaw.z * 0.4, C.z + 0.06),
      );
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
      const perp = A.pole.clone().addScaledVector(dirST, -A.pole.dot(dirST));
      if (perp.lengthSq() < 1e-8) perp.set(0, -1, 0);
      perp.normalize();
      const upperDir = dirST.clone().multiplyScalar(Math.cos(angA))
        .addScaledVector(perp, Math.sin(angA)).normalize();
      const elbowPos = A.S.clone().addScaledVector(upperDir, A.lenU);
      const foreDir = target.clone().sub(elbowPos).normalize();

      A.upper.node.parent.updateWorldMatrix(true, false);
      const parentQU = new THREE.Quaternion();
      A.upper.node.parent.getWorldQuaternion(parentQU);
      const qU = new THREE.Quaternion()
        .setFromUnitVectors(A.restDirU, upperDir).multiply(A.restWQU);
      smoothWithDt(A.upper.node, parentQU.invert().multiply(qU), dt);

      A.lower.node.parent.updateWorldMatrix(true, false);
      const parentQL = new THREE.Quaternion();
      A.lower.node.parent.getWorldQuaternion(parentQL);
      const qL = new THREE.Quaternion()
        .setFromUnitVectors(A.restDirL, foreDir).multiply(A.restWQL);
      smoothWithDt(A.lower.node, parentQL.invert().multiply(qL), dt);
    };

    const applyHand = (hand, isLeft, dt) => {
      const lms = hand.landmarks;
      if (!lms || lms.length !== 21) return;
      const side = isLeft ? 'left' : 'right';
      // Arms first so the hands ride up into signing space.
      applyArm(side, lms[0], dt);
      const avB = st.avBasis[side];
      if (!avB) return;
      const P = (i) => lmVec(lms[i], new THREE.Vector3());
      const p0 = P(0);
      const p5 = P(5);
      const p9 = P(9);
      const lmB = basisFromPalm(p0, p5, p9);
      if (!lmB) return;
      const qAv = quatFromBasis(avB);
      const qLm = quatFromBasis(lmB);
      const palmDelta = qLm.multiply(qAv.invert());

      const handEntry = st.bones[side + 'Hand'];
      if (handEntry) {
        smoothWithDt(handEntry.node, palmDelta.multiply(handEntry.restLocal.clone()), dt);
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
          applyFingerBone(entry, seg(f.lm[j], f.lm[j + 1]), lmB, avB, dt);
        }
      }
      for (let j = 0; j < 3; j++) {
        const entry = st.bones[side + 'Thumb' + THUMB_BONES[j]];
        const [a, b] = THUMB_SEGS[j];
        applyFingerBone(entry, seg(a, b), lmB, avB, dt);
      }
    };

    const applyFrame = (frame, dt) => {
      if (!frame || !Array.isArray(frame.hands)) return;
      for (const hand of frame.hands) {
        if (!hand || !Array.isArray(hand.landmarks) || hand.landmarks.length !== 21) continue;
        const h = String(hand.handedness || '').toLowerCase();
        if (!h.includes('left') && !h.includes('right')) continue;
        try {
          applyHand(hand, h.includes('left'), dt);
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
          }
          advanced = true;
        }
        if (advanced || st.frameTimer === dt) {
          applyFrame(ds.frames[st.frameIndex], dt);
        }
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
