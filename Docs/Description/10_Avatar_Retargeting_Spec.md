# 10 - Avatar Retargeting Spec

How a stream of MediaPipe landmarks becomes a posed VRM avatar, and the
constants the two implementations must agree on.

## Why this document exists

The avatar is driven by **two independent implementations of the same
algorithm**:

| Implementation | File | Used by |
|---|---|---|
| Web | `frontend/src/components/VrmAvatar.jsx` | browser demo (three.js) |
| Unity | `ISL-3D Avatar/Assets/Scripts/HandPoseMapper.cs` | desktop build (UniVRM) |

They are not shared code: one is JavaScript against three.js, the other C#
against Unity transforms, and neither can import the other. Keeping them in
step is therefore a manual act, and they had already drifted - the same
namaste source produced a measured hand separation of 0.0547 m on the web and
0.0481 m in Unity, and only one side carried a fix at a time during
development.

The constants below are therefore declared **normative here**, and
`backend/tests/test_retarget_conformance.py` parses this table plus both
source files and fails if any of the three disagrees. It runs in CI.

## The problem the geometry solves

The original torso collision volume was a capsule on the hips-to-neck axis
with `radius = 0.24 x |hips - neck|`, which is 0.113 m on this avatar. The
chest actually reaches 0.175 m from the spine axis at y 1.40-1.50 m - exactly
the height a namaste drives the hands to. Two consequences:

- the wrist was clamped to a surface **inside** the bust, so the torso mesh
  was drawn over the palm ("the hands join and part of them disappears");
- the clamp constrained a single **point**, while a hand is ~16 cm long, so
  the rest of the hand could sit inside the chest even with the wrist legal.

The fix is to measure the real surface once at load and to constrain the hand
volume, not the wrist point.

## Shared algorithm shape

1. **Measure the torso once**, at avatar load, from the skinned meshes:
   `TORSO_PROFILE_BANDS` bands from hips to neck; within each band the front
   surface is the furthest-forward vertex inside a narrow slab around the
   spine axis (`|x - axisX| < TORSO_MIDLINE_HALF`). The slab is what excludes
   the shoulders and arms, which reach the same heights at `|x| > 0.15`.
2. **Drive the wrist** from the landmark, with the usual scale/offset and
   smoothing.
3. **Guard the whole hand volume.** For the hand's points (palm vertices and
   finger joints), find the furthest-forward torso surface anywhere across the
   hand's vertical span - `HAND_SPAN_SAMPLES` samples over
   `[y - HAND_SPAN_DOWN, y + HAND_SPAN_UP]` - and push the hand out until it
   clears that surface by at least `HAND_CLEARANCE`. Sampling the span rather
   than the wrist height alone is what stops a single band dropout (or the
   bust apex) from undoing the guard. The push is capped at `HAND_BACK_MAX`
   so a degenerate hand orientation can never fling the arm forward.
4. **Mesh past the last finger joint** by `FINGER_TIP_MARGIN`; the finger tips
   are geometry, not joints, and hang beyond the final bone.
5. **Contact assist (namaste clasp).** When the source holds the hands close
   but not touching, compress their lateral separation toward the pair's
   midpoint, ramped from no effect at `CLASP_THRESHOLD` to maximum at
   `CLASP_FULL`, floored at `CLASP_MIN_SCALE` so the pair is never collapsed
   to a single point. The floor may never be tighter than the **measured**
   hand thickness, so the two meshes cannot be pressed through each other.
6. **Cluster detections, then assign arms by continuity.** The tracker's
   per-frame handedness label is not trustworthy: on the shipped clips 79 of
   184 per-gloss clips flip it mid-clip, and 113 report one physical hand
   twice. Grouping by that label therefore moved a sign to the other arm on a
   flip (across the body, into the space the idle hand occupied) and let two
   detections of one hand drive two arms to the same place. Instead, merge
   detections whose wrists are closer than `HAND_MERGE_DIST` (keep the higher
   `handedness_score`), then give each arm the cluster nearest that arm's
   previous wrist. Image side is only the fallback for an arm with no history
   (a clip's first frame, and the tie-break for a two-hand pair): a
   front-facing signer's right hand is on the viewer's left. `SIDE_SWITCH_HYST`
   is the margin by which the other arm must be nearer before it takes the
   hand over, so slow drift cannot oscillate.
7. **Idle pose instead of the T-pose.** The model's rest pose has both arms
   straight out, which is not a neutral a signer ever returns to. Each
   implementation solves the same two-bone IK once at setup towards a wrist
   target beside the body (`IDLE_LATERAL` outward, `IDLE_DROP` down,
   `IDLE_FWD` forward, all relative to the shoulder and the arm length) and
   keeps the resulting upper/forearm rotations. A side with no landmarks this
   frame eases to that pose rather than freezing wherever the previous clip
   left it, and the web's clip-switch blend passes through it instead of the
   rest pose (which popped both arms out to a T-pose between glosses).
8. **Hold through dropouts, then idle.** A hand does not disappear for one
   frame because the sign ended: the tracker loses it. Short holes are
   bridged by holding that arm's last pose for `HAND_GAP_HOLD` seconds; only a
   hand that stays missing longer eases down to the idle. Without this every
   detection dropout dropped the arm to the hip and snapped it back, reading
   as the arm pumping mid-sign (164 such holes in the shipped clips, median 2
   frames). Holes longer than `HAND_GAP_HOLD` are missing footage rather than
   a hold and are removed from the data by
   `ISL_MediaPipe/clean_landmarks.py` (R4), so playback continues on the next
   detected pose instead of freezing.
9. **Coarticulate, do not reset, between glosses.** A composed sentence plays
   one clip per gloss. Passing through the arms-down idle at every boundary
   reads as a list of words, so the web player crossfades directly into the
   next clip when the hands do not jump (`SWITCH_JUMP_MAX`, 88% of the
   curated boundaries) and only passes through the idle at a sentence edge or
   after a large jump. The composed sentence then reads as continuous
   signing rather than a sequence of isolated signs.

## Normative constants

Every value below must be identical in
`frontend/src/components/VrmAvatar.jsx`,
`ISL-3D Avatar/Assets/Scripts/HandPoseMapper.cs` and this table.

| Constant | Value | Meaning |
|---|---|---|
| `TORSO_PROFILE_BANDS` | 24 | height bands the torso profile is measured in |
| `TORSO_MIDLINE_HALF` | 0.06 | half-width of the midline slab (m) that excludes arms |
| `HAND_CLEARANCE` | 0.005 | standoff kept behind the palm (m) |
| `HAND_SPAN_UP` | 0.17 | hand reach above the wrist (m) |
| `HAND_SPAN_DOWN` | 0.04 | hand reach below the wrist (m) |
| `HAND_SPAN_SAMPLES` | 9 | samples taken across the hand's vertical span |
| `HAND_BACK_MAX` | 0.12 | cap on how far the guard may push the arm forward (m) |
| `FINGER_TIP_MARGIN` | 0.015 | mesh that hangs past the last finger joint (m) |
| `CLASP_THRESHOLD` | 0.3 | clasp ramp start, in metres of lateral separation |
| `CLASP_FULL` | 0.13 | separation at full clasp effect (m) |
| `CLASP_MIN_SCALE` | 0.35 | floor on the contact-assist scale factor |
| `HAND_MERGE_DIST` | 0.05 | wrist distance under which two detections are one hand (normalised image units) |
| `SIDE_SWITCH_HYST` | 0.06 | margin by which the other arm must be nearer before it takes a hand over (normalised) |
| `IDLE_LATERAL` | 0.1 | idle wrist outward offset from the shoulder (x arm length) |
| `IDLE_DROP` | 0.8 | idle wrist drop below the shoulder (x arm length) |
| `IDLE_FWD` | 0.12 | idle wrist forward offset from the shoulder (x arm length) |
| `HAND_GAP_HOLD` | 0.5 | seconds a missing hand keeps its last pose before the arm eases to idle |

## Deliberately not shared

These look like they belong in the table above and do not:

- **Sampling budgets.** The web keeps up to 900 points (`PEN_POINT_MAX`) for
  the penetration probe; Unity keeps 300 (`HAND_POINT_MAX`). These are
  performance budgets for two different renderers, not semantics; both are
  "every nth point", so the measured geometry agrees even though the counts
  differ.
- **Source normalisation.** The web calibrates a span from the clip itself
  (`CALIB_X_SPAN`, `CALIB_MIN_SPAN_Y`, `CALIB_MIN_SPAN_Z`, `LAT_GAIN`); Unity
  applies fixed factors (`LATERAL_SCALE`, `HEIGHT_SCALE`). The two produce the
  same wrist positions for the same clip; they are different routes to it, and
  unifying them would be a behaviour change, not a refactor.
- **Web-only reactive tuning.** `SMOOTHNESS`, `FLEX_LIMITS` and
  `ELBOW_MAX_RAD` serve the browser's per-frame interpolation and are not part
  of the shared geometry.
- **Idle convergence budget.** The web converges the idle by iterating the
  smoothed IK (`IDLE_SETTLE_STEPS`); Unity solves it once with smoothing
  bypassed. Same pose, different route.
- **Finger relax.** The web also eases the idle hand's wrist and fingers back
  to rest, because its playback model can leave a fist from the previous
  clip; Unity's per-frame hand application overwrites them anyway.
- **Clip-switch blending.** The web player crossfades between the clips of a
  composed sentence (`BLEND_LOOP_S`, `BLEND_SWITCH_S`, `SWITCH_JUMP_MAX`) and
  else passes through the idle; Unity applies each clip's frames straight into
  the same smoothed IK, so it has no clip-boundary pose to blend. The semantic
  that matters - arms do not reset to the idle between words - is the same.

## Changing a value

Change it in all three places (the two sources and the table), then run:

```
cd backend
pytest -q tests/test_retarget_conformance.py
```

If the value is tuned per renderer, add it to "Deliberately not shared"
instead of the table - the test asserts the table lists exactly the shared
constants and nothing else.
