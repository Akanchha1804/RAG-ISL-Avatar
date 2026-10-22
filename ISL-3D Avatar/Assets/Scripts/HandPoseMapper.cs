
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Maps MediaPipe hand landmarks onto a humanoid avatar.
///
/// Changes from the previous version:
///   - Uses the shared LandmarkModels data types (no nested duplicates).
///   - Adds LoadLandmarkFile / SetDataset (previously missing — caused a
///     compile error when AvatarWebSocketClient called LoadLandmarkFile).
///   - Calibrated palm-frame retargeting: wrist + index-MCP + middle-MCP
///     define an orthonormal palm basis on both the landmark side and the
///     avatar side; finger segment directions are expressed in that basis.
///     Rotations are applied absolutely (non-accumulating) in LateUpdate
///     after the Animator has posed the skeleton.
///   - Configurable axis inversion (invertX/Y/Z), per-axis coordinate scale,
///     rotation smoothing, and optional hand mirroring.
///   - Validates exactly 21 landmarks per hand and skips empty/invalid
///     frames gracefully (never crashes on sparse data).
/// </summary>
public class HandPoseMapper : MonoBehaviour
{
    // ==========================================
    // SETTINGS
    // ==========================================

    [Header("JSON Settings")]
    public string jsonFileName = "landmarks.json";

    [Header("Animation Settings")]
    public bool playAnimation = true;
    public float animationSpeed = 1f;
    public float rotationSmoothness = 15f;

    [Header("Coordinate Calibration")]
    public bool invertX = false;
    public bool invertY = true;
    public bool invertZ = true;
    public Vector3 coordinateScale = Vector3.one;
    [Tooltip("Swap Left/Right landmarks onto the opposite avatar hands.")]
    public bool mirrorHands = false;

    [Header("Backend Fallback (HTTP fetch when not in StreamingAssets)")]
    public string backendUrl = "http://localhost:8000";

    [Header("Arm IK (wrist trajectories into signing space)")]
    [Tooltip("Drive upper arms from the landmark wrist trajectory (two-bone IK). "
        + "Hands alone only rotate in place; without this the avatar signs at its hips.")]
    public bool enableArmIK = true;
    [Tooltip("Scales the signing box in front of the chest (avatar-size tuning).")]
    public float signingScale = 1f;

    // Lateral / height scales from the source centre. Class scope so the
    // contact assist below can compute the same lateral coordinates the arm
    // solver uses. X is deliberately NOT range-normalised: normalising by the
    // clip's own x-span maps a compact pose onto the avatar's full signing
    // width and the hands can never meet (web parity).
    const float LATERAL_SCALE = 0.8f;
    const float HEIGHT_SCALE = 0.5f;
    // Contact assist (web parity): when the source holds the two hands close
    // together but not quite touching (namaste source wrists read ~0.14
    // frame-widths apart), compress the pair's lateral separation toward
    // their own midpoint so the palms actually meet. Ramped from
    // CLASP_THRESHOLD (no effect) to CLASP_FULL (max), so shoulder-width and
    // wider poses are left untouched.
    const float CLASP_THRESHOLD = 0.3f;
    const float CLASP_FULL = 0.13f;
    const float CLASP_MIN_SCALE = 0.35f;
    // Hand-to-arm assignment (web parity). The tracker's per-frame handedness
    // label is not trustworthy on the shipped clips: it flips mid-clip (79 of
    // 184 per-gloss files) and one physical hand is sometimes reported twice
    // (113 of 184). Bucketing by that label drove the wrong arm across the
    // body on a flip, and two detections of one hand fought over a single arm
    // - the overlapping hands in composed playback. Detections are clustered,
    // then assigned to an arm by continuity with that arm's previous wrist;
    // image side is only the fallback for a hand with no history yet.
    const float HAND_MERGE_DIST = 0.05f;
    const float SIDE_SWITCH_HYST = 0.06f;
    // Idle arm pose (web parity). The model's rest pose is a T-pose, so a
    // side with no landmarks (or the neutral a clip switch blends through) is
    // eased to a wrist target beside the body instead of freezing with the
    // arm straight out. DROP/FWD are fractions of the arm length; LATERAL is
    // the wrist's outward offset from the shoulder.
    const float IDLE_LATERAL = 0.1f;
    const float IDLE_DROP = 0.8f;
    const float IDLE_FWD = 0.12f;
    private float claspAdjLeft, claspAdjRight;
    private bool clasping;
    // Measured hand thickness (metres): the contact assist may press the
    // palms together but never drive the two meshes through each other.
    private float handMinSep = 0f;

    // Composed-playback playlist (novel sentences): per-gloss landmark
    // files played in order. Sentence clips ignore the queue entirely.
    private readonly Queue<string> playlistQueue = new Queue<string>();
    private bool playlistMode = false;

    // Previous-frame wrist positions (raw landmark coords) for the
    // per-side single-hand rule.
    private Vector3 lastWristLeft, lastWristRight;
    // Side the last single-hand frame was assigned to. Hysteresis only: an
    // arm only takes a hand over from the other arm once that arm's previous
    // wrist is clearly farther away.
    private HandSide? lastSingleSide;
    private bool haveLastWristLeft, haveLastWristRight;
    // Previous wrist orientations for the angular-velocity cap.
    private Quaternion lastWristQLeft, lastWristQRight;
    private bool haveLastWristQLeft, haveLastWristQRight;

    // Last good landmark palm basis per side (degenerate-frame hold, web
    // parity). Held together with the arm target it was captured at so a
    // stale basis on a traveled arm is skipped rather than kinked.
    private Vector3 lastPalmLeftX, lastPalmLeftY, lastPalmLeftZ, lastPalmLeftAt;
    private Vector3 lastPalmRightX, lastPalmRightY, lastPalmRightZ, lastPalmRightAt;
    private bool haveLastPalmLeft, haveLastPalmRight;
    // The posed hand's world rotation change for the frame being animated.
    // Finger bones are aimed against the REST-pose avatar palm basis, so this
    // carries the wrist's own rotation into them (web parity fix: without it
    // a rolled wrist left the fingers ~90 degrees off the new palm axis).
    private Quaternion fingerHandDelta = Quaternion.identity;
    // Torso collision capsule (hips->neck axis + radius), web parity.
    private Vector3 torsoHips, torsoNeck;
    private float torsoRadius = 0.15f;
    private bool haveTorsoCapsule = false;

    // ---- measured chest surface (web parity) ----
    // The capsule radius above is 0.24 * |hips->neck| = 0.11 m on this avatar,
    // but its bust actually reaches 0.175 m from the spine axis at y 1.40-1.50
    // -- exactly the height a namaste drives the hands to. Clamping the wrist
    // POINT to that radius therefore parked it ~3 cm inside the chest and the
    // torso mesh was drawn over the palm ("the hands join and part of them
    // disappears"). These profile tables are measured from the skinned meshes
    // at setup instead of assumed, and the guard below constrains the whole
    // HAND, not the wrist point.
    const int TORSO_PROFILE_BANDS = 24;
    const float TORSO_MIDLINE_HALF = 0.06f;   // geometric front-slab half width
    const float HAND_CLEARANCE = 0.005f;      // standoff kept behind the palm
    const float HAND_SPAN_UP = 0.17f;         // hand reach above the wrist (m)
    const float HAND_SPAN_DOWN = 0.04f;
    const int HAND_SPAN_SAMPLES = 9;
    const float HAND_BACK_MAX = 0.12f;        // never fling the arm forward
    const float FINGER_TIP_MARGIN = 0.015f;   // mesh past the last finger joint
    const int HAND_POINT_MAX = 300;           // palm samples kept per hand

    private float torsoHipsY, torsoAxisLen, torsoHipsX, torsoNeckX;
    private float[] torsoFrontZ;              // per band: front surface world z
    private float[] torsoHalfX;               // per band: half width, arms excluded
    private bool haveTorsoProfile = false;
    // Palm vertices in each hand bone's own frame (rigid to that bone), plus
    // that bone's joint chain, rebuilt at setup; index 0 = left, 1 = right.
    private Vector3[][] handPts = new Vector3[2][];
    private Transform[] handBoneXf = new Transform[2];
    private Transform[][] handJoints = new Transform[2][];

    // Two-bone arm rig per side (mirrors the web VrmAvatar solver so both
    // viewports carry the wrists into the same signing box). The landmark
    // files carry hand points only; without arms the hands would sign down
    // at the hips. Null when the avatar lacks the required bones, in which
    // case hands still rotate and arms stay on the Animator.
    private class ArmRig
    {
        public Transform upper, lower, hand;
        public bool isLeft;
        public Vector3 shoulder;      // rest world position (S)
        public float lenU, lenL;      // upper / forearm lengths
        public Quaternion restWQU, restWQL;
        public Vector3 restDirU, restDirL;
        public Vector3 pole;          // elbow hint (world)
        public Vector3 chest;         // signing-box anchor (world)
        // Arms-down idle solved once at setup (web parity); the rest pose is
        // a T-pose, which is not a neutral a signer ever returns to.
        public bool hasIdle;
        public Quaternion idleUpper, idleLower, handRestLocal;
    }
    private ArmRig leftArm, rightArm;

    // Per-clip signing-space calibration (M6 fix): source videos frame the
    // signer differently, so absolute image coords reproduce the wrong height
    // (waist-level source -> waist-level avatar). Each dataset's wrist
    // trajectory bbox is mapped into the signing box instead. Min-span
    // guarded so a static hold maps mid-box rather than amplifying jitter.
    private Vector3 calibMin, calibSpan, calibCenter;
    private bool calibrated = false;

    // ---- diagnostics (opt-in; zero cost while disabled) ----
    // Records what each bone was actually driven by and where the anatomical
    // clamp bit, so the retarget probe can verify the segment -> bone mapping
    // at runtime instead of inferring it from aggregate motion.
    [System.Serializable]
    public struct BoneDriveDiag
    {
        public HumanBodyBones bone;
        public int startIdx;           // landmark indices actually used, so the
        public int endIdx;             // segment->bone mapping is provable directly
        public Vector3 segWorld;      // world-space direction of the driving segment
        public Vector3 restLocal;     // rest reference direction (parent space)
        public Vector3 targetLocal;   // target before clamping (parent space)
        public float flexAngPreClamp; // degrees between rest and target
        public float maxFlexDeg;
        public bool clamped;
        public float damp;
        public float appliedDeg;      // rotation actually applied vs rest
    }

    private readonly Dictionary<HumanBodyBones, BoneDriveDiag> lastDiagnostics =
        new Dictionary<HumanBodyBones, BoneDriveDiag>();
    private bool captureDiagnostics = false;

    /// <summary>When enabled, every finger bone records its drive inputs.</summary>
    public bool CaptureDiagnostics
    {
        get { return captureDiagnostics; }
        set { captureDiagnostics = value; }
    }

    /// <summary>Latest per-bone drive record (only valid while capturing).</summary>
    public Dictionary<HumanBodyBones, BoneDriveDiag> LastDiagnostics => lastDiagnostics;

    // Last wrist targets from the arm IK, for the playback probe to compare
    // against where the wrists actually ended up.
    private Vector3 lastTargetLeft, lastTargetRight;
    public Vector3 LastTargetLeft => lastTargetLeft;
    public Vector3 LastTargetRight => lastTargetRight;

    // ==========================================
    // PRIVATE VARIABLES
    // ==========================================

    private LandmarkDataset dataset;
    private Animator animator;
    private Coroutine httpLoadCoroutine;

    private int currentFrameIndex = 0;
    private float frameTimer = 0f;

    private readonly Dictionary<HumanBodyBones, Transform> boneMap =
        new Dictionary<HumanBodyBones, Transform>();

    private readonly Dictionary<HumanBodyBones, Quaternion> initialRotations =
        new Dictionary<HumanBodyBones, Quaternion>();

    // Rest-pose WORLD rotation. palmDelta is a world-space rotation, so the
    // wrist target must rotate the hand's rest WORLD orientation and only
    // then be expressed in the (current) parent's space. Applying palmDelta
    // as a local rotation is only correct when the arm parent happens to be
    // identity, which it never is once arm IK has run.
    private readonly Dictionary<HumanBodyBones, Quaternion> initialWorldRotations =
        new Dictionary<HumanBodyBones, Quaternion>();

    // Rest-pose segment direction in the bone's parent space (palm-frame reference).
    private readonly Dictionary<HumanBodyBones, Vector3> restLocalDirections =
        new Dictionary<HumanBodyBones, Vector3>();

    // Cached palm basis for the avatar's rest pose (per side).
    private Vector3 avatarLeftOrigin, avatarLeftAxisX, avatarLeftAxisY, avatarLeftAxisZ;
    private Vector3 avatarRightOrigin, avatarRightAxisX, avatarRightAxisY, avatarRightAxisZ;

    // ==========================================
    // LIFECYCLE
    // ==========================================

    void Start()
    {
        animator = GetComponent<Animator>();
        if (animator == null)
            animator = GetComponentInChildren<Animator>();

        if (animator == null)
        {
            Debug.LogError("HandPoseMapper: Animator not found.");
            enabled = false;
            return;
        }

        if (!animator.isHuman)
        {
            Debug.LogError("HandPoseMapper: Avatar is not Humanoid.");
            enabled = false;
            return;
        }

        SetupBones();
        BuildAvatarPalmBasis();
        SetupArms();

        // Load default file if present; otherwise wait for SetDataset /
        // LoadLandmarkFile from LandmarkFetcher or AvatarWebSocketClient.
        if (!TryLoadFromStreamingAssets(jsonFileName))
        {
            Debug.Log("HandPoseMapper: No default landmark file; waiting for SetDataset/LoadLandmarkFile.");
        }
    }

    void Update()
    {
        if (!playAnimation || dataset == null || dataset.frames == null || dataset.frames.Count == 0)
            return;

        float fps = dataset.fps > 0f ? dataset.fps : 25f;
        frameTimer += Time.deltaTime * animationSpeed;
        float frameDuration = 1f / fps;

        bool wrapped = false;
        while (frameTimer >= frameDuration)
        {
            frameTimer -= frameDuration;
            currentFrameIndex++;
            if (currentFrameIndex >= dataset.frames.Count)
            {
                currentFrameIndex = 0;
                wrapped = true;
            }
        }

        // Composed playback: each clip plays once, in order. The final
        // clip keeps looping (predictable idle) until a new file/playlist
        // arrives. Single sentence files never enter playlist mode.
        if (playlistMode && wrapped)
        {
            if (playlistQueue.Count > 0)
                LoadLandmarkFile(playlistQueue.Dequeue(), fromPlaylist: true);
            else
                playlistMode = false;
        }
    }

    // Pose application runs after Animator so our rotations are not overwritten.
    void LateUpdate()
    {
        if (!playAnimation || dataset == null || dataset.frames == null || dataset.frames.Count == 0)
            return;
        if (currentFrameIndex < 0 || currentFrameIndex >= dataset.frames.Count)
            return;

        FrameData frame = dataset.frames[currentFrameIndex];
        if (frame == null || frame.hands == null || frame.hands.Count == 0)
            return; // sparse frame: skip gracefully, keep previous pose

        // Cluster the frame's detections first (web parity): the tracker
        // reports one physical hand twice often enough to matter, and two
        // copies drive two arms that then occupy the same space.
        var clusters = new List<HandData>();
        foreach (HandData hand in frame.hands)
        {
            if (!HandIdentity.IsValidForAnimation(hand))
                continue;
            Vector3 w = ToRaw(hand.landmarks[0]);
            int near = -1;
            for (int i = 0; i < clusters.Count; i++)
            {
                Vector3 p = ToRaw(clusters[i].landmarks[0]);
                float dx = w.x - p.x;
                float dy = w.y - p.y;
                if (Mathf.Sqrt(dx * dx + dy * dy) < HAND_MERGE_DIST)
                {
                    near = i;
                    break;
                }
            }
            if (near < 0)
            {
                clusters.Add(hand);
            }
            else if (hand.handedness_score > clusters[near].handedness_score)
            {
                // One hand seen twice: keep the more confident detection.
                clusters[near] = hand;
            }
        }

        // Assign clusters to arms by CONTINUITY - the arm whose previous
        // wrist is nearest keeps the hand - never by the per-frame handedness
        // label, which flips mid-clip and would swap the signing arm. Image
        // side is only the fallback for an arm with no history yet.
        HandData leftPick = null, rightPick = null;
        if (clusters.Count == 1)
        {
            HandData hand = clusters[0];
            Vector3 wrist = ToRaw(hand.landmarks[0]);
            float cl = WristCost(wrist, lastWristLeft, haveLastWristLeft);
            float cr = WristCost(wrist, lastWristRight, haveLastWristRight);
            HandSide side;
            if (Mathf.Approximately(cl, cr))
            {
                // No history on either arm: the signer faces the camera, so
                // image-left is the signer's right hand = the avatar's right.
                side = hand.landmarks[0].x > 0.5f ? HandSide.Left : HandSide.Right;
            }
            else
            {
                side = cl < cr ? HandSide.Left : HandSide.Right;
                HandSide other = side == HandSide.Left ? HandSide.Right : HandSide.Left;
                if (lastSingleSide == other && Mathf.Abs(cl - cr) < SIDE_SWITCH_HYST)
                    side = other;
            }
            lastSingleSide = side;
            if (side == HandSide.Left)
                leftPick = hand;
            else
                rightPick = hand;
        }
        else if (clusters.Count >= 2)
        {
            // Two or more distinct hands: keep the two most confident and pick
            // the pairing (or its swap) with the lower total distance. A tie
            // means neither arm has history, and the lateral rule breaks it.
            clusters.Sort((a, b) => b.handedness_score.CompareTo(a.handedness_score));
            HandData a0 = clusters[0];
            HandData a1 = clusters[1];
            bool swap = a0.landmarks[0].x > a1.landmarks[0].x;
            HandData leftMost = swap ? a1 : a0;   // smaller image x
            HandData rightMost = swap ? a0 : a1;
            float straight =
                WristCost(ToRaw(leftMost.landmarks[0]), lastWristRight, haveLastWristRight)
                + WristCost(ToRaw(rightMost.landmarks[0]), lastWristLeft, haveLastWristLeft);
            float crossed =
                WristCost(ToRaw(leftMost.landmarks[0]), lastWristLeft, haveLastWristLeft)
                + WristCost(ToRaw(rightMost.landmarks[0]), lastWristRight, haveLastWristRight);
            if (straight <= crossed)
            {
                rightPick = leftMost;
                leftPick = rightMost;
            }
            else
            {
                leftPick = leftMost;
                rightPick = rightMost;
            }
            lastSingleSide = null;
        }
        if (mirrorHands)
        {
            HandData swapPick = leftPick;
            leftPick = rightPick;
            rightPick = swapPick;
        }
        // Contact assist (web parity): compute the pair's lateral squeeze
        // BEFORE the hands are animated, because SolveArm consumes it.
        claspAdjLeft = 0f;
        claspAdjRight = 0f;
        clasping = false;
        if (leftPick != null && rightPick != null)
        {
            float ls = Mathf.Max(signingScale, 1e-3f);
            float latL = (ToRaw(leftPick.landmarks[0]).x - calibCenter.x) * LATERAL_SCALE * ls;
            float latR = (ToRaw(rightPick.landmarks[0]).x - calibCenter.x) * LATERAL_SCALE * ls;
            float gap = Mathf.Abs(latL - latR);
            if (gap > 1e-4f && gap < CLASP_THRESHOLD)
            {
                float t = Mathf.Clamp01(
                    (CLASP_THRESHOLD - gap) / (CLASP_THRESHOLD - CLASP_FULL));
                float scale = 1f - t * (1f - CLASP_MIN_SCALE);
                // Palms are solid: CLASP_MIN_SCALE alone could drive the pair
                // closer than the hands are thick, so the two meshes pass
                // through each other. Floor the compression at the measured
                // hand thickness.
                if (handMinSep > 0f && gap * scale < handMinSep)
                    scale = Mathf.Min(1f, handMinSep / gap);
                float mid = (latL + latR) * 0.5f;
                claspAdjLeft = (mid - latL) * (1f - scale);
                claspAdjRight = (mid - latR) * (1f - scale);
                clasping = true;
            }
        }
        if (leftPick != null)
        {
            lastWristLeft = ToRaw(leftPick.landmarks[0]);
            haveLastWristLeft = true;
            AnimateHand(leftPick, true);
        }
        else
        {
            // No landmarks for this side: ease to the arms-down idle instead
            // of freezing wherever the previous clip left it (web parity).
            RelaxArm(leftArm);
        }
        if (rightPick != null)
        {
            lastWristRight = ToRaw(rightPick.landmarks[0]);
            haveLastWristRight = true;
            AnimateHand(rightPick, false);
        }
        else
        {
            RelaxArm(rightArm);
        }
    }

    /// <summary>
    /// Distance from a wrist to an arm's previous wrist, in raw landmark
    /// units. A fixed cost when the arm has no history, so "no history" loses
    /// to any real distance and ties fall back to the lateral rule.
    /// </summary>
    float WristCost(Vector3 wrist, Vector3 previous, bool havePrevious)
    {
        if (!havePrevious)
            return 0.5f;
        float dx = wrist.x - previous.x;
        float dy = wrist.y - previous.y;
        return Mathf.Sqrt(dx * dx + dy * dy);
    }

    // ==========================================
    // PUBLIC API: dataset loading
    // ==========================================

    /// <summary>
    /// Load a landmark file by name. Tries StreamingAssets/Landmarks first,
    /// then falls back to HTTP GET {backendUrl}/landmarks/{fileName}.
    /// Called by AvatarWebSocketClient on landmark_file / landmark_update.
    /// A direct file load always exits playlist mode (sentence clips play
    /// on their own); pass fromPlaylist for internal queue advancement.
    public void LoadLandmarkFile(string fileName, bool fromPlaylist = false)
    {
        if (string.IsNullOrEmpty(fileName))
        {
            Debug.LogWarning("HandPoseMapper.LoadLandmarkFile: empty file name.");
            return;
        }

        if (!fromPlaylist)
            playlistMode = false;

        // Strip any directory components the server may have included.
        fileName = Path.GetFileName(fileName);

        if (TryLoadFromStreamingAssets(fileName))
            return;

        if (httpLoadCoroutine != null)
            StopCoroutine(httpLoadCoroutine);
        httpLoadCoroutine = StartCoroutine(FetchFromBackend(fileName));
    }

    /// <summary>Directly inject an already-parsed dataset (used by LandmarkFetcher).</summary>
    public void SetDataset(LandmarkDataset newDataset)
    {
        if (newDataset == null || newDataset.frames == null || newDataset.frames.Count == 0)
        {
            Debug.LogWarning("HandPoseMapper.SetDataset: null or empty dataset ignored.");
            return;
        }

        dataset = newDataset;
        currentFrameIndex = 0;
        frameTimer = 0f;
        haveLastWristLeft = haveLastWristRight = false;
        haveLastWristQLeft = haveLastWristQRight = false;
        CalibrateSigningSpace();
        Debug.Log($"HandPoseMapper: dataset set ({dataset.frames.Count} frames).");
    }

    public LandmarkDataset GetDataset() => dataset;

    /// <summary>Frame currently applied (BoneAudit reads this to align samples).</summary>
    public int CurrentFrameIndex => currentFrameIndex;

    /// <summary>Jump to a specific frame (used by the sign-playback probe).</summary>
    public void SetFrameIndex(int index)
    {
        if (dataset == null || dataset.frames == null || dataset.frames.Count == 0) return;
        currentFrameIndex = Mathf.Clamp(index, 0, dataset.frames.Count - 1);
        frameTimer = 0f;
    }

    /// <summary>Per-bone rest reference direction, or false when the bone is
    /// absent or has no usable reference direction.</summary>
    public bool TryGetRestDirection(HumanBodyBones bone, out Vector3 dir)
    {
        if (restLocalDirections.TryGetValue(bone, out dir)) return true;
        dir = Vector3.zero;
        return false;
    }

    /// <summary>True when the animator actually resolved this bone.</summary>
    public bool HasBone(HumanBodyBones bone) =>
        boneMap.TryGetValue(bone, out Transform t) && t != null;

    /// <summary>True when this clip's signing-space calibration succeeded.</summary>
    public bool IsCalibrated => calibrated;

    /// <summary>True when arm IK is enabled (so the playback probe can confirm).</summary>
    public bool IsArmIkEnabled => enableArmIK;

    /// <summary>Avatar rest-palm normal per side, for the playback probe.</summary>
    public Vector3 LeftPalmNormal => avatarLeftAxisZ;
    public Vector3 RightPalmNormal => avatarRightAxisZ;

    /// <summary>
    /// World-space palm normal from a raw landmark hand (wrist + index MCP +
    /// middle MCP), using the same ToWorld convention as the mapper. Lets
    /// the probe compare the landmark frame against LeftPalmNormal /
    /// RightPalmNormal per side and expose a chirality mismatch directly.
    /// </summary>
    public Vector3 LandmarkPalmNormal(HandData hand)
    {
        if (hand == null || hand.landmarks == null || hand.landmarks.Count < 10)
            return Vector3.zero;
        Vector3 o = ToWorld(hand.landmarks[0]);
        Vector3 i = ToWorld(hand.landmarks[5]) - o;
        Vector3 m = ToWorld(hand.landmarks[9]) - o;
        Vector3 n = Vector3.Cross(i, m);
        return n.sqrMagnitude < 1e-10f ? Vector3.zero : n.normalized;
    }

    /// <summary>
    /// Fit the dataset's wrist trajectory into signing space. Called on
    /// every dataset load (file, playlist item, direct inject) so each clip
    /// is calibrated to its own framing.
    /// </summary>
    void CalibrateSigningSpace()
    {
        calibrated = false;
        if (dataset == null || dataset.frames == null)
            return;

        System.Collections.Generic.List<float> xs =
            new System.Collections.Generic.List<float>();
        System.Collections.Generic.List<float> ys =
            new System.Collections.Generic.List<float>();
        System.Collections.Generic.List<float> zs =
            new System.Collections.Generic.List<float>();
        foreach (FrameData frame in dataset.frames)
        {
            if (frame == null || frame.hands == null)
                continue;
            foreach (HandData hand in frame.hands)
            {
                if (hand == null || hand.landmarks == null || hand.landmarks.Count == 0)
                    continue;
                LandmarkPoint w = hand.landmarks[0];
                xs.Add(w.x);
                ys.Add(w.y);
                zs.Add(w.z);
            }
        }
        if (xs.Count == 0)
            return;

        // Percentile bbox: edge-of-frame mistracks must not define the range.
        float minX = Pct(xs, 0.02f), maxX = Pct(xs, 0.98f);
        float minY = Pct(ys, 0.02f), maxY = Pct(ys, 0.98f);
        float minZ = Pct(zs, 0.02f), maxZ = Pct(zs, 0.98f);

        calibMin = new Vector3(FitMin(minX, maxX, 0.12f), FitMin(minY, maxY, 0.12f), FitMin(minZ, maxZ, 0.05f));
        calibSpan = new Vector3(
            Mathf.Max(maxX - minX, 0.12f),
            Mathf.Max(maxY - minY, 0.12f),
            Mathf.Max(maxZ - minZ, 0.05f));
        // Lateral source centre. X is deliberately NOT range-normalised in
        // SolveArm: normalising by the clip's own x-span maps whatever the
        // source happened to span onto the avatar's full signing width, which
        // destroyed gestures whose hands are close together (namaste source
        // hands are ~0.14 of frame width apart, yet landed 0.55 m apart).
        calibCenter = new Vector3(
            (minX + maxX) * 0.5f,
            (minY + maxY) * 0.5f,
            (minZ + maxZ) * 0.5f);
        calibrated = true;
    }

    float Pct(System.Collections.Generic.List<float> values, float q)
    {
        values.Sort();
        int i = Mathf.Min(values.Count - 1, Mathf.FloorToInt(q * values.Count));
        return values[Mathf.Max(0, i)];
    }

    float FitMin(float lo, float hi, float minSpan)
    {
        float span = Mathf.Max(hi - lo, minSpan);
        return (lo + hi) * 0.5f - span * 0.5f;
    }

    Vector3 NormalizeWrist(Vector3 raw)
    {
        if (!calibrated)
            return raw;
        return new Vector3(
            Mathf.Clamp((raw.x - calibMin.x) / calibSpan.x, -0.2f, 1.2f),
            Mathf.Clamp((raw.y - calibMin.y) / calibSpan.y, -0.2f, 1.2f),
            Mathf.Clamp((raw.z - calibMin.z) / calibSpan.z, -0.2f, 1.2f));
    }

    /// <summary>
    /// Play per-gloss landmark files in order (composed/novel sentences).
    /// Each clip plays once; the final clip keeps looping until replaced.
    /// A later single-file load (sentence clip) always exits playlist mode.
    /// </summary>
    public void PlayPlaylist(System.Collections.Generic.List<string> files)
    {
        playlistQueue.Clear();
        if (files != null)
        {
            foreach (string f in files)
            {
                if (!string.IsNullOrEmpty(f))
                    playlistQueue.Enqueue(f);
            }
        }
        playlistMode = playlistQueue.Count > 0;
        if (playlistMode)
            LoadLandmarkFile(playlistQueue.Dequeue(), fromPlaylist: true);
        else
            Debug.LogWarning("HandPoseMapper.PlayPlaylist: empty playlist ignored.");
    }

    // ==========================================
    // LOADING HELPERS
    // ==========================================

    bool TryLoadFromStreamingAssets(string fileName)
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, "Landmarks", fileName);
        if (!File.Exists(filePath))
            return false;

        try
        {
            string json = File.ReadAllText(filePath);
            return ApplyJson(json, "file " + filePath);
        }
        catch (Exception e)
        {
            Debug.LogError($"HandPoseMapper: read error for {filePath}: {e.Message}");
            return false;
        }
    }

    IEnumerator FetchFromBackend(string fileName)
    {
        string url = $"{backendUrl.TrimEnd('/')}/landmarks/{fileName}";
        Debug.Log($"HandPoseMapper: fetching {url}");

        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"HandPoseMapper: fetch failed: {request.error}");
                yield break;
            }

            ApplyJson(request.downloadHandler.text, "backend " + url);
        }
    }

    bool ApplyJson(string json, string source)
    {
        LandmarkDataset parsed;
        try
        {
            parsed = JsonUtility.FromJson<LandmarkDataset>(json);
        }
        catch (Exception e)
        {
            Debug.LogError($"HandPoseMapper: JSON parse error ({source}): {e.Message}");
            return false;
        }

        if (parsed == null || parsed.frames == null || parsed.frames.Count == 0)
        {
            Debug.LogError($"HandPoseMapper: invalid or empty dataset ({source}).");
            return false;
        }

        dataset = parsed;
        currentFrameIndex = 0;
        frameTimer = 0f;
        haveLastWristLeft = haveLastWristRight = false;
        haveLastWristQLeft = haveLastWristQRight = false;
        CalibrateSigningSpace();
        Debug.Log($"HandPoseMapper: loaded {dataset.frames.Count} frames from {source}.");
        return true;
    }

    // ==========================================
    // BONE SETUP
    // ==========================================

    void SetupBones()
    {
        RegisterBone(HumanBodyBones.LeftHand);
        RegisterBone(HumanBodyBones.RightHand);
        RegisterBone(HumanBodyBones.LeftUpperArm);
        RegisterBone(HumanBodyBones.LeftLowerArm);
        RegisterBone(HumanBodyBones.RightUpperArm);
        RegisterBone(HumanBodyBones.RightLowerArm);

        RegisterBone(HumanBodyBones.LeftIndexProximal);
        RegisterBone(HumanBodyBones.LeftIndexIntermediate);
        RegisterBone(HumanBodyBones.LeftIndexDistal);
        RegisterBone(HumanBodyBones.LeftMiddleProximal);
        RegisterBone(HumanBodyBones.LeftMiddleIntermediate);
        RegisterBone(HumanBodyBones.LeftMiddleDistal);
        RegisterBone(HumanBodyBones.LeftRingProximal);
        RegisterBone(HumanBodyBones.LeftRingIntermediate);
        RegisterBone(HumanBodyBones.LeftRingDistal);
        RegisterBone(HumanBodyBones.LeftLittleProximal);
        RegisterBone(HumanBodyBones.LeftLittleIntermediate);
        RegisterBone(HumanBodyBones.LeftLittleDistal);
        RegisterBone(HumanBodyBones.LeftThumbProximal);
        RegisterBone(HumanBodyBones.LeftThumbIntermediate);
        RegisterBone(HumanBodyBones.LeftThumbDistal);

        RegisterBone(HumanBodyBones.RightIndexProximal);
        RegisterBone(HumanBodyBones.RightIndexIntermediate);
        RegisterBone(HumanBodyBones.RightIndexDistal);
        RegisterBone(HumanBodyBones.RightMiddleProximal);
        RegisterBone(HumanBodyBones.RightMiddleIntermediate);
        RegisterBone(HumanBodyBones.RightMiddleDistal);
        RegisterBone(HumanBodyBones.RightRingProximal);
        RegisterBone(HumanBodyBones.RightRingIntermediate);
        RegisterBone(HumanBodyBones.RightRingDistal);
        RegisterBone(HumanBodyBones.RightLittleProximal);
        RegisterBone(HumanBodyBones.RightLittleIntermediate);
        RegisterBone(HumanBodyBones.RightLittleDistal);
        RegisterBone(HumanBodyBones.RightThumbProximal);
        RegisterBone(HumanBodyBones.RightThumbIntermediate);
        RegisterBone(HumanBodyBones.RightThumbDistal);

        Debug.Log($"HandPoseMapper: registered {boneMap.Count} bones.");
    }

    void RegisterBone(HumanBodyBones bone)
    {
        Transform boneTransform = animator.GetBoneTransform(bone);
        if (boneTransform == null)
        {
            Debug.LogWarning($"HandPoseMapper: missing avatar bone {bone}.");
            return;
        }

        boneMap[bone] = boneTransform;
        initialRotations[bone] = boneTransform.localRotation;
        initialWorldRotations[bone] = boneTransform.rotation;

        if (boneTransform.childCount > 0)
        {
            Transform child = boneTransform.GetChild(0);
            Vector3 direction = child.position - boneTransform.position;
            Vector3 localDirection = boneTransform.parent != null
                ? boneTransform.parent.InverseTransformDirection(direction)
                : direction;

            if (localDirection.sqrMagnitude > 0.0001f)
                restLocalDirections[bone] = localDirection.normalized;
        }
        // Tipless Distal bones: inherit the incoming segment direction so
        // the fingertip follows its parent instead of going rigid
        // (web parity).
        if (!restLocalDirections.ContainsKey(bone) && boneTransform.parent != null)
        {
            Vector3 incoming = boneTransform.position - boneTransform.parent.position;
            Vector3 localIncoming = boneTransform.parent.InverseTransformDirection(incoming);
            if (localIncoming.sqrMagnitude > 0.0001f)
                restLocalDirections[bone] = localIncoming.normalized;
        }
    }

    // ==========================================
    // AVATAR PALM BASIS (rest pose)
    // ==========================================

    void BuildAvatarPalmBasis()
    {
        avatarLeftOrigin = Vector3.zero; avatarLeftAxisX = Vector3.right;
        avatarLeftAxisY = Vector3.up; avatarLeftAxisZ = Vector3.forward;
        avatarRightOrigin = Vector3.zero; avatarRightAxisX = Vector3.right;
        avatarRightAxisY = Vector3.up; avatarRightAxisZ = Vector3.forward;

        BuildSideBasis(HumanBodyBones.LeftHand, HumanBodyBones.LeftIndexProximal,
            HumanBodyBones.LeftLittleProximal,
            ref avatarLeftOrigin, ref avatarLeftAxisX, ref avatarLeftAxisY, ref avatarLeftAxisZ);

        BuildSideBasis(HumanBodyBones.RightHand, HumanBodyBones.RightIndexProximal,
            HumanBodyBones.RightLittleProximal,
            ref avatarRightOrigin, ref avatarRightAxisX, ref avatarRightAxisY, ref avatarRightAxisZ);

        // Mirror verification (web parity): each palm normal must point
        // AWAY from the body midline. A mirrored rig would otherwise drive
        // one hand's fingers twisted backward. Correction is a 180-degree
        // roll about the finger axis (keeps a proper rotation).
        VerifyPalmSide(true, ref avatarLeftAxisX, ref avatarLeftAxisY, ref avatarLeftAxisZ);
        VerifyPalmSide(false, ref avatarRightAxisX, ref avatarRightAxisY, ref avatarRightAxisZ);
    }

    void VerifyPalmSide(bool isLeft,
        ref Vector3 axisX, ref Vector3 axisY, ref Vector3 axisZ)
    {
        Transform chest = null;
        try
        {
            chest = animator.GetBoneTransform(HumanBodyBones.Chest)
                ?? animator.GetBoneTransform(HumanBodyBones.Spine);
        }
        catch { /* ignore */ }
        Vector3 mid = chest != null ? chest.position : Vector3.zero;
        Vector3 handPos = isLeft ? avatarLeftOrigin : avatarRightOrigin;
        Vector3 outward = handPos - mid;
        outward.y = 0f;
        if (outward.sqrMagnitude < 1e-8f)
            outward = new Vector3(isLeft ? 1f : -1f, 0f, 0f);
        outward.Normalize();
        // Only correct a CLEARLY inward palm normal. In the imported rest
        // pose the raw cross product is very nearly perpendicular to
        // `outward` (dot ~ 0), where the old `< 0` test flipped on numeric
        // noise and silently twisted one hand 180 degrees (every run logged
        // exactly one "mirror-corrected" side). A genuinely mirrored rig
        // gives a strongly inward normal, which this still catches.
        float inward = Vector3.Dot(axisZ, outward);
        if (inward < -0.35f)
        {
            axisY = -axisY;
            axisZ = -axisZ;
            Debug.Log($"HandPoseMapper: mirror-corrected the {(isLeft ? "left" : "right")} palm basis (dot {inward:F2}).");
        }
        else
        {
            Debug.Log($"HandPoseMapper: {(isLeft ? "left" : "right")} palm normal vs outward dot {inward:F2} (no flip).");
        }
    }

    void BuildSideBasis(
        HumanBodyBones handBone, HumanBodyBones indexMcp, HumanBodyBones littleMcp,
        ref Vector3 origin, ref Vector3 axisX, ref Vector3 axisY, ref Vector3 axisZ)
    {
        if (!boneMap.TryGetValue(handBone, out Transform hand) || hand == null)
            return;

        boneMap.TryGetValue(indexMcp, out Transform index);
        boneMap.TryGetValue(littleMcp, out Transform little);
        if (index == null || little == null)
            return;

        origin = hand.position;
        Vector3 toIndex = index.position - hand.position;
        // Across the knuckles (index MCP -> little MCP), NOT wrist -> middle
        // MCP. Wrist->index-MCP and wrist->middle-MCP are only ~3-20 degrees
        // apart on a flat hand, so their cross product is dominated by
        // landmark noise: the palm normal came out random every frame and the
        // wrist roll was effectively undriven (web parity fix).
        Vector3 across = little.position - index.position;

        if (toIndex.sqrMagnitude < 1e-8f || across.sqrMagnitude < 1e-8f)
            return;

        axisX = toIndex.normalized;                         // toward index MCP
        Vector3 palmNormal = Vector3.Cross(toIndex, across);
        if (palmNormal.sqrMagnitude < 1e-8f)
            return;
        axisZ = palmNormal.normalized;                      // palm normal
        axisY = Vector3.Cross(axisZ, axisX).normalized;     // orthogonalized
        axisX = Vector3.Cross(axisY, axisZ).normalized;     // re-orthogonalize
    }

    // ==========================================
    // ARM IK (two-bone, wrist trajectory -> signing box)
    // ==========================================

    void SetupArms()
    {
        Vector3 chest = Vector3.zero;
        bool haveChest = false;
        // Chest/Spine/Neck are not registered in SetupBones; resolve them
        // directly from the Animator. A missing chest disables arms only
        // (hands keep working).
        foreach (HumanBodyBones b in new[] {
            HumanBodyBones.Chest, HumanBodyBones.Spine, HumanBodyBones.Neck })
        {
            try
            {
                Transform direct = animator.GetBoneTransform(b);
                if (direct != null)
                {
                    chest = direct.position;
                    haveChest = true;
                    break;
                }
            }
            catch { /* ignore, try next */ }
        }
        if (!haveChest)
        {
            Debug.LogWarning("HandPoseMapper: no chest/spine/neck bone; arm IK disabled.");
            return;
        }

        // Torso capsule endpoints for collision (wrist/elbow clamp, web
        // parity). Static rest pose: the torso never animates here.
        try
        {
            Transform hipsT = animator.GetBoneTransform(HumanBodyBones.Hips);
            Transform neckT = animator.GetBoneTransform(HumanBodyBones.Neck)
                ?? animator.GetBoneTransform(HumanBodyBones.Head);
            if (hipsT != null && neckT != null)
            {
                torsoHips = hipsT.position;
                torsoNeck = neckT.position;
                torsoRadius = Mathf.Max(0.1f,
                    Vector3.Distance(torsoHips, torsoNeck) * 0.24f);
                haveTorsoCapsule = true;
            }
        }
        catch { /* arms still work without collision */ }

        // Measure the real chest surface + hand volume from the skinned
        // meshes (web parity). Must run while the rig is in its rest pose,
        // which is where setup runs.
        MeasureTorso();

        leftArm = BuildArmRig(true, chest);
        rightArm = BuildArmRig(false, chest);
        if (leftArm == null && rightArm == null)
            Debug.LogWarning("HandPoseMapper: no arm bones found; arm IK disabled.");
        else
            BuildIdlePose();
    }

    /// <summary>
    /// Solve the arms-down idle once, through the same two-bone IK the signs
    /// use, so it obeys the capsule / chest clamps and the reach limit instead
    /// of fighting them (web parity). Leaves the avatar in that pose at rest.
    /// </summary>
    void BuildIdlePose()
    {
        foreach (ArmRig rig in new[] { leftArm, rightArm })
        {
            if (rig == null)
                continue;
            float len = rig.lenU + rig.lenL;
            float s = rig.isLeft ? 1f : -1f;
            Vector3 target = new Vector3(
                rig.shoulder.x + s * IDLE_LATERAL * len,
                rig.shoulder.y - IDLE_DROP * len,
                rig.shoulder.z + IDLE_FWD * len);
            SolveArmAt(rig, target, instant: true);
            rig.idleUpper = rig.upper.localRotation;
            rig.idleLower = rig.lower.localRotation;
            rig.hasIdle = true;
        }
    }

    /// <summary>
    /// Ease an arm with no landmarks this frame back to the idle pose instead
    /// of freezing wherever the previous clip left it (web parity).
    /// </summary>
    void RelaxArm(ArmRig rig)
    {
        if (rig == null || !rig.hasIdle)
            return;
        rig.upper.localRotation = Smooth(rig.upper.localRotation, rig.idleUpper);
        rig.lower.localRotation = Smooth(rig.lower.localRotation, rig.idleLower);
        if (rig.hand != null)
            rig.hand.localRotation = Smooth(rig.hand.localRotation, rig.handRestLocal, 0.6f);
    }

    ArmRig BuildArmRig(bool isLeft, Vector3 chest)
    {
        HumanBodyBones upperBone = isLeft ? HumanBodyBones.LeftUpperArm : HumanBodyBones.RightUpperArm;
        HumanBodyBones lowerBone = isLeft ? HumanBodyBones.LeftLowerArm : HumanBodyBones.RightLowerArm;
        HumanBodyBones handBone = isLeft ? HumanBodyBones.LeftHand : HumanBodyBones.RightHand;
        if (!boneMap.TryGetValue(upperBone, out Transform upper) || upper == null)
            return null;
        if (!boneMap.TryGetValue(lowerBone, out Transform lower) || lower == null)
            return null;
        if (!boneMap.TryGetValue(handBone, out Transform hand) || hand == null)
            return null;

        Vector3 S = upper.position;
        Vector3 E = lower.position;
        Vector3 W = hand.position;
        float lenU = Vector3.Distance(S, E);
        float lenL = Vector3.Distance(E, W);
        if (lenU < 1e-6f || lenL < 1e-6f)
            return null;

        Vector3 restDirU = (E - S).normalized;
        Vector3 restDirL = (W - E).normalized;
        Quaternion restWQU = upper.rotation;
        Quaternion restWQL = lower.rotation;

        // Elbow hint: outward from chest, down, slightly FORWARD. A backward-biased
        // pole parks elbows behind the torso plane, where the torso mesh
        // swallows the upper arms (arms must read in front).
        Vector3 outward = S - chest;
        outward.y = 0f;
        if (outward.sqrMagnitude < 1e-8f)
            outward = new Vector3(isLeft ? 1f : -1f, 0f, 0f);
        outward.Normalize();
        Vector3 pole = (outward * 0.45f + new Vector3(0f, -1f, 0.2f)).normalized;

        return new ArmRig
        {
            upper = upper, lower = lower, hand = hand,
            isLeft = isLeft,
            shoulder = S, lenU = lenU, lenL = lenL,
            restWQU = restWQU, restWQL = restWQL,
            restDirU = restDirU, restDirL = restDirL,
            pole = pole, chest = chest,
            handRestLocal = hand.localRotation,
        };
    }

    /// <summary>
    /// Carry one arm's wrist into the signing box. wristRaw uses the raw
    /// landmark coords (origin top-left, z toward viewer), normalized
    /// through the clip calibration first (same convention as web).
    /// </summary>
    void SolveArm(ArmRig rig, Vector3 wristRaw)
    {
        if (rig == null || !enableArmIK)
            return;
        try
        {
            Vector3 C = rig.chest;
            float s = Mathf.Max(signingScale, 1e-3f);
            Vector3 w = NormalizeWrist(wristRaw);
            // Lateral and vertical: fixed scale from the source centre, NOT
            // range-normalised. Range-normalising by the clip's own span maps
            // whatever the source happened to span onto the avatar's full
            // signing box, which destroyed gestures whose hands are close
            // together (namaste source hands are ~0.14 frame-widths apart, yet
            // landed ~0.55 m apart) and amplified a 0.17 frame-height source
            // range into 0.7 m of avatar travel, dropping the wrists below the
            // shoulders where the arm cannot reach them. A fixed scale
            // preserves the source's relative hand distance and height.
            // `claspAdj*` is the contact assist's lateral squeeze.
            Vector3 target = new Vector3(
                C.x + (wristRaw.x - calibCenter.x) * LATERAL_SCALE * s
                    + (rig.isLeft ? claspAdjLeft : claspAdjRight),
                C.y + (calibCenter.y - wristRaw.y) * HEIGHT_SCALE * s,
                Mathf.Max(C.z + 0.3f * s - w.z * 0.4f * s, C.z + 0.06f * s));
            SolveArmAt(rig, target, instant: false);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"HandPoseMapper: arm IK skipped ({e.Message}).");
        }
    }

    /// <summary>
    /// Two-bone IK from an explicit world wrist target (web parity with
    /// applyArmAt). `instant` skips the smoothing; only the idle solve at
    /// setup uses it, so the captured idle is the converged pose.
    /// </summary>
    void SolveArmAt(ArmRig rig, Vector3 target, bool instant)
    {
        if (rig == null || !enableArmIK)
            return;
        try
        {
            Vector3 C = rig.chest;
            Vector3 toT = target - rig.shoulder;
            float dist = toT.magnitude;
            float maxReach = rig.lenU + rig.lenL - 0.02f;
            if (dist > maxReach && dist > 1e-6f)
            {
                dist = maxReach;
                target = rig.shoulder + toT.normalized * dist;
                toT = target - rig.shoulder;
            }
            if (dist < 1e-6f)
                return;
            // Torso collision (web parity): keep the wrist target outside
            // the chest capsule (+3cm standoff) before solving.
            ClampOutsideCapsule(ref target, 0.03f);
            // Chest-surface guard (web parity): the capsule above constrains
            // the wrist POINT with a radius this avatar's bust exceeds, so it
            // is applied AFTER the reach clamp (pulling the arm back in must
            // not undo it) and it constrains the whole HAND -- a 16 cm hand is
            // not a point. Gated laterally so wide poses are never shoved
            // forward.
            float surfFront, surfHalfX, surfAxisX;
            if (TorsoSurfaceAt(target.y, out surfFront, out surfHalfX, out surfAxisX)
                && Mathf.Abs(target.x - surfAxisX) < surfHalfX + 0.02f)
            {
                float back = HandBackExtent(rig.isLeft);
                float need = surfFront + HAND_CLEARANCE + back;
                if (target.z < need)
                    target.z = need;
            }
            // `toT` feeds the pole and `dist` feeds the shoulder angle, so both
            // must be refreshed here: a stale `dist` aims the upper arm for the
            // old target and the wrist lands short of the new one, which is
            // exactly what put the hand back into the chest.
            toT = target - rig.shoulder;
            dist = toT.magnitude;
            if (dist < 1e-6f)
                return;
            // Record the FINAL target (after the reach clamp and the chest
            // guard, as the web does): the diagnostics compare the solved wrist
            // against this, and the palm-basis hold tags itself with it, so a
            // pre-guard value reads as a phantom 9 cm wrist error.
            if (rig.isLeft)
                lastTargetLeft = target;
            else
                lastTargetRight = target;
            Vector3 dirST = toT / dist;

            float cosA = Mathf.Clamp(
                (rig.lenU * rig.lenU + dist * dist - rig.lenL * rig.lenL)
                / (2f * rig.lenU * dist), -1f, 1f);
            float angA = Mathf.Acos(cosA);
            // Adaptive pole (web parity): bias the elbow downward as the
            // target crosses the midline, so the elbow folds naturally
            // instead of flaring or flipping. Namaste brings both hands to
            // the centre, exactly the case that exposed the fixed pole.
            float sideSign = rig.isLeft ? 1f : -1f;
            float cross = Mathf.Clamp01(-sideSign * (target.x - C.x) / 0.3f);
            Vector3 pole = (rig.pole + Vector3.down * (cross * 0.8f)).normalized;
            Vector3 perp = pole - dirST * Vector3.Dot(pole, dirST);
            if (perp.sqrMagnitude < 1e-8f)
                perp = Vector3.down;
            perp.Normalize();
            Vector3 upperDir = (dirST * Mathf.Cos(angA) + perp * Mathf.Sin(angA)).normalized;
            // Forward guard (web parity): relaxed hanging arms (z≈0) pass
            // through untouched; only true backward aims are fixed.
            if (upperDir.z < -0.03f)
            {
                upperDir.z = -0.03f;
                upperDir.Normalize();
            }
            Vector3 elbowPos = rig.shoulder + upperDir * rig.lenU;
            // Elbow collision: keep the joint out of the torso too.
            ClampOutsideCapsule(ref elbowPos, 0f);
            Vector3 foreDir = (target - elbowPos).normalized;
            // Elbow hinge clamp, max 150 deg (web parity): hyperextended
            // elbows read as broken.
            float flexAng = Vector3.Angle(upperDir, foreDir);
            if (flexAng > 150f)
            {
                Vector3 hinge = Vector3.Cross(upperDir, foreDir);
                if (hinge.sqrMagnitude > 1e-10f)
                    foreDir = Quaternion.AngleAxis(150f - flexAng, hinge.normalized) * foreDir;
            }

            // ---- upper arm: twist-locked swing (web parity) ----
            // Build rest/target frames around the pole-side axis so
            // upper-arm roll is determined, not free. Falls back to the
            // minimal swing when the pole side degenerates. The parent may
            // have moved this frame; Unity refreshes a Transform's world data
            // when it is read, so the rotation below is already current.
            Quaternion parentQU = rig.upper.parent != null
                ? rig.upper.parent.rotation : Quaternion.identity;
            Vector3 restPoleSide = rig.pole - rig.restDirU * Vector3.Dot(rig.pole, rig.restDirU);
            Vector3 poleSide = pole - upperDir * Vector3.Dot(pole, upperDir);
            Quaternion qU;
            if (restPoleSide.sqrMagnitude > 1e-8f && poleSide.sqrMagnitude > 1e-8f)
            {
                restPoleSide.Normalize();
                poleSide.Normalize();
                Vector3 restSide2 = Vector3.Cross(rig.restDirU, restPoleSide).normalized;
                Vector3 targSide2 = Vector3.Cross(upperDir, poleSide).normalized;
                Quaternion qRest = QuatFromBasis(rig.restDirU, restPoleSide, restSide2);
                Quaternion qTarg = QuatFromBasis(upperDir, poleSide, targSide2);
                qU = qTarg * Quaternion.Inverse(qRest) * rig.restWQU;
            }
            else
            {
                qU = Quaternion.FromToRotation(rig.restDirU, upperDir) * rig.restWQU;
            }
            rig.upper.localRotation = instant
                ? Quaternion.Inverse(parentQU) * qU
                : Smooth(rig.upper.localRotation, Quaternion.Inverse(parentQU) * qU);

            // ---- forearm ----  (its parent IS the upper arm just posed;
            // Unity refreshes world data on read, so the rotation below
            // already includes this frame's upper-arm change).
            Quaternion parentQL = rig.lower.parent != null
                ? rig.lower.parent.rotation : Quaternion.identity;
            Quaternion qL = Quaternion.FromToRotation(rig.restDirL, foreDir) * rig.restWQL;
            rig.lower.localRotation = instant
                ? Quaternion.Inverse(parentQL) * qL
                : Smooth(rig.lower.localRotation, Quaternion.Inverse(parentQL) * qL);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"HandPoseMapper: arm IK skipped ({e.Message}).");
        }
    }

    /// <summary>
    /// Build a rotation whose local X / Y / Z axes are the given (world)
    /// vectors. Web parity with quatFromBasis / Matrix4.makeBasis, used by
    /// the twist-locked upper-arm swing.
    /// </summary>
    static Quaternion QuatFromBasis(Vector3 x, Vector3 y, Vector3 z)
    {
        Matrix4x4 m = new Matrix4x4();
        m.SetColumn(0, new Vector4(x.x, x.y, x.z, 0f));
        m.SetColumn(1, new Vector4(y.x, y.y, y.z, 0f));
        m.SetColumn(2, new Vector4(z.x, z.y, z.z, 0f));
        m.SetColumn(3, new Vector4(0f, 0f, 0f, 1f));
        return m.rotation;
    }

    // ==========================================
    // ANIMATE HAND (palm-frame retargeting)
    // ==========================================

    void AnimateHand(HandData hand, bool isLeft)
    {
        fingerHandDelta = Quaternion.identity;
        // Arms first so the hands ride up into signing space (web parity).
        // Uses raw landmark coords, same convention as the web solver.
        if (hand.landmarks != null && hand.landmarks.Count > 0)
        {
            LandmarkPoint wrist = hand.landmarks[0];
            SolveArm(isLeft ? leftArm : rightArm,
                new Vector3(wrist.x, wrist.y, wrist.z));
        }

        // Landmark palm basis: wrist(0) -> index MCP(5), plus the knuckle line
        // index MCP(5) -> little MCP(17). wrist -> index-MCP and wrist ->
        // middle-MCP(9) are only a few degrees apart on a flat hand, so their
        // cross product was noise and the wrist roll came out random every
        // frame (web parity fix).
        Vector3 lmOrigin = ToWorld(hand.landmarks[0]);
        Vector3 lmToIndex = ToWorld(hand.landmarks[5]) - lmOrigin;
        Vector3 lmAcross = hand.landmarks.Count > 17
            ? ToWorld(hand.landmarks[17]) - ToWorld(hand.landmarks[5])
            : Vector3.zero;

        Vector3 lmX = Vector3.zero, lmY = Vector3.zero, lmZ = Vector3.zero;
        bool haveBasis = false;
        if (lmToIndex.sqrMagnitude > 1e-10f && lmAcross.sqrMagnitude > 1e-10f)
        {
            Vector3 rawZ = Vector3.Cross(lmToIndex, lmAcross);
            if (rawZ.sqrMagnitude > 1e-10f)
            {
                lmX = lmToIndex.normalized;
                lmZ = rawZ.normalized;
                lmY = Vector3.Cross(lmZ, lmX).normalized;
                lmX = Vector3.Cross(lmY, lmZ).normalized;
                haveBasis = true;
                // Remember the last good basis, tagged with the arm target
                // it was seen at (degenerate-frame hold below).
                Vector3 at = isLeft ? lastTargetLeft : lastTargetRight;
                if (isLeft)
                {
                    lastPalmLeftX = lmX; lastPalmLeftY = lmY; lastPalmLeftZ = lmZ;
                    lastPalmLeftAt = at; haveLastPalmLeft = true;
                }
                else
                {
                    lastPalmRightX = lmX; lastPalmRightY = lmY; lastPalmRightZ = lmZ;
                    lastPalmRightAt = at; haveLastPalmRight = true;
                }
            }
        }

        if (!haveBasis)
        {
            // Degenerate frame (collinear finger roots): keep the LAST good
            // palm frame instead of freezing the wrist while the arm keeps
            // moving (a detached/flailing look) - but only while the arm is
            // still near where that basis was captured. A stale basis on a
            // traveled arm kinks the wrist sharply, so skip the frame then
            // (web parity).
            Vector3 at = isLeft ? lastTargetLeft : lastTargetRight;
            if (isLeft && haveLastPalmLeft && Vector3.Distance(lastPalmLeftAt, at) <= 0.15f)
            {
                lmX = lastPalmLeftX; lmY = lastPalmLeftY; lmZ = lastPalmLeftZ;
                haveBasis = true;
            }
            else if (!isLeft && haveLastPalmRight && Vector3.Distance(lastPalmRightAt, at) <= 0.15f)
            {
                lmX = lastPalmRightX; lmY = lastPalmRightY; lmZ = lastPalmRightZ;
                haveBasis = true;
            }
            else
            {
                return;
            }
        }

        // Clasp palm roll (web parity): a landmark palm basis can only ever
        // say "flat to the camera", so a palms-together pose leaves both palms
        // pointing at the viewer and the wrists effectively un-pronated. Force
        // the palm normal toward the other hand while the pair is clasped.
        if (clasping)
        {
            // World image of the source's +x axis. The basis z is the
            // BACK-of-hand normal on the left and the PALM normal on the
            // right, so this single target direction turns both palms inward.
            Vector3 lmWorldX = ToWorld(new LandmarkPoint { x = 1f, y = 0f, z = 0f });
            Vector3 forced = lmWorldX - lmX * Vector3.Dot(lmWorldX, lmX);
            if (forced.sqrMagnitude > 0.05f)
            {
                lmZ = forced.normalized;
                lmY = Vector3.Cross(lmZ, lmX).normalized;
            }
        }

        // Avatar rest palm basis (built once in Start).
        Vector3 avOrigin, avX, avY, avZ;
        if (isLeft)
        {
            avOrigin = avatarLeftOrigin; avX = avatarLeftAxisX;
            avY = avatarLeftAxisY; avZ = avatarLeftAxisZ;
        }
        else
        {
            avOrigin = avatarRightOrigin; avX = avatarRightAxisX;
            avY = avatarRightAxisY; avZ = avatarRightAxisZ;
        }

        // Rotation that takes the avatar rest palm frame to the landmark palm frame.
        Quaternion avatarToLandmark = Quaternion.LookRotation(avZ, avY);
        Quaternion landmarkFrame = Quaternion.LookRotation(lmZ, lmY);
        Quaternion palmDelta = landmarkFrame * Quaternion.Inverse(avatarToLandmark);

        // Rotate hand root with the palm delta (absolute, non-accumulating),
        // capped per-frame so tracker basis flips read as eased turns
        // instead of wrist snaps (web parity: ~7 rad/s cap).
        HumanBodyBones handBone = isLeft ? HumanBodyBones.LeftHand : HumanBodyBones.RightHand;
        if (boneMap.TryGetValue(handBone, out Transform handTransform) &&
            initialWorldRotations.TryGetValue(handBone, out Quaternion handRestWorld))
        {
            // palmDelta is a WORLD-space rotation. Rotate the hand's rest
            // WORLD orientation, then express the result in the CURRENT
            // parent (lower-arm) space. Applying palmDelta to the local
            // rotation - as before - silently conjugated the wrist by the
            // arm's rotation and, with the arm posed by IK, left the palm
            // pointing the wrong way (hands could not face each other).
            // Unity refreshes world data on read, so this parent rotation
            // already has this frame's arm pose in it.
            Quaternion parentWorld = handTransform.parent != null
                ? handTransform.parent.rotation : Quaternion.identity;
            Quaternion target = Quaternion.Inverse(parentWorld) * (palmDelta * handRestWorld);
            if (isLeft)
            {
                if (!haveLastWristQLeft) { lastWristQLeft = target; haveLastWristQLeft = true; }
                else
                {
                    // Quaternion.Angle / RotateTowards work in DEGREES, so the
                    // 7 rad/s budget must be converted (it previously clamped
                    // to ~0.12 deg/frame - the wrist barely turned at all).
                    float maxStepDeg = 7f * Mathf.Max(Time.deltaTime, 1e-3f) * Mathf.Rad2Deg;
                    float step = Quaternion.Angle(lastWristQLeft, target);
                    if (step > maxStepDeg)
                        target = Quaternion.RotateTowards(lastWristQLeft, target, maxStepDeg);
                    lastWristQLeft = target;
                }
            }
            else
            {
                if (!haveLastWristQRight) { lastWristQRight = target; haveLastWristQRight = true; }
                else
                {
                    float maxStepDeg = 7f * Mathf.Max(Time.deltaTime, 1e-3f) * Mathf.Rad2Deg;
                    float step = Quaternion.Angle(lastWristQRight, target);
                    if (step > maxStepDeg)
                        target = Quaternion.RotateTowards(lastWristQRight, target, maxStepDeg);
                    lastWristQRight = target;
                }
            }
            handTransform.localRotation = Smooth(handTransform.localRotation, target);

            // The hand's ACTUAL world rotation change this frame. Finger
            // bones are aimed with the REST-pose avatar basis (avX/Y/Z),
            // which lives in the rest world frame, so without carrying this
            // delta across the fingers do not follow the wrist at all: a
            // rolled wrist left them hanging ~90 degrees off the new palm
            // axis (web parity fix).
            fingerHandDelta = handTransform.rotation * Quaternion.Inverse(handRestWorld);
        }

        // Fingers: express each landmark segment direction in the landmark palm
        // frame, map into the avatar palm frame, then FromTo against rest.
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexProximal : HumanBodyBones.RightIndexProximal, 5, 6, hand.landmarks[5], hand.landmarks[6], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexIntermediate : HumanBodyBones.RightIndexIntermediate, 6, 7, hand.landmarks[6], hand.landmarks[7], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexDistal : HumanBodyBones.RightIndexDistal, 7, 8, hand.landmarks[7], hand.landmarks[8], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleProximal : HumanBodyBones.RightMiddleProximal, 9, 10, hand.landmarks[9], hand.landmarks[10], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleIntermediate : HumanBodyBones.RightMiddleIntermediate, 10, 11, hand.landmarks[10], hand.landmarks[11], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleDistal : HumanBodyBones.RightMiddleDistal, 11, 12, hand.landmarks[11], hand.landmarks[12], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingProximal : HumanBodyBones.RightRingProximal, 13, 14, hand.landmarks[13], hand.landmarks[14], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingIntermediate : HumanBodyBones.RightRingIntermediate, 14, 15, hand.landmarks[14], hand.landmarks[15], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingDistal : HumanBodyBones.RightRingDistal, 15, 16, hand.landmarks[15], hand.landmarks[16], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleProximal : HumanBodyBones.RightLittleProximal, 17, 18, hand.landmarks[17], hand.landmarks[18], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleIntermediate : HumanBodyBones.RightLittleIntermediate, 18, 19, hand.landmarks[18], hand.landmarks[19], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleDistal : HumanBodyBones.RightLittleDistal, 19, 20, hand.landmarks[19], hand.landmarks[20], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbProximal : HumanBodyBones.RightThumbProximal, 1, 2, hand.landmarks[1], hand.landmarks[2], lmX, lmY, lmZ, avX, avY, avZ, 0.6f, 90f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbIntermediate : HumanBodyBones.RightThumbIntermediate, 2, 3, hand.landmarks[2], hand.landmarks[3], lmX, lmY, lmZ, avX, avY, avZ, 0.8f, 90f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbDistal : HumanBodyBones.RightThumbDistal, 3, 4, hand.landmarks[3], hand.landmarks[4], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 90f);
    }

    void RotateFingerBone(
        HumanBodyBones bone,
        int startIdx, int endIdx,
        LandmarkPoint start, LandmarkPoint end,
        Vector3 lmX, Vector3 lmY, Vector3 lmZ,
        Vector3 avX, Vector3 avY, Vector3 avZ,
        float damp = 1f,
        float maxFlexDeg = 110f)
    {
        if (!boneMap.TryGetValue(bone, out Transform boneTransform) || boneTransform == null)
            return;
        if (!restLocalDirections.TryGetValue(bone, out Vector3 restLocal) ||
            !initialRotations.TryGetValue(bone, out Quaternion restRot))
            return;
        if (boneTransform.parent == null)
            return;

        Vector3 worldStart = ToWorld(start);
        Vector3 worldEnd = ToWorld(end);
        Vector3 seg = worldEnd - worldStart;
        if (seg.sqrMagnitude < 1e-12f)
            return;

        // Segment direction in landmark palm frame.
        Vector3 inLm = new Vector3(Vector3.Dot(seg, lmX), Vector3.Dot(seg, lmY), Vector3.Dot(seg, lmZ)).normalized;

        // Map into avatar palm frame (world space for this frame's basis),
        // then carry the hand's own rotation change so the fingers ride with
        // the posed wrist instead of staying in the rest world frame.
        Vector3 targetWorld = fingerHandDelta *
            (avX * inLm.x + avY * inLm.y + avZ * inLm.z);

        // Into parent space, then FromTo against rest direction, clamped
        // to the anatomical flexion limit (web parity: no folded-back
        // fingers no matter what the tracker reports). The parent's rotation
        // was just written this frame; Unity refreshes world data when it is
        // read, so the inverse below already uses the fresh parent.
        Vector3 targetLocal = boneTransform.parent.InverseTransformDirection(targetWorld).normalized;
        if (targetLocal.sqrMagnitude < 1e-8f)
            return;

        Quaternion target;
        float flexAng = Vector3.Angle(restLocal, targetLocal);
        bool didClamp = false;
        if (flexAng > maxFlexDeg)
        {
            Vector3 hinge = Vector3.Cross(restLocal, targetLocal);
            if (hinge.sqrMagnitude > 1e-10f)
                targetLocal = Quaternion.AngleAxis(maxFlexDeg, hinge.normalized) * restLocal;
            else
                targetLocal = restLocal;
            didClamp = true;
        }
        Quaternion delta = Quaternion.FromToRotation(restLocal, targetLocal);
        target = delta * restRot;
        boneTransform.localRotation = Smooth(boneTransform.localRotation, target, damp);

        if (captureDiagnostics)
        {
            lastDiagnostics[bone] = new BoneDriveDiag
            {
                bone = bone,
                startIdx = startIdx,
                endIdx = endIdx,
                segWorld = seg.normalized,
                restLocal = restLocal,
                targetLocal = targetLocal,
                flexAngPreClamp = flexAng,
                maxFlexDeg = maxFlexDeg,
                clamped = didClamp,
                damp = damp,
                appliedDeg = Quaternion.Angle(boneTransform.localRotation, restRot),
            };
        }
    }

    Quaternion Smooth(Quaternion current, Quaternion target, float damp = 1f)
    {
        float t = Mathf.Min(1f, (1f - Mathf.Exp(-rotationSmoothness * Time.deltaTime)) * damp);
        return Quaternion.Slerp(current, target, t);
    }

    /// <summary>
    /// Push a point outside the torso capsule (hips-&gt;neck axis). Returns
    /// true when pushed. Web parity with clampOutsideCapsule.
    /// </summary>
    bool ClampOutsideCapsule(ref Vector3 point, float standoff)
    {
        if (!haveTorsoCapsule) return false;
        Vector3 axis = torsoNeck - torsoHips;
        float lenSq = axis.sqrMagnitude;
        if (lenSq < 1e-8f) return false;
        // Only constrain points that are actually behind the torso's front
        // surface. A wrist in front of the chest (namaste palms together,
        // fingers interlaced) is NOT inside the body, and pushing it out to
        // the capsule radius forces the two hands apart.
        if (point.z > torsoNeck.z - 0.05f) return false;
        float t = Mathf.Clamp01(Vector3.Dot(point - torsoHips, axis) / lenSq);
        Vector3 center = torsoHips + axis * t;
        Vector3 radial = point - center;
        float d = radial.magnitude;
        if (d >= torsoRadius + standoff || d < 1e-9f) return false;
        point = center + radial.normalized * (torsoRadius + standoff);
        return true;
    }

    // ==========================================
    // MEASURED CHEST SURFACE (web parity)
    // ==========================================

    /// <summary>
    /// One pass over the skinned meshes to measure the torso's forward
    /// surface per height band, its lateral half-width (arms excluded), the
    /// palm vertices each hand bone owns, and the hand's thickness. Runs in
    /// the rest pose, so every vertex is read where the rig actually is.
    /// </summary>
    void MeasureTorso()
    {
        try
        {
            Transform hipsT = animator != null
                ? animator.GetBoneTransform(HumanBodyBones.Hips) : null;
            Transform neckT = animator != null
                ? (animator.GetBoneTransform(HumanBodyBones.Neck)
                    ?? animator.GetBoneTransform(HumanBodyBones.Head)) : null;
            if (hipsT == null || neckT == null) return;
            Vector3 H = hipsT.position, N = neckT.position;
            float axisLen = N.y - H.y;
            if (Mathf.Abs(axisLen) < 1e-3f) return;

            Transform handL = animator.GetBoneTransform(HumanBodyBones.LeftHand);
            Transform handR = animator.GetBoneTransform(HumanBodyBones.RightHand);
            handBoneXf[0] = handL;
            handBoneXf[1] = handR;
            for (int s = 0; s < 2; s++)
            {
                Transform hand = handBoneXf[s];
                if (hand == null) { handJoints[s] = new Transform[0]; continue; }
                // Hand + every finger/thumb joint below it: the mesh moves
                // with these, so their world positions bound the palm.
                var joints = new List<Transform>();
                foreach (Transform t in hand.GetComponentsInChildren<Transform>(true))
                    joints.Add(t);
                handJoints[s] = joints.ToArray();
                handPts[s] = new Vector3[0];
            }
            var handPoints = new List<Vector3>[2];
            handPoints[0] = new List<Vector3>();
            handPoints[1] = new List<Vector3>();

            // Torso bones own the torso surface; secondary (bust / clothing)
            // bones do not, which is why the front depth is ALSO taken
            // geometrically from a midline slab -- a bone filter alone leaves
            // holes where the clothing is skinned to J_Sec_* bones.
            var torsoBones = new HashSet<Transform>();
            foreach (HumanBodyBones b in new[] { HumanBodyBones.Hips,
                HumanBodyBones.Spine, HumanBodyBones.Chest, HumanBodyBones.UpperChest })
            {
                Transform t = animator.GetBoneTransform(b);
                if (t != null) torsoBones.Add(t);
            }

            int bands = TORSO_PROFILE_BANDS;
            float[] frontMid = new float[bands];
            float[] frontBone = new float[bands];
            float[] half = new float[bands];
            for (int i = 0; i < bands; i++)
            {
                frontMid[i] = float.NegativeInfinity;
                frontBone[i] = float.NegativeInfinity;
                half[i] = 0f;
            }

            var smrs = GetComponentsInChildren<SkinnedMeshRenderer>(true);
            if (smrs != null)
            {
                for (int r = 0; r < smrs.Length; r++)
                {
                    SkinnedMeshRenderer smr = smrs[r];
                    if (smr == null) continue;
                    Mesh mesh = smr.sharedMesh;
                    if (mesh == null) continue;
                    Vector3[] verts = mesh.vertices;
                    if (verts == null || verts.Length == 0) continue;
                    BoneWeight[] bw = mesh.boneWeights;
                    Matrix4x4[] bind = mesh.bindposes;
                    Transform[] bones = smr.bones;
                    bool skinned = bw != null && bw.Length == verts.Length
                        && bind != null && bones != null;

                    for (int i = 0; i < verts.Length; i++)
                    {
                        Transform dom = null;
                        Vector3 world;
                        if (skinned)
                        {
                            int bi = bw[i].weight0 > 0f ? bw[i].boneIndex0 : bw[i].boneIndex1;
                            if (bi >= 0 && bi < bones.Length && bi < bind.Length)
                            {
                                dom = bones[bi];
                                world = dom.localToWorldMatrix
                                    .MultiplyPoint3x4(bind[bi].MultiplyPoint3x4(verts[i]));
                            }
                            else
                            {
                                world = smr.transform.TransformPoint(verts[i]);
                            }
                        }
                        else
                        {
                            world = smr.transform.TransformPoint(verts[i]);
                        }

                        float t = (world.y - H.y) / axisLen;
                        if (t < 0f || t > 1f)
                        {
                            // Only the hand samples live outside the torso's
                            // vertical span; keep collecting those below.
                            t = Mathf.Clamp01(t);
                        }
                        int band = Mathf.Clamp(Mathf.FloorToInt(t * (bands - 1)), 0, bands - 1);
                        float axisX = Mathf.Lerp(H.x, N.x, t);
                        float u = Mathf.Abs(world.x - axisX);

                        bool inSpan = world.y >= H.y - 0.02f && world.y <= N.y + 0.02f;
                        if (inSpan)
                        {
                            if (u < TORSO_MIDLINE_HALF && world.z > frontMid[band])
                                frontMid[band] = world.z;
                            if (dom != null && torsoBones.Contains(dom))
                            {
                                if (world.z > frontBone[band]) frontBone[band] = world.z;
                                if (u > half[band]) half[band] = u;
                            }
                        }

                        if (dom == null) continue;
                        for (int s = 0; s < 2; s++)
                        {
                            Transform hand = handBoneXf[s];
                            if (hand == null || dom != hand) continue;
                            if (handPoints[s].Count >= HAND_POINT_MAX) break;
                            // Store the vertex in the hand bone's own frame:
                            // it is rigid to that bone, so a plain
                            // TransformPoint reproduces the posed surface on
                            // any later frame.
                            handPoints[s].Add(hand.InverseTransformPoint(world));
                        }
                    }
                }
            }

            // Fall back to the bone-filtered depth where the slab saw nothing
            // (a band whose midline is a hole) and carry the last good
            // half-width forward past a zero band.
            float lastHalf = 0f;
            for (int i = 0; i < bands; i++)
            {
                if (float.IsNegativeInfinity(frontMid[i]))
                    frontMid[i] = float.IsNegativeInfinity(frontBone[i]) ? 0f : frontBone[i];
                if (half[i] <= 0f) half[i] = lastHalf; else lastHalf = half[i];
            }
            // Fill any band that produced nothing at all from its neighbours,
            // so the interpolated query never reads a zero through the chest.
            for (int i = 0; i < bands; i++)
            {
                if (frontMid[i] > 0f) continue;
                float nz = 0f;
                for (int j = i - 1; j >= 0; j--) if (frontMid[j] > 0f) { nz = frontMid[j]; break; }
                for (int j = i + 1; j < bands && nz <= 0f; j++) if (frontMid[j] > 0f) nz = frontMid[j];
                frontMid[i] = nz;
            }

            torsoFrontZ = frontMid;
            torsoHalfX = half;
            torsoHipsY = H.y;
            torsoHipsX = H.x;
            torsoNeckX = N.x;
            torsoAxisLen = axisLen;
            haveTorsoProfile = torsoAxisLen > 1e-3f;

            for (int s = 0; s < 2; s++)
            {
                handPts[s] = handPoints[s].ToArray();
                // Smallest full extent of the palm box = the surface gap the
                // palms need to touch without intersecting.
                if (handPts[s].Length > 8)
                {
                    Vector3 lo = handPts[s][0], hi = handPts[s][0];
                    for (int k = 1; k < handPts[s].Length; k++)
                    {
                        lo = Vector3.Min(lo, handPts[s][k]);
                        hi = Vector3.Max(hi, handPts[s][k]);
                    }
                    Vector3 e = hi - lo;
                    // `e` is the palm's FULL extent in the hand bone's frame,
                    // so its smallest dimension is already the thickness; the
                    // web's box was expressed as half-extents and doubling
                    // here made the clasp floor 9 cm, which stopped the palms
                    // a hand's width apart before they ever met.
                    float thick = Mathf.Min(e.x, Mathf.Min(e.y, e.z));
                    if (thick > 0f) handMinSep = Mathf.Max(handMinSep, thick + HAND_CLEARANCE);
                }
            }

            Debug.Log(string.Format(
                "HandPoseMapper: torso profile measured ({0} bands, hips y {1:F3}, axis {2:F3} m, "
                + "bust front {3:F3} m, half width {4:F3} m); hand pts L/R {5}/{6}, min sep {7:F4} m.",
                bands, torsoHipsY, torsoAxisLen, MaxOf(torsoFrontZ), MaxOf(torsoHalfX),
                handPts[0].Length, handPts[1].Length, handMinSep));
        }
        catch (Exception e)
        {
            haveTorsoProfile = false;
            Debug.LogWarning("HandPoseMapper: torso profile measurement skipped ("
                + e.Message + ").");
        }
    }

    static float MaxOf(float[] a)
    {
        if (a == null) return 0f;
        float m = float.NegativeInfinity;
        for (int i = 0; i < a.Length; i++) if (a[i] > m) m = a[i];
        return float.IsNegativeInfinity(m) ? 0f : m;
    }

    /// <summary>Torso surface at one exact height (no window).</summary>
    bool BandSurface(float y, out float front, out float halfX, out float axisX)
    {
        front = 0f; halfX = 0f; axisX = 0f;
        if (!haveTorsoProfile || torsoFrontZ == null) return false;
        float t = (y - torsoHipsY) / torsoAxisLen;
        if (t < -0.05f || t > 1.05f) return false;
        t = Mathf.Clamp01(t);
        int n = torsoFrontZ.Length;
        float f = t * (n - 1);
        int b = Mathf.Clamp(Mathf.FloorToInt(f), 0, n - 2);
        float k = f - b;
        front = torsoFrontZ[b] * (1f - k) + torsoFrontZ[b + 1] * k;
        halfX = torsoHalfX[b] * (1f - k) + torsoHalfX[b + 1] * k;
        axisX = Mathf.Lerp(torsoHipsX, torsoNeckX, t);
        return true;
    }

    /// <summary>
    /// Torso surface for a hand at height <paramref name="y"/>. Takes the max
    /// front over the hand's whole vertical span so a single 2 cm band dip
    /// (the neckline between the bust and the collar) cannot undo the guard,
    /// then hands back the LATERAL gate the caller uses to leave wide poses
    /// alone.
    /// </summary>
    bool TorsoSurfaceAt(float y, out float front, out float halfX, out float axisX)
    {
        front = 0f; halfX = 0f; axisX = 0f;
        if (!haveTorsoProfile) return false;
        float span = HAND_SPAN_DOWN + HAND_SPAN_UP;
        bool any = false;
        for (int i = 0; i < HAND_SPAN_SAMPLES; i++)
        {
            float yy = y - HAND_SPAN_DOWN + span * i / (HAND_SPAN_SAMPLES - 1f);
            float f, h, a;
            if (!BandSurface(yy, out f, out h, out a)) continue;
            if (!any || f > front) front = f;
            any = true;
        }
        if (!any) return false;
        float fb, hb, ab;
        if (!BandSurface(y, out fb, out hb, out ab)) return false;
        halfX = hb;
        axisX = ab;
        return true;
    }

    /// <summary>
    /// How far the hand's real mesh reaches BEHIND the wrist (world -Z) right
    /// now. Built from the posed palm vertices and the posed finger joints, so
    /// it tracks the actual hand: a rest-pose bounding box cannot, because its
    /// splayed fingers over-estimate the across extent by ~2x and shove the
    /// hand a hand's width clear of the chest.
    /// </summary>
    float HandBackExtent(bool isLeft)
    {
        int s = isLeft ? 0 : 1;
        Transform hand = handBoneXf[s];
        if (hand == null) return 0f;
        float wristZ = hand.position.z;
        float back = 0f;
        Vector3[] pts = handPts[s];
        if (pts != null)
        {
            for (int k = 0; k < pts.Length; k++)
            {
                float d = wristZ - hand.TransformPoint(pts[k]).z;
                if (d > back) back = d;
            }
        }
        Transform[] joints = handJoints[s];
        if (joints != null)
        {
            for (int k = 0; k < joints.Length; k++)
            {
                Transform j = joints[k];
                if (j == null || j == hand) continue;
                float d = wristZ - j.position.z + FINGER_TIP_MARGIN;
                if (d > back) back = d;
            }
        }
        return Mathf.Clamp(back, 0f, HAND_BACK_MAX);
    }

    /// <summary>
    /// Ground truth for "the hand sinks into the chest": the posed palm
    /// vertices tested against the measured torso surface. `inside` counts
    /// the vertices the torso mesh is drawn over, `deepest` is the worst
    /// intrusion in metres. Used by the playback probe.
    /// </summary>
    public void MeasureHandPenetration(bool isLeft, out int inside, out float deepest)
    {
        inside = 0; deepest = 0f;
        int s = isLeft ? 0 : 1;
        Transform hand = handBoneXf[s];
        Vector3[] pts = handPts[s];
        if (hand == null || pts == null || !haveTorsoProfile) return;
        for (int k = 0; k < pts.Length; k++)
        {
            Vector3 w = hand.TransformPoint(pts[k]);
            float front, halfX, axisX;
            if (!BandSurface(w.y, out front, out halfX, out axisX)) continue;
            if (Mathf.Abs(w.x - axisX) > halfX) continue;
            if (w.z < front)
            {
                inside++;
                float d = front - w.z;
                if (d > deepest) deepest = d;
            }
        }
    }

    Vector3 ToWorld(LandmarkPoint p)
    {
        float x = (invertX ? -p.x : p.x) * coordinateScale.x;
        float y = (invertY ? -p.y : p.y) * coordinateScale.y;
        float z = (invertZ ? -p.z : p.z) * coordinateScale.z;
        return transform.TransformDirection(new Vector3(x, y, z));
    }

    static Vector3 ToRaw(LandmarkPoint p)
    {
        return new Vector3(p.x, p.y, p.z);
    }

    // ==========================================
    // RESET POSE
    // ==========================================

    public void ResetPose()
    {
        foreach (KeyValuePair<HumanBodyBones, Quaternion> pair in initialRotations)
        {
            if (!boneMap.TryGetValue(pair.Key, out Transform bone) || bone == null)
                continue;
            bone.localRotation = pair.Value;
        }
        Debug.Log("HandPoseMapper: pose reset.");
    }
}
