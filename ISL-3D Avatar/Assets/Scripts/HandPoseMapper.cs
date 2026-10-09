
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

    // Composed-playback playlist (novel sentences): per-gloss landmark
    // files played in order. Sentence clips ignore the queue entirely.
    private readonly Queue<string> playlistQueue = new Queue<string>();
    private bool playlistMode = false;

    // Previous-frame wrist positions (raw landmark coords) for the
    // per-side single-hand rule.
    private Vector3 lastWristLeft, lastWristRight;
    private bool haveLastWristLeft, haveLastWristRight;
    // Previous wrist orientations for the angular-velocity cap.
    private Quaternion lastWristQLeft, lastWristQRight;
    private bool haveLastWristQLeft, haveLastWristQRight;
    // Torso collision capsule (hips->neck axis + radius), web parity.
    private Vector3 torsoHips, torsoNeck;
    private float torsoRadius = 0.15f;
    private bool haveTorsoCapsule = false;

    // Two-bone arm rig per side (mirrors the web VrmAvatar solver so both
    // viewports carry the wrists into the same signing box). The landmark
    // files carry hand points only; without arms the hands would sign down
    // at the hips. Null when the avatar lacks the required bones, in which
    // case hands still rotate and arms stay on the Animator.
    private class ArmRig
    {
        public Transform upper, lower;
        public Vector3 shoulder;      // rest world position (S)
        public float lenU, lenL;      // upper / forearm lengths
        public Quaternion restWQU, restWQL;
        public Vector3 restDirU, restDirL;
        public Vector3 pole;          // elbow hint (world)
        public Vector3 chest;         // signing-box anchor (world)
    }
    private ArmRig leftArm, rightArm;

    // Per-clip signing-space calibration (M6 fix): source videos frame the
    // signer differently, so absolute image coords reproduce the wrong height
    // (waist-level source -> waist-level avatar). Each dataset's wrist
    // trajectory bbox is mapped into the signing box instead. Min-span
    // guarded so a static hold maps mid-box rather than amplifying jitter.
    private Vector3 calibMin, calibSpan;
    private bool calibrated = false;

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

        // At most ONE hand per side per frame: tracker mislabels put two
        // same-side hands in a frame (190/288 clips affected), and driving
        // both double-drives one arm (last-wins flicker + starved side).
        // Keep the wrist nearest that side's previous wrist (web parity).
        HandData leftPick = null, rightPick = null;
        Vector3 leftPrev = lastWristLeft, rightPrev = lastWristRight;
        bool haveLeftPrev = haveLastWristLeft, haveRightPrev = haveLastWristRight;
        foreach (HandData hand in frame.hands)
        {
            if (!HandIdentity.IsValidForAnimation(hand))
                continue;

            HandSide side = HandIdentity.Normalize(hand.handedness);
            if (mirrorHands)
                side = side == HandSide.Left ? HandSide.Right : HandSide.Left;

            Vector3 wrist = ToRaw(hand.landmarks[0]);
            if (side == HandSide.Left)
            {
                if (leftPick == null || (haveLeftPrev && (wrist - leftPrev).sqrMagnitude
                    < (ToRaw(leftPick.landmarks[0]) - leftPrev).sqrMagnitude))
                    leftPick = hand;
            }
            else if (side == HandSide.Right)
            {
                if (rightPick == null || (haveRightPrev && (wrist - rightPrev).sqrMagnitude
                    < (ToRaw(rightPick.landmarks[0]) - rightPrev).sqrMagnitude))
                    rightPick = hand;
            }
        }
        if (leftPick != null)
        {
            lastWristLeft = ToRaw(leftPick.landmarks[0]);
            haveLastWristLeft = true;
            AnimateHand(leftPick, true);
        }
        if (rightPick != null)
        {
            lastWristRight = ToRaw(rightPick.landmarks[0]);
            haveLastWristRight = true;
            AnimateHand(rightPick, false);
        }
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
        if (string.IsNullOrEmpty(fileName))
        {
            Debug.LogWarning("HandPoseMapper.LoadLandmarkFile: empty file name.");
            return;
        }

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
            HumanBodyBones.LeftMiddleProximal,
            ref avatarLeftOrigin, ref avatarLeftAxisX, ref avatarLeftAxisY, ref avatarLeftAxisZ);

        BuildSideBasis(HumanBodyBones.RightHand, HumanBodyBones.RightIndexProximal,
            HumanBodyBones.RightMiddleProximal,
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
        if (Vector3.Dot(axisZ, outward) < 0f)
        {
            axisY = -axisY;
            axisZ = -axisZ;
            Debug.Log($"HandPoseMapper: mirror-corrected the {(isLeft ? "left" : "right")} palm basis.");
        }
    }

    void BuildSideBasis(
        HumanBodyBones handBone, HumanBodyBones indexMcp, HumanBodyBones middleMcp,
        ref Vector3 origin, ref Vector3 axisX, ref Vector3 axisY, ref Vector3 axisZ)
    {
        if (!boneMap.TryGetValue(handBone, out Transform hand) || hand == null)
            return;

        boneMap.TryGetValue(indexMcp, out Transform index);
        boneMap.TryGetValue(middleMcp, out Transform middle);
        if (index == null || middle == null)
            return;

        origin = hand.position;
        Vector3 toIndex = index.position - hand.position;
        Vector3 toMiddle = middle.position - hand.position;

        if (toIndex.sqrMagnitude < 1e-8f || toMiddle.sqrMagnitude < 1e-8f)
            return;

        axisX = toIndex.normalized;                         // toward index MCP
        Vector3 palmNormal = Vector3.Cross(toIndex, toMiddle);
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

        leftArm = BuildArmRig(true, chest);
        rightArm = BuildArmRig(false, chest);
        if (leftArm == null && rightArm == null)
            Debug.LogWarning("HandPoseMapper: no arm bones found; arm IK disabled.");
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
            upper = upper, lower = lower,
            shoulder = S, lenU = lenU, lenL = lenL,
            restWQU = restWQU, restWQL = restWQL,
            restDirU = restDirU, restDirL = restDirL,
            pole = pole, chest = chest,
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
            // Lateral gain matches the web solver (±0.275m): full-box
            // lateral sweeps read as stiff wingspans on short avatar arms.
            Vector3 target = new Vector3(
                C.x + (w.x - 0.5f) * 0.55f * s,
                C.y + 0.02f * s + (0.5f - w.y) * 0.7f * s,
                Mathf.Max(C.z + 0.3f * s - w.z * 0.4f * s, C.z + 0.06f * s));

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
            Vector3 dirST = (target - rig.shoulder).normalized;

            float cosA = Mathf.Clamp(
                (rig.lenU * rig.lenU + dist * dist - rig.lenL * rig.lenL)
                / (2f * rig.lenU * dist), -1f, 1f);
            float angA = Mathf.Acos(cosA);
            Vector3 perp = rig.pole - dirST * Vector3.Dot(rig.pole, dirST);
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

            Quaternion parentQU = rig.upper.parent != null
                ? rig.upper.parent.rotation : Quaternion.identity;
            Quaternion qU = Quaternion.FromToRotation(rig.restDirU, upperDir) * rig.restWQU;
            rig.upper.rotation = Smooth(rig.upper.rotation,
                Quaternion.Inverse(parentQU) * qU);

            Quaternion parentQL = rig.lower.parent != null
                ? rig.lower.parent.rotation : Quaternion.identity;
            Quaternion qL = Quaternion.FromToRotation(rig.restDirL, foreDir) * rig.restWQL;
            rig.lower.rotation = Smooth(rig.lower.rotation,
                Quaternion.Inverse(parentQL) * qL);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"HandPoseMapper: arm IK skipped ({e.Message}).");
        }
    }

    // ==========================================
    // ANIMATE HAND (palm-frame retargeting)
    // ==========================================

    void AnimateHand(HandData hand, bool isLeft)
    {
        // Arms first so the hands ride up into signing space (web parity).
        // Uses raw landmark coords, same convention as the web solver.
        if (hand.landmarks != null && hand.landmarks.Count > 0)
        {
            LandmarkPoint wrist = hand.landmarks[0];
            SolveArm(isLeft ? leftArm : rightArm,
                new Vector3(wrist.x, wrist.y, wrist.z));
        }

        // Landmark palm basis: wrist(0) -> index MCP(5) -> middle MCP(9)
        Vector3 lmOrigin = ToWorld(hand.landmarks[0]);
        Vector3 lmToIndex = ToWorld(hand.landmarks[5]) - lmOrigin;
        Vector3 lmToMiddle = ToWorld(hand.landmarks[9]) - lmOrigin;

        if (lmToIndex.sqrMagnitude < 1e-10f || lmToMiddle.sqrMagnitude < 1e-10f)
            return;

        Vector3 lmX = lmToIndex.normalized;
        Vector3 lmZ = Vector3.Cross(lmToIndex, lmToMiddle).normalized;
        if (lmZ.sqrMagnitude < 1e-10f)
            return;
        Vector3 lmY = Vector3.Cross(lmZ, lmX).normalized;
        lmX = Vector3.Cross(lmY, lmZ).normalized;

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
            initialRotations.TryGetValue(handBone, out Quaternion handRest))
        {
            Quaternion target = palmDelta * handRest;
            if (isLeft)
            {
                if (!haveLastWristQLeft) { lastWristQLeft = target; haveLastWristQLeft = true; }
                else
                {
                    float maxStep = 7f * Mathf.Max(Time.deltaTime, 1e-3f);
                    float step = Quaternion.Angle(lastWristQLeft, target);
                    if (step > maxStep * Mathf.Rad2Deg)
                        target = Quaternion.RotateTowards(lastWristQLeft, target, maxStep);
                    lastWristQLeft = target;
                }
            }
            else
            {
                if (!haveLastWristQRight) { lastWristQRight = target; haveLastWristQRight = true; }
                else
                {
                    float maxStep = 7f * Mathf.Max(Time.deltaTime, 1e-3f);
                    float step = Quaternion.Angle(lastWristQRight, target);
                    if (step > maxStep * Mathf.Rad2Deg)
                        target = Quaternion.RotateTowards(lastWristQRight, target, maxStep);
                    lastWristQRight = target;
                }
            }
            handTransform.localRotation = Smooth(handTransform.localRotation, target);
        }

        // Fingers: express each landmark segment direction in the landmark palm
        // frame, map into the avatar palm frame, then FromTo against rest.
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexProximal : HumanBodyBones.RightIndexProximal, hand.landmarks[5], hand.landmarks[6], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexIntermediate : HumanBodyBones.RightIndexIntermediate, hand.landmarks[6], hand.landmarks[7], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexDistal : HumanBodyBones.RightIndexDistal, hand.landmarks[7], hand.landmarks[8], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleProximal : HumanBodyBones.RightMiddleProximal, hand.landmarks[9], hand.landmarks[10], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleIntermediate : HumanBodyBones.RightMiddleIntermediate, hand.landmarks[10], hand.landmarks[11], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleDistal : HumanBodyBones.RightMiddleDistal, hand.landmarks[11], hand.landmarks[12], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingProximal : HumanBodyBones.RightRingProximal, hand.landmarks[13], hand.landmarks[14], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingIntermediate : HumanBodyBones.RightRingIntermediate, hand.landmarks[14], hand.landmarks[15], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingDistal : HumanBodyBones.RightRingDistal, hand.landmarks[15], hand.landmarks[16], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleProximal : HumanBodyBones.RightLittleProximal, hand.landmarks[17], hand.landmarks[18], lmX, lmY, lmZ, avX, avY, avZ, 1f, 100f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleIntermediate : HumanBodyBones.RightLittleIntermediate, hand.landmarks[18], hand.landmarks[19], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleDistal : HumanBodyBones.RightLittleDistal, hand.landmarks[19], hand.landmarks[20], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 80f);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbProximal : HumanBodyBones.RightThumbProximal, hand.landmarks[1], hand.landmarks[2], lmX, lmY, lmZ, avX, avY, avZ, 0.6f, 90f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbIntermediate : HumanBodyBones.RightThumbIntermediate, hand.landmarks[2], hand.landmarks[3], lmX, lmY, lmZ, avX, avY, avZ, 0.8f, 90f);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbDistal : HumanBodyBones.RightThumbDistal, hand.landmarks[3], hand.landmarks[4], lmX, lmY, lmZ, avX, avY, avZ, 0.45f, 90f);
    }

    void RotateFingerBone(
        HumanBodyBones bone,
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

        // Map into avatar palm frame (world space for this frame's basis).
        Vector3 targetWorld = avX * inLm.x + avY * inLm.y + avZ * inLm.z;

        // Into parent space, then FromTo against rest direction, clamped
        // to the anatomical flexion limit (web parity: no folded-back
        // fingers no matter what the tracker reports).
        Vector3 targetLocal = boneTransform.parent.InverseTransformDirection(targetWorld).normalized;
        if (targetLocal.sqrMagnitude < 1e-8f)
            return;

        Quaternion target;
        float flexAng = Vector3.Angle(restLocal, targetLocal);
        if (flexAng > maxFlexDeg)
        {
            Vector3 hinge = Vector3.Cross(restLocal, targetLocal);
            if (hinge.sqrMagnitude > 1e-10f)
                targetLocal = Quaternion.AngleAxis(maxFlexDeg, hinge.normalized) * restLocal;
            else
                targetLocal = restLocal;
        }
        Quaternion delta = Quaternion.FromToRotation(restLocal, targetLocal);
        target = delta * restRot;
        boneTransform.localRotation = Smooth(boneTransform.localRotation, target, damp);
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
        float t = Mathf.Clamp01(Vector3.Dot(point - torsoHips, axis) / lenSq);
        Vector3 center = torsoHips + axis * t;
        Vector3 radial = point - center;
        float d = radial.magnitude;
        if (d >= torsoRadius + standoff || d < 1e-9f) return false;
        point = center + radial.normalized * (torsoRadius + standoff);
        return true;
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
