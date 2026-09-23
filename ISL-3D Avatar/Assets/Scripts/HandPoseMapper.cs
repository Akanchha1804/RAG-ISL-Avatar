
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
    private bool avatarBasisBuilt;
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

        while (frameTimer >= frameDuration)
        {
            frameTimer -= frameDuration;
            currentFrameIndex++;
            if (currentFrameIndex >= dataset.frames.Count)
                currentFrameIndex = 0;
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

        foreach (HandData hand in frame.hands)
        {
            if (!HandIdentity.IsValidForAnimation(hand))
                continue;

            HandSide side = HandIdentity.Normalize(hand.handedness);
            if (mirrorHands)
                side = side == HandSide.Left ? HandSide.Right : HandSide.Left;

            AnimateHand(hand, side == HandSide.Left);
        }
    }

    // ==========================================
    // PUBLIC API: dataset loading
    // ==========================================

    /// <summary>
    /// Load a landmark file by name. Tries StreamingAssets/Landmarks first,
    /// then falls back to HTTP GET {backendUrl}/landmarks/{fileName}.
    /// Called by AvatarWebSocketClient on landmark_file / landmark_update.
    /// </summary>
    public void LoadLandmarkFile(string fileName)
    {
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
        Debug.Log($"HandPoseMapper: dataset set ({dataset.frames.Count} frames).");
    }

    public LandmarkDataset GetDataset() => dataset;

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

        avatarBasisBuilt = true;
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
    // ANIMATE HAND (palm-frame retargeting)
    // ==========================================

    void AnimateHand(HandData hand, bool isLeft)
    {
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

        // Rotate hand root with the palm delta (absolute, non-accumulating).
        HumanBodyBones handBone = isLeft ? HumanBodyBones.LeftHand : HumanBodyBones.RightHand;
        if (boneMap.TryGetValue(handBone, out Transform handTransform) &&
            initialRotations.TryGetValue(handBone, out Quaternion handRest))
        {
            Quaternion target = palmDelta * handRest;
            handTransform.localRotation = Smooth(handTransform.localRotation, target);
        }

        // Fingers: express each landmark segment direction in the landmark palm
        // frame, map into the avatar palm frame, then FromTo against rest.
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexProximal : HumanBodyBones.RightIndexProximal, hand.landmarks[5], hand.landmarks[6], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexIntermediate : HumanBodyBones.RightIndexIntermediate, hand.landmarks[6], hand.landmarks[7], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftIndexDistal : HumanBodyBones.RightIndexDistal, hand.landmarks[7], hand.landmarks[8], lmX, lmY, lmZ, avX, avY, avZ);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleProximal : HumanBodyBones.RightMiddleProximal, hand.landmarks[9], hand.landmarks[10], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleIntermediate : HumanBodyBones.RightMiddleIntermediate, hand.landmarks[10], hand.landmarks[11], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftMiddleDistal : HumanBodyBones.RightMiddleDistal, hand.landmarks[11], hand.landmarks[12], lmX, lmY, lmZ, avX, avY, avZ);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingProximal : HumanBodyBones.RightRingProximal, hand.landmarks[13], hand.landmarks[14], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingIntermediate : HumanBodyBones.RightRingIntermediate, hand.landmarks[14], hand.landmarks[15], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftRingDistal : HumanBodyBones.RightRingDistal, hand.landmarks[15], hand.landmarks[16], lmX, lmY, lmZ, avX, avY, avZ);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleProximal : HumanBodyBones.RightLittleProximal, hand.landmarks[17], hand.landmarks[18], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleIntermediate : HumanBodyBones.RightLittleIntermediate, hand.landmarks[18], hand.landmarks[19], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftLittleDistal : HumanBodyBones.RightLittleDistal, hand.landmarks[19], hand.landmarks[20], lmX, lmY, lmZ, avX, avY, avZ);

        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbProximal : HumanBodyBones.RightThumbProximal, hand.landmarks[1], hand.landmarks[2], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbIntermediate : HumanBodyBones.RightThumbIntermediate, hand.landmarks[2], hand.landmarks[3], lmX, lmY, lmZ, avX, avY, avZ);
        RotateFingerBone(isLeft ? HumanBodyBones.LeftThumbDistal : HumanBodyBones.RightThumbDistal, hand.landmarks[3], hand.landmarks[4], lmX, lmY, lmZ, avX, avY, avZ);
    }

    void RotateFingerBone(
        HumanBodyBones bone,
        LandmarkPoint start, LandmarkPoint end,
        Vector3 lmX, Vector3 lmY, Vector3 lmZ,
        Vector3 avX, Vector3 avY, Vector3 avZ)
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

        // Into parent space, then FromTo against rest direction.
        Vector3 targetLocal = boneTransform.parent.InverseTransformDirection(targetWorld).normalized;
        if (targetLocal.sqrMagnitude < 1e-8f)
            return;

        Quaternion delta = Quaternion.FromToRotation(restLocal, targetLocal);
        Quaternion target = delta * restRot;
        boneTransform.localRotation = Smooth(boneTransform.localRotation, target);
    }

    Quaternion Smooth(Quaternion current, Quaternion target)
    {
        float t = 1f - Mathf.Exp(-rotationSmoothness * Time.deltaTime);
        return Quaternion.Slerp(current, target, t);
    }

    Vector3 ToWorld(LandmarkPoint p)
    {
        float x = (invertX ? -p.x : p.x) * coordinateScale.x;
        float y = (invertY ? -p.y : p.y) * coordinateScale.y;
        float z = (invertZ ? -p.z : p.z) * coordinateScale.z;
        return transform.TransformDirection(new Vector3(x, y, z));
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
