
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class HandPoseMapper : MonoBehaviour
{
    // ==========================================
    // DATA STRUCTURES
    // ==========================================

    [Serializable]
    public class LandmarkPoint
    {
        public float x;
        public float y;
        public float z;
    }

    [Serializable]
    public class HandData
    {
        public int hand_index;
        public List<LandmarkPoint> landmarks;
        public string handedness;
        public float handedness_score;
    }

    [Serializable]
    public class FrameData
    {
        public int frame_index;
        public int timestamp_ms;
        public List<HandData> hands;
    }

    [Serializable]
    public class LandmarkDataset
    {
        public string video_name;
        public float fps;
        public int frame_count;
        public int width;
        public int height;
        public int processed_frames;
        public List<FrameData> frames;
    }


    // ==========================================
    // SETTINGS
    // ==========================================

    [Header("JSON Settings")]
    public string jsonFileName = "landmarks.json";

    [Header("Animation Settings")]
    public bool playAnimation = true;
    public float animationSpeed = 1f;
    public float rotationSmoothness = 15f;

    [Header("Coordinate Settings")]
    public bool invertZ = true;
    public float coordinateScale = 1f;


    // ==========================================
    // PRIVATE VARIABLES
    // ==========================================

    private LandmarkDataset dataset;
    private Animator animator;

    private int currentFrameIndex = 0;
    private float frameTimer = 0f;

    private Dictionary<HumanBodyBones, Transform> boneMap =
        new Dictionary<HumanBodyBones, Transform>();

    private Dictionary<HumanBodyBones, Quaternion> initialRotations =
        new Dictionary<HumanBodyBones, Quaternion>();

    private Dictionary<HumanBodyBones, Vector3> initialDirections =
        new Dictionary<HumanBodyBones, Vector3>();


    // ==========================================
    // START
    // ==========================================

    void Start()
    {
        animator = GetComponent<Animator>();

        if (animator == null)
        {
            animator = GetComponentInChildren<Animator>();
        }

        if (animator == null)
        {
            Debug.LogError(
                "HandPoseMapper: Animator not found."
            );

            enabled = false;
            return;
        }

        if (!animator.isHuman)
        {
            Debug.LogError(
                "HandPoseMapper: Avatar is not Humanoid."
            );

            enabled = false;
            return;
        }

        SetupBones();
        LoadDataset();
    }


    // ==========================================
    // LOAD DATASET
    // ==========================================

    void LoadDataset()
    {
        string filePath = Path.Combine(
            Application.streamingAssetsPath,
            "Landmarks",
            jsonFileName
        );

        Debug.Log(
            "HandPoseMapper loading: " + filePath
        );

        if (!File.Exists(filePath))
        {
            Debug.LogError(
                "JSON file not found: " + filePath
            );

            enabled = false;
            return;
        }

        try
        {
            string json = File.ReadAllText(filePath);

            dataset =
                JsonUtility.FromJson<LandmarkDataset>(json);

            if (dataset == null ||
                dataset.frames == null ||
                dataset.frames.Count == 0)
            {
                Debug.LogError(
                    "Invalid or empty landmark dataset."
                );

                enabled = false;
                return;
            }

            Debug.Log(
                "HandPoseMapper dataset loaded: " +
                dataset.frames.Count + " frames"
            );
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "Dataset loading error: " +
                exception.Message
            );

            enabled = false;
        }
    }


    // ==========================================
    // SETUP HUMANOID BONES
    // ==========================================

    void SetupBones()
    {
        RegisterBone(HumanBodyBones.LeftHand);
        RegisterBone(HumanBodyBones.RightHand);

        // Left index
        RegisterBone(HumanBodyBones.LeftIndexProximal);
        RegisterBone(HumanBodyBones.LeftIndexIntermediate);
        RegisterBone(HumanBodyBones.LeftIndexDistal);

        // Left middle
        RegisterBone(HumanBodyBones.LeftMiddleProximal);
        RegisterBone(HumanBodyBones.LeftMiddleIntermediate);
        RegisterBone(HumanBodyBones.LeftMiddleDistal);

        // Left ring
        RegisterBone(HumanBodyBones.LeftRingProximal);
        RegisterBone(HumanBodyBones.LeftRingIntermediate);
        RegisterBone(HumanBodyBones.LeftRingDistal);

        // Left little
        RegisterBone(HumanBodyBones.LeftLittleProximal);
        RegisterBone(HumanBodyBones.LeftLittleIntermediate);
        RegisterBone(HumanBodyBones.LeftLittleDistal);

        // Left thumb
        RegisterBone(HumanBodyBones.LeftThumbProximal);
        RegisterBone(HumanBodyBones.LeftThumbDistal);

        // Right index
        RegisterBone(HumanBodyBones.RightIndexProximal);
        RegisterBone(HumanBodyBones.RightIndexIntermediate);
        RegisterBone(HumanBodyBones.RightIndexDistal);

        // Right middle
        RegisterBone(HumanBodyBones.RightMiddleProximal);
        RegisterBone(HumanBodyBones.RightMiddleIntermediate);
        RegisterBone(HumanBodyBones.RightMiddleDistal);

        // Right ring
        RegisterBone(HumanBodyBones.RightRingProximal);
        RegisterBone(HumanBodyBones.RightRingIntermediate);
        RegisterBone(HumanBodyBones.RightRingDistal);

        // Right little
        RegisterBone(HumanBodyBones.RightLittleProximal);
        RegisterBone(HumanBodyBones.RightLittleIntermediate);
        RegisterBone(HumanBodyBones.RightLittleDistal);

        // Right thumb
        RegisterBone(HumanBodyBones.RightThumbProximal);
        RegisterBone(HumanBodyBones.RightThumbDistal);

        Debug.Log(
            "HandPoseMapper registered bones: " +
            boneMap.Count
        );
    }


    void RegisterBone(HumanBodyBones bone)
    {
        Transform boneTransform =
            animator.GetBoneTransform(bone);

        if (boneTransform == null)
        {
            Debug.LogWarning(
                "Missing avatar bone: " + bone
            );

            return;
        }

        boneMap[bone] = boneTransform;

        initialRotations[bone] =
            boneTransform.localRotation;

        if (boneTransform.childCount > 0)
        {
            Transform child =
                boneTransform.GetChild(0);

            Vector3 direction =
                child.position -
                boneTransform.position;

            Vector3 localDirection =
                boneTransform.parent != null
                ? boneTransform.parent.InverseTransformDirection(
                    direction
                )
                : direction;

            if (localDirection.sqrMagnitude > 0.0001f)
            {
                initialDirections[bone] =
                    localDirection.normalized;
            }
        }
    }


    // ==========================================
    // UPDATE
    // ==========================================

    void Update()
    {
        if (!playAnimation ||
            dataset == null ||
            dataset.frames == null ||
            dataset.frames.Count == 0)
        {
            return;
        }

        float fps = dataset.fps;

        if (fps <= 0f)
        {
            fps = 25f;
        }

        frameTimer +=
            Time.deltaTime * animationSpeed;

        float frameDuration = 1f / fps;

        while (frameTimer >= frameDuration)
        {
            frameTimer -= frameDuration;

            currentFrameIndex++;

            if (currentFrameIndex >= dataset.frames.Count)
            {
                currentFrameIndex = 0;
            }
        }

        FrameData frame =
            dataset.frames[currentFrameIndex];

        if (frame == null ||
            frame.hands == null ||
            frame.hands.Count == 0)
        {
            return;
        }

        foreach (HandData hand in frame.hands)
        {
            if (hand == null ||
                hand.landmarks == null ||
                hand.landmarks.Count < 21)
            {
                continue;
            }

            if (string.IsNullOrEmpty(hand.handedness))
            {
                continue;
            }

            bool isLeft =
                hand.handedness.ToLower().Contains("left");

            bool isRight =
                hand.handedness.ToLower().Contains("right");

            if (isLeft)
            {
                AnimateHand(hand, true);
            }
            else if (isRight)
            {
                AnimateHand(hand, false);
            }
        }
    }


    // ==========================================
    // ANIMATE HAND
    // ==========================================

    void AnimateHand(
        HandData hand,
        bool isLeft
    )
    {
        // Index finger
        RotateBone(
            isLeft
                ? HumanBodyBones.LeftIndexProximal
                : HumanBodyBones.RightIndexProximal,
            hand.landmarks[5],
            hand.landmarks[6]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftIndexIntermediate
                : HumanBodyBones.RightIndexIntermediate,
            hand.landmarks[6],
            hand.landmarks[7]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftIndexDistal
                : HumanBodyBones.RightIndexDistal,
            hand.landmarks[7],
            hand.landmarks[8]
        );


        // Middle finger
        RotateBone(
            isLeft
                ? HumanBodyBones.LeftMiddleProximal
                : HumanBodyBones.RightMiddleProximal,
            hand.landmarks[9],
            hand.landmarks[10]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftMiddleIntermediate
                : HumanBodyBones.RightMiddleIntermediate,
            hand.landmarks[10],
            hand.landmarks[11]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftMiddleDistal
                : HumanBodyBones.RightMiddleDistal,
            hand.landmarks[11],
            hand.landmarks[12]
        );


        // Ring finger
        RotateBone(
            isLeft
                ? HumanBodyBones.LeftRingProximal
                : HumanBodyBones.RightRingProximal,
            hand.landmarks[13],
            hand.landmarks[14]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftRingIntermediate
                : HumanBodyBones.RightRingIntermediate,
            hand.landmarks[14],
            hand.landmarks[15]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftRingDistal
                : HumanBodyBones.RightRingDistal,
            hand.landmarks[15],
            hand.landmarks[16]
        );


        // Little finger
        RotateBone(
            isLeft
                ? HumanBodyBones.LeftLittleProximal
                : HumanBodyBones.RightLittleProximal,
            hand.landmarks[17],
            hand.landmarks[18]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftLittleIntermediate
                : HumanBodyBones.RightLittleIntermediate,
            hand.landmarks[18],
            hand.landmarks[19]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftLittleDistal
                : HumanBodyBones.RightLittleDistal,
            hand.landmarks[19],
            hand.landmarks[20]
        );


        // Thumb
        RotateBone(
            isLeft
                ? HumanBodyBones.LeftThumbProximal
                : HumanBodyBones.RightThumbProximal,
            hand.landmarks[1],
            hand.landmarks[2]
        );

        RotateBone(
            isLeft
                ? HumanBodyBones.LeftThumbDistal
                : HumanBodyBones.RightThumbDistal,
            hand.landmarks[2],
            hand.landmarks[3]
        );
    }


    // ==========================================
    // ROTATE BONE
    // ==========================================

    void RotateBone(
        HumanBodyBones bone,
        LandmarkPoint startPoint,
        LandmarkPoint endPoint
    )
    {
        if (!boneMap.ContainsKey(bone) ||
            !initialDirections.ContainsKey(bone))
        {
            return;
        }

        Transform boneTransform =
            boneMap[bone];

        if (boneTransform == null ||
            boneTransform.parent == null)
        {
            return;
        }

        Vector3 targetDirection =
            ConvertLandmarkDirection(
                startPoint,
                endPoint
            );

        if (targetDirection.sqrMagnitude < 0.0001f)
        {
            return;
        }

        targetDirection.Normalize();

        Transform parent =
            boneTransform.parent;

        Vector3 targetLocalDirection =
            parent.InverseTransformDirection(
                targetDirection
            ).normalized;

        Vector3 initialDirection =
            initialDirections[bone];

        Quaternion rotationDifference =
            Quaternion.FromToRotation(
                initialDirection,
                targetLocalDirection
            );

        Quaternion targetLocalRotation =
            rotationDifference *
            initialRotations[bone];

        float smoothFactor =
            1f - Mathf.Exp(
                -rotationSmoothness * Time.deltaTime
            );

        boneTransform.localRotation =
            Quaternion.Slerp(
                boneTransform.localRotation,
                targetLocalRotation,
                smoothFactor
            );
    }


    // ==========================================
    // COORDINATE CONVERSION
    // ==========================================

    Vector3 ConvertLandmarkDirection(
        LandmarkPoint startPoint,
        LandmarkPoint endPoint
    )
    {
        float zDirection =
            endPoint.z - startPoint.z;

        if (invertZ)
        {
            zDirection = -zDirection;
        }

        Vector3 direction =
            new Vector3(
                (endPoint.x - startPoint.x) * coordinateScale,
                -(endPoint.y - startPoint.y) * coordinateScale,
                zDirection * coordinateScale
            );

        return transform.TransformDirection(
            direction
        );
    }


    // ==========================================
    // RESET POSE
    // ==========================================

    public void ResetPose()
    {
        foreach (
            KeyValuePair<
                HumanBodyBones,
                Quaternion
            > pair in initialRotations
        )
        {
            if (!boneMap.ContainsKey(pair.Key))
            {
                continue;
            }

            Transform bone =
                boneMap[pair.Key];

            if (bone != null)
            {
                bone.localRotation =
                    pair.Value;
            }
        }

        Debug.Log(
            "HandPoseMapper: Pose reset."
        );
    }
}
