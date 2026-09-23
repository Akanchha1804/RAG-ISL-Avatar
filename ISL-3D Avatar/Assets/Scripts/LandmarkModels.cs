using System;
using System.Collections.Generic;

/// <summary>
/// Shared landmark data model used by HandPoseMapper, LandmarkLoader,
/// LandmarkFetcher, and AvatarWebSocketClient. Single source of truth so
/// JsonUtility deserialization stays consistent across scripts.
///
/// JSON contract (backend / ISL_MediaPipe output):
///   handedness: "Left" | "Right"
///   landmarks: exactly 21 points per hand (x, y, z)
///   sparse frames: some frames have 0 hands — loaders must skip, not crash.
/// </summary>

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

/// <summary>Canonical handedness identity (independent of raw JSON string).</summary>
public enum HandSide
{
    Unknown = 0,
    Left = 1,
    Right = 2
}

/// <summary>Helpers for validating and normalizing hand identity.</summary>
public static class HandIdentity
{
    public const int ExpectedLandmarksPerHand = 21;

    /// <summary>Normalize a raw handedness string to HandSide.</summary>
    public static HandSide Normalize(string handedness)
    {
        if (string.IsNullOrEmpty(handedness))
            return HandSide.Unknown;

        string lower = handedness.ToLowerInvariant();
        if (lower.Contains("left"))
            return HandSide.Left;
        if (lower.Contains("right"))
            return HandSide.Right;
        return HandSide.Unknown;
    }

    /// <summary>True when the string is a recognized Left/Right label.</summary>
    public static bool IsValidHand(string handedness)
    {
        return Normalize(handedness) != HandSide.Unknown;
    }

    /// <summary>
    /// Validate a hand for animation: non-null, exactly 21 landmarks,
    /// recognized handedness.
    /// </summary>
    public static bool IsValidForAnimation(HandData hand)
    {
        if (hand == null || hand.landmarks == null)
            return false;
        if (hand.landmarks.Count != ExpectedLandmarksPerHand)
            return false;
        if (!IsValidHand(hand.handedness))
            return false;
        return true;
    }
}
