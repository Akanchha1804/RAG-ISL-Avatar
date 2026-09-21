
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class LandmarkLoader : MonoBehaviour
{
    // ==========================================
    // LANDMARK DATA STRUCTURES
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

    [Header("Visualization Settings")]
    public float coordinateScale = 4f;
    public float landmarkSize = 0.04f;

    private LandmarkDataset dataset;


    // ==========================================
    // START
    // ==========================================

    void Start()
    {
        LoadLandmarks();
    }


    // ==========================================
    // LOAD LANDMARK JSON
    // ==========================================

    public void LoadLandmarks()
    {
        string filePath = Path.Combine(
            Application.streamingAssetsPath,
            "Landmarks",
            jsonFileName
        );

        Debug.Log("Loading landmark file: " + filePath);

        if (!File.Exists(filePath))
        {
            Debug.LogError(
                "Landmark JSON file not found: " + filePath
            );

            return;
        }

        try
        {
            string json = File.ReadAllText(filePath);

            dataset = JsonUtility.FromJson<LandmarkDataset>(json);

            if (dataset == null || dataset.frames == null)
            {
                Debug.LogError(
                    "Failed to parse landmark JSON."
                );

                return;
            }

            Debug.Log("===== LANDMARK DATA LOADED =====");

            Debug.Log(
                "Video: " + dataset.video_name
            );

            Debug.Log(
                "FPS: " + dataset.fps
            );

            Debug.Log(
                "Total frames: " + dataset.frames.Count
            );

            int framesWithHands = 0;
            int totalHands = 0;

            foreach (FrameData frame in dataset.frames)
            {
                if (frame.hands != null && frame.hands.Count > 0)
                {
                    framesWithHands++;
                    totalHands += frame.hands.Count;
                }
            }

            Debug.Log(
                "Frames with hands: " + framesWithHands
            );

            Debug.Log(
                "Total detected hands: " + totalHands
            );

            // Visualize the first detected frame
            VisualizeFirstDetectedFrame();

            Debug.Log(
                "Landmark loading completed."
            );
        }
        catch (Exception exception)
        {
            Debug.LogError(
                "Error loading landmarks: " + exception.Message
            );
        }
    }


    // ==========================================
    // VISUALIZE FIRST DETECTED FRAME
    // ==========================================

    void VisualizeFirstDetectedFrame()
    {
        if (dataset == null || dataset.frames == null)
        {
            Debug.LogError(
                "Dataset is not available."
            );

            return;
        }

        foreach (FrameData frame in dataset.frames)
        {
            if (frame.hands == null || frame.hands.Count == 0)
            {
                continue;
            }

            foreach (HandData hand in frame.hands)
            {
                if (hand.landmarks == null ||
                    hand.landmarks.Count == 0)
                {
                    continue;
                }

                foreach (LandmarkPoint point in hand.landmarks)
                {
                    // Convert normalized MediaPipe coordinates
                    // into Unity world coordinates.

                    float x =
                        (point.x - 0.5f) * coordinateScale;

                    float y =
                        (0.5f - point.y) * coordinateScale;

                    float z =
                        -point.z * coordinateScale;

                    // Create a sphere for the landmark

                    GameObject sphere =
                        GameObject.CreatePrimitive(
                            PrimitiveType.Sphere
                        );

                    sphere.name = "HandLandmark";

                    sphere.transform.position =
                        transform.position +
                        new Vector3(x, y, z);

                    sphere.transform.localScale =
                        Vector3.one * landmarkSize;

                    // Make the LandmarkSystem the parent

                    sphere.transform.SetParent(
                        transform
                    );
                }
            }

            Debug.Log(
                "Visualized frame: " + frame.frame_index
            );

            // Only visualize the first detected frame

            break;
        }
    }
}
