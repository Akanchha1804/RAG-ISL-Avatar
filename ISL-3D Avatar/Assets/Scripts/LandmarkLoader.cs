
using System;
using System.IO;
using UnityEngine;

/// <summary>
/// Loads a landmark JSON for debug visualization.
///
/// Changes from the previous version:
///   - Uses the shared LandmarkModels data types (no nested duplicates).
///   - Validates exactly 21 landmarks per hand via HandIdentity before use.
///   - Gracefully skips frames with zero hands (sparse data is normal).
/// </summary>
public class LandmarkLoader : MonoBehaviour
{
    [Header("JSON Settings")]
    public string jsonFileName = "landmarks.json";

    [Header("Visualization Settings")]
    public float coordinateScale = 4f;
    public float landmarkSize = 0.04f;

    private LandmarkDataset dataset;

    void Start()
    {
        LoadLandmarks();
    }

    public void LoadLandmarks()
    {
        string filePath = Path.Combine(
            Application.streamingAssetsPath,
            "Landmarks",
            jsonFileName
        );

        Debug.Log("LandmarkLoader: loading " + filePath);

        if (!File.Exists(filePath))
        {
            Debug.LogError("LandmarkLoader: file not found: " + filePath);
            return;
        }

        try
        {
            string json = File.ReadAllText(filePath);
            dataset = JsonUtility.FromJson<LandmarkDataset>(json);

            if (dataset == null || dataset.frames == null || dataset.frames.Count == 0)
            {
                Debug.LogError("LandmarkLoader: invalid or empty dataset.");
                return;
            }

            int framesWithHands = 0;
            int validHands = 0;
            int invalidHands = 0;

            foreach (FrameData frame in dataset.frames)
            {
                if (frame.hands == null || frame.hands.Count == 0)
                    continue; // sparse frame — not an error

                framesWithHands++;
                foreach (HandData hand in frame.hands)
                {
                    if (HandIdentity.IsValidForAnimation(hand))
                        validHands++;
                    else
                        invalidHands++;
                }
            }

            Debug.Log($"LandmarkLoader: {dataset.frames.Count} frames, " +
                      $"{framesWithHands} with hands, " +
                      $"{validHands} valid hands, {invalidHands} invalid hands.");

            VisualizeFirstDetectedFrame();
        }
        catch (Exception e)
        {
            Debug.LogError("LandmarkLoader: error loading landmarks: " + e.Message);
        }
    }

    void VisualizeFirstDetectedFrame()
    {
        if (dataset == null || dataset.frames == null)
        {
            Debug.LogError("LandmarkLoader: dataset is not available.");
            return;
        }

        foreach (FrameData frame in dataset.frames)
        {
            if (frame.hands == null || frame.hands.Count == 0)
                continue;

            foreach (HandData hand in frame.hands)
            {
                if (hand.landmarks == null || hand.landmarks.Count == 0)
                    continue;

                foreach (LandmarkPoint point in hand.landmarks)
                {
                    // Convert normalized MediaPipe coordinates into Unity world coordinates.
                    float x = (point.x - 0.5f) * coordinateScale;
                    float y = (0.5f - point.y) * coordinateScale;
                    float z = -point.z * coordinateScale;

                    GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    sphere.name = "HandLandmark";
                    sphere.transform.position = transform.position + new Vector3(x, y, z);
                    sphere.transform.localScale = Vector3.one * landmarkSize;
                    sphere.transform.SetParent(transform);
                }
            }

            Debug.Log("LandmarkLoader: visualized frame " + frame.frame_index);
            break; // only the first detected frame
        }
    }
}
