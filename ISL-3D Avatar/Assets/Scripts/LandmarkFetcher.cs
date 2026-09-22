using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class LandmarkFetcher : MonoBehaviour
{
    // ==========================================
    // SETTINGS
    // ==========================================

    [Header("Backend Settings")]
    public string backendUrl = "http://localhost:8000";
    public string landmarkEndpoint = "/landmarks/";

    [Header("Input Settings")]
    public string jsonFileName = "landmarks.json";
    public bool loadFromBackend = false;

    [Header("Debug")]
    public bool logResponses = true;

    // ==========================================
    // EVENTS
    // ==========================================

    public event Action<LandmarkDataset> OnLandmarksLoaded;
    public event Action<string> OnLoadError;
    public event Action<float> OnLoadProgress;

    // ==========================================
    // DATA STRUCTURES (same as HandPoseMapper)
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
    // PRIVATE
    // ==========================================

    private LandmarkDataset currentDataset;
    private HandPoseMapper handPoseMapper;

    // ==========================================
    // START
    // ==========================================

    void Start()
    {
        handPoseMapper = GetComponent<HandPoseMapper>();

        if (handPoseMapper == null)
        {
            handPoseMapper = GetComponentInChildren<HandPoseMapper>();
        }

        if (loadFromBackend)
        {
            FetchLandmarksFromBackend(jsonFileName);
        }
        else
        {
            LoadLandmarksFromStreamingAssets(jsonFileName);
        }
    }

    // ==========================================
    // LOAD FROM STREAMING ASSETS
    // ==========================================

    public void LoadLandmarksFromStreamingAssets(string fileName)
    {
        string filePath = Path.Combine(
            Application.streamingAssetsPath,
            "Landmarks",
            fileName
        );

        if (!File.Exists(filePath))
        {
            Debug.LogError("Landmark file not found: " + filePath);
            OnLoadError?.Invoke("File not found: " + fileName);
            return;
        }

        StartCoroutine(LoadLocalFile(filePath));
    }

    // ==========================================
    // FETCH FROM BACKEND
    // ==========================================

    public void FetchLandmarksFromBackend(string fileName)
    {
        string url = backendUrl + landmarkEndpoint + fileName;
        Debug.Log("Fetching landmarks from: " + url);
        StartCoroutine(FetchRemoteFile(url));
    }

    public void FetchLandmarksBySentence(string sentence)
    {
        string encoded = Uri.EscapeDataString(sentence);
        string url = backendUrl + "/api/translate";
        StartCoroutine(PostTranslateAndLoad(url, sentence));
    }

    // ==========================================
    // COROUTINES
    // ==========================================

    IEnumerator LoadLocalFile(string filePath)
    {
        OnLoadProgress?.Invoke(0f);

        string json = File.ReadAllText(filePath);
        OnLoadProgress?.Invoke(0.5f);

        yield return null;

        currentDataset = JsonUtility.FromJson<LandmarkDataset>(json);
        OnLoadProgress?.Invoke(1f);

        if (currentDataset != null && currentDataset.frames != null)
        {
            Debug.Log("Loaded " + currentDataset.frames.Count + " frames from file.");
            OnLandmarksLoaded?.Invoke(currentDataset);
        }
        else
        {
            Debug.LogError("Invalid landmark data.");
            OnLoadError?.Invoke("Invalid landmark data.");
        }
    }

    IEnumerator FetchRemoteFile(string url)
    {
        OnLoadProgress?.Invoke(0f);

        using (UnityWebRequest request = UnityWebRequest.Get(url))
        {
            yield return request.SendWebRequest();

            OnLoadProgress?.Invoke(0.5f);

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Failed to fetch landmarks: " + request.error);
                OnLoadError?.Invoke(request.error);
                yield break;
            }

            string json = request.downloadHandler.text;
            OnLoadProgress?.Invoke(0.8f);

            currentDataset = JsonUtility.FromJson<LandmarkDataset>(json);
            OnLoadProgress?.Invoke(1f);

            if (currentDataset != null && currentDataset.frames != null)
            {
                Debug.Log("Fetched " + currentDataset.frames.Count + " frames from backend.");
                OnLandmarksLoaded?.Invoke(currentDataset);
            }
            else
            {
                Debug.LogError("Invalid landmark data from backend.");
                OnLoadError?.Invoke("Invalid landmark data.");
            }
        }
    }

    IEnumerator PostTranslateAndLoad(string url, string sentence)
    {
        OnLoadProgress?.Invoke(0f);

        string body = JsonUtility.ToJson(new TranslateRequest { text = sentence, use_rag = true });
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(body);

        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            request.uploadHandler = new UploadHandlerRaw(bodyRaw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");

            yield return request.SendWebRequest();

            if (request.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError("Translate request failed: " + request.error);
                OnLoadError?.Invoke(request.error);
                yield break;
            }

            TranslateResponse response = JsonUtility.FromJson<TranslateResponse>(request.downloadHandler.text);

            if (response != null && !string.IsNullOrEmpty(response.landmark_file))
            {
                Debug.Log("Matched: " + response.matched_sentence + " (method: " + response.method + ")");
                FetchLandmarksFromBackend(response.landmark_file);
            }
            else
            {
                Debug.LogWarning("No landmark match for: " + sentence);
                OnLoadError?.Invoke("No match found for: " + sentence);
            }
        }
    }

    // ==========================================
    // PUBLIC API
    // ==========================================

    public void LoadSentence(string sentence)
    {
        if (loadFromBackend)
        {
            FetchLandmarksBySentence(sentence);
        }
        else
        {
            string safeName = sentence.Replace(" ", "_").Replace("'", "").Replace(",", "");
            string fileName = safeName + "_landmarks.json";
            LoadLandmarksFromStreamingAssets(fileName);
        }
    }

    public LandmarkDataset GetCurrentDataset()
    {
        return currentDataset;
    }

    public bool IsLoaded()
    {
        return currentDataset != null &&
               currentDataset.frames != null &&
               currentDataset.frames.Count > 0;
    }

    // ==========================================
    // REQUEST/RESPONSE MODELS
    // ==========================================

    [Serializable]
    private class TranslateRequest
    {
        public string text;
        public bool use_rag;
    }

    [Serializable]
    private class TranslateResponse
    {
        public string input_text;
        public string matched_sentence;
        public string glosses;
        public string landmark_file;
        public float similarity;
        public string landmark_url;
        public string method;
    }
}
