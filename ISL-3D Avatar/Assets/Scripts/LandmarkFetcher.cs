using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

/// <summary>
/// Fetches landmark JSON from StreamingAssets or the backend and injects the
/// parsed dataset into HandPoseMapper via SetDataset (no duplicate models).
///
/// Changes from the previous version:
///   - Uses the shared LandmarkModels data types.
///   - Wires SetDataset into HandPoseMapper after every successful load so the
///     mapper actually animates what was fetched.
/// </summary>
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
            handPoseMapper = GetComponentInChildren<HandPoseMapper>();

        if (loadFromBackend)
            FetchLandmarksFromBackend(jsonFileName);
        else
            LoadLandmarksFromStreamingAssets(jsonFileName);
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
            Debug.LogError("LandmarkFetcher: file not found: " + filePath);
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
        if (logResponses)
            Debug.Log("LandmarkFetcher: fetching " + url);
        StartCoroutine(FetchRemoteFile(url));
    }

    public void FetchLandmarksBySentence(string sentence)
    {
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

        if (currentDataset != null && currentDataset.frames != null && currentDataset.frames.Count > 0)
        {
            if (logResponses)
                Debug.Log($"LandmarkFetcher: loaded {currentDataset.frames.Count} frames from file.");
            PushToMapper();
            OnLandmarksLoaded?.Invoke(currentDataset);
        }
        else
        {
            Debug.LogError("LandmarkFetcher: invalid landmark data.");
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
                Debug.LogError("LandmarkFetcher: fetch failed: " + request.error);
                OnLoadError?.Invoke(request.error);
                yield break;
            }

            string json = request.downloadHandler.text;
            OnLoadProgress?.Invoke(0.8f);

            currentDataset = JsonUtility.FromJson<LandmarkDataset>(json);
            OnLoadProgress?.Invoke(1f);

            if (currentDataset != null && currentDataset.frames != null && currentDataset.frames.Count > 0)
            {
                if (logResponses)
                    Debug.Log($"LandmarkFetcher: fetched {currentDataset.frames.Count} frames from backend.");
                PushToMapper();
                OnLandmarksLoaded?.Invoke(currentDataset);
            }
            else
            {
                Debug.LogError("LandmarkFetcher: invalid landmark data from backend.");
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
                Debug.LogError("LandmarkFetcher: translate request failed: " + request.error);
                OnLoadError?.Invoke(request.error);
                yield break;
            }

            TranslateResponse response =
                JsonUtility.FromJson<TranslateResponse>(request.downloadHandler.text);

            if (response != null && !string.IsNullOrEmpty(response.landmark_file))
            {
                if (logResponses)
                    Debug.Log($"LandmarkFetcher: matched '{response.matched_sentence}' (method: {response.method})");
                FetchLandmarksFromBackend(response.landmark_file);
            }
            else
            {
                Debug.LogWarning("LandmarkFetcher: no landmark match for: " + sentence);
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

    public LandmarkDataset GetCurrentDataset() => currentDataset;

    public bool IsLoaded() =>
        currentDataset != null &&
        currentDataset.frames != null &&
        currentDataset.frames.Count > 0;

    void PushToMapper()
    {
        if (handPoseMapper == null)
        {
            handPoseMapper = GetComponent<HandPoseMapper>();
            if (handPoseMapper == null)
                handPoseMapper = GetComponentInChildren<HandPoseMapper>();
        }

        if (handPoseMapper != null && currentDataset != null)
            handPoseMapper.SetDataset(currentDataset);
        else if (logResponses)
            Debug.Log("LandmarkFetcher: no HandPoseMapper found to receive dataset.");
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
