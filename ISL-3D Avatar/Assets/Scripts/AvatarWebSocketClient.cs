using System;
using System.Collections;
using System.IO;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

/// <summary>
/// WebSocket client for the Unity avatar endpoint (/api/avatar/ws).
///
/// Changes from the previous version:
///   - Fragmented receive: accumulates chunks into a MemoryStream until
///     EndOfMessage so large payloads are never truncated.
///   - Handles the new "landmark_file" reply (file name + HTTP URL only;
///     landmark data is fetched over HTTP, not pushed over WS).
///   - landmark_update always reloads (previously skipped when the file
///     name matched, which prevented re-playing the same gloss).
/// </summary>
public class AvatarWebSocketClient : MonoBehaviour
{
    [Header("Server Configuration")]
    public string serverUrl = "ws://localhost:8000/api/avatar/ws";

    [Header("References")]
    public HandPoseMapper handPoseMapper;

    [Header("Landmark HTTP fallback")]
    public string httpBaseUrl = "http://localhost:8000";

    private ClientWebSocket _webSocket;
    private CancellationTokenSource _cts;
    private bool _isConnected;
    private string _currentLandmarkFile;

    public event Action<string> OnLandmarkLoaded;
    public event Action<string> OnError;
    public event Action<bool> OnConnectionChanged;

    private async void Start()
    {
        if (handPoseMapper == null)
            handPoseMapper = GetComponent<HandPoseMapper>();

        await Connect();
    }

    private async Task Connect()
    {
        try
        {
            _webSocket = new ClientWebSocket();
            _cts = new CancellationTokenSource();

            await _webSocket.ConnectAsync(new Uri(serverUrl), _cts.Token);
            _isConnected = true;
            OnConnectionChanged?.Invoke(true);
            Debug.Log("[AvatarWS] Connected to server");

            _ = ReceiveLoop();
        }
        catch (Exception e)
        {
            Debug.LogError($"[AvatarWS] Connection failed: {e.Message}");
            OnError?.Invoke(e.Message);
            OnConnectionChanged?.Invoke(false);

            StartCoroutine(ReconnectAfterDelay(3f));
        }
    }

    private IEnumerator ReconnectAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        if (!_isConnected)
        {
            Debug.Log("[AvatarWS] Attempting reconnection...");
            _ = Connect();
        }
    }

    private async Task ReceiveLoop()
    {
        var buffer = new byte[1024 * 64];

        try
        {
            while (_webSocket.State == WebSocketState.Open && !_cts.Token.IsCancellationRequested)
            {
                // Accumulate fragmented messages until EndOfMessage.
                using (var messageStream = new MemoryStream())
                {
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await _webSocket.ReceiveAsync(
                            new ArraySegment<byte>(buffer), _cts.Token);

                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            await _webSocket.CloseAsync(
                                WebSocketCloseStatus.NormalClosure, "", _cts.Token);
                            _isConnected = false;
                            OnConnectionChanged?.Invoke(false);
                            return;
                        }

                        if (result.MessageType == WebSocketMessageType.Text)
                            messageStream.Write(buffer, 0, result.Count);
                    }
                    while (!result.EndOfMessage && _webSocket.State == WebSocketState.Open);

                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        string json = Encoding.UTF8.GetString(messageStream.ToArray());
                        HandleMessage(json);
                    }
                }
            }
        }
        catch (Exception e)
        {
            if (!_cts.Token.IsCancellationRequested)
            {
                Debug.LogError($"[AvatarWS] Receive error: {e.Message}");
                _isConnected = false;
                OnConnectionChanged?.Invoke(false);
                StartCoroutine(ReconnectAfterDelay(3f));
            }
        }
    }

    private void HandleMessage(string json)
    {
        try
        {
            var message = JsonUtility.FromJson<AvatarMessage>(json);

            if (message.type == "landmark_file")
            {
                // Server sends file name + URL only; load via mapper (StreamingAssets → HTTP).
                ApplyLandmark(message.landmark_file, force: true);
            }
            else if (message.type == "landmark_update")
            {
                // Composed (novel) sentences carry a per-gloss motion playlist
                // instead of a sentence file: play clips in signing order.
                // Sentence hits carry an empty playlist and play the file.
                if (message.playlist != null && message.playlist.Length > 0)
                    PlayPlaylist(message.playlist);
                else
                    ApplyLandmark(message.landmark_file, force: true);
            }
            else if (message.type == "error")
            {
                Debug.LogError($"[AvatarWS] Server error: {message.message}");
                OnError?.Invoke(message.message);
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"[AvatarWS] Parse error: {e.Message}");
        }
    }

    private void ApplyLandmark(string landmarkFile, bool force)
    {
        if (string.IsNullOrEmpty(landmarkFile))
            return;

        if (!force && landmarkFile == _currentLandmarkFile)
            return;

        _currentLandmarkFile = landmarkFile;

        if (handPoseMapper == null)
            handPoseMapper = GetComponent<HandPoseMapper>();
        if (handPoseMapper == null)
            handPoseMapper = GetComponentInChildren<HandPoseMapper>();

        if (handPoseMapper != null)
            handPoseMapper.LoadLandmarkFile(landmarkFile);

        OnLandmarkLoaded?.Invoke(landmarkFile);
        Debug.Log($"[AvatarWS] Landmark requested: {landmarkFile}");
    }

    /// <summary>
    /// Play a composed per-gloss motion playlist in signing order.
    /// URLs look like "/landmarks/gloss_x.json": the file name is carried
    /// to the mapper, which fetches each clip over HTTP in turn.
    /// </summary>
    private void PlayPlaylist(PlaylistItem[] playlist)
    {
        if (handPoseMapper == null)
            handPoseMapper = GetComponent<HandPoseMapper>();
        if (handPoseMapper == null)
            handPoseMapper = GetComponentInChildren<HandPoseMapper>();
        if (handPoseMapper == null)
        {
            Debug.LogError("[AvatarWS] No HandPoseMapper for playlist playback.");
            return;
        }

        var files = new System.Collections.Generic.List<string>();
        foreach (PlaylistItem item in playlist)
        {
            if (item == null)
                continue;
            string url = item.landmark_clip_url;
            if (string.IsNullOrEmpty(url))
                continue;
            int slash = url.LastIndexOf('/');
            files.Add(slash >= 0 ? url.Substring(slash + 1) : url);
        }
        handPoseMapper.PlayPlaylist(files);
        OnLandmarkLoaded?.Invoke($"playlist:{files.Count}");
        Debug.Log($"[AvatarWS] Playlist requested ({files.Count} clips).");
    }

    public async void RequestLandmark(string landmarkFile)
    {
        if (!_isConnected || _webSocket.State != WebSocketState.Open)
        {
            Debug.LogWarning("[AvatarWS] Not connected, cannot request landmark");
            return;
        }

        var request = new LandmarkRequest
        {
            action = "request_landmark",
            landmark_file = landmarkFile
        };

        string json = JsonUtility.ToJson(request);
        var bytes = Encoding.UTF8.GetBytes(json);

        try
        {
            await _webSocket.SendAsync(
                new ArraySegment<byte>(bytes),
                WebSocketMessageType.Text,
                true,
                _cts.Token);
        }
        catch (Exception e)
        {
            Debug.LogError($"[AvatarWS] Send error: {e.Message}");
        }
    }

    private async void OnApplicationQuit()
    {
        if (_webSocket != null && _webSocket.State == WebSocketState.Open)
        {
            _cts.Cancel();
            await _webSocket.CloseAsync(
                WebSocketCloseStatus.NormalClosure,
                "Application quitting",
                CancellationToken.None);
        }
    }

    [Serializable]
    private class AvatarMessage
    {
        public string type;
        public string landmark_file;
        public string landmark_url;
        public string message;
        public string glosses;
        public PlaylistItem[] playlist;
    }

    [Serializable]
    private class PlaylistItem
    {
        public string gloss;
        public string landmark_clip_url;
    }

    [Serializable]
    private class LandmarkRequest
    {
        public string action;
        public string landmark_file;
    }
}
