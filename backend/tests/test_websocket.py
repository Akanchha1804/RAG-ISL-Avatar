"""Pipeline WebSocket stage order + avatar WebSocket file replies (HTTP data)."""


def test_pipeline_ws_stage_order(client, rag_module):
    with client.websocket_connect("/api/pipeline/ws") as ws:
        ws.send_json(
            {
                "input_mode": "text",
                "text": "hello how are you friend",
                "use_rag": True,
                "top_k": 3,
            }
        )
        stages = []
        last = None
        while True:
            msg = ws.receive_json()
            stages.append(msg.get("stage"))
            last = msg
            if msg.get("stage") == "complete":
                break

    assert stages == [
        "transcribing",
        "transcribing",
        "retrieving",
        "generating",
        "animating",
        "complete",
    ]
    assert last["status"] == "done"
    assert "gloss_sequence" in last["result"]
    assert last["result"]["use_rag"] is True
    assert rag_module.calls == [{"query": "hello how are you friend", "top_k": 3}]


def test_pipeline_ws_rag_disabled(client, rag_module):
    with client.websocket_connect("/api/pipeline/ws") as ws:
        ws.send_json(
            {"input_mode": "text", "text": "hello how are you", "use_rag": False}
        )
        while True:
            msg = ws.receive_json()
            if msg.get("stage") == "complete":
                break
    assert rag_module.calls == []


def test_pipeline_ws_empty_input_error_then_complete(client):
    with client.websocket_connect("/api/pipeline/ws") as ws:
        ws.send_json({"input_mode": "text", "text": "   ", "use_rag": True})
        messages = []
        while True:
            msg = ws.receive_json()
            messages.append(msg)
            if msg.get("stage") == "complete":
                break
    assert messages[-1]["stage"] == "complete"
    assert messages[-1]["status"] == "error"
    error_stages = [m for m in messages if m.get("status") == "error"]
    assert error_stages, "must send an error for the failing stage"


def test_avatar_ws_request_landmark_file_reply(client):
    """Avatar receives file name + HTTP URL only (no landmark payload)."""
    with client.websocket_connect("/api/avatar/ws") as ws:
        ws.send_json(
            {"action": "request_landmark", "landmark_file": "hello_landmarks.json"}
        )
        msg = ws.receive_json()
    assert msg["type"] == "landmark_file"
    assert msg["landmark_file"] == "hello_landmarks.json"
    assert msg["landmark_url"] == "/landmarks/hello_landmarks.json"
    assert "data" not in msg and "frames" not in msg


def test_avatar_ws_rejects_path_traversal(client):
    with client.websocket_connect("/api/avatar/ws") as ws:
        ws.send_json(
            {"action": "request_landmark", "landmark_file": "../../main.py"}
        )
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_avatar_ws_rejects_empty_name(client):
    with client.websocket_connect("/api/avatar/ws") as ws:
        ws.send_json({"action": "request_landmark", "landmark_file": ""})
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_avatar_ws_rejects_malformed_landmark(client):
    with client.websocket_connect("/api/avatar/ws") as ws:
        ws.send_json(
            {"action": "request_landmark", "landmark_file": "malformed_not_json.json"}
        )
        msg = ws.receive_json()
    assert msg["type"] == "error"


def test_pipeline_ws_broadcasts_landmark_update_to_avatar(client):
    """After a full pipeline run, connected avatars get landmark_file + URL."""
    with client.websocket_connect("/api/avatar/ws") as avatar:
        with client.websocket_connect("/api/pipeline/ws") as frontend:
            frontend.send_json(
                {
                    "input_mode": "text",
                    "text": "hello how are you",
                    "use_rag": True,
                }
            )
            while True:
                msg = frontend.receive_json()
                if msg.get("stage") == "complete":
                    break
        update = avatar.receive_json()
    assert update["type"] == "landmark_update"
    assert update["landmark_file"] == "hello_landmarks.json"
    assert update["landmark_url"] == "/landmarks/hello_landmarks.json"
    assert "frames" not in update
    assert update["playlist"] == []  # sentence level: file plays, no playlist


def test_pipeline_ws_broadcasts_playlist_for_composed(client, monkeypatch):
    """Composed results broadcast per-gloss motion clips for Unity sequencing."""
    import main

    async def fake_animation(seq, sentence_hit=None):
        assert sentence_hit is None
        return {
            "clip_playlist": [
                {"gloss": "A", "landmark_clip_url": "/landmarks/gloss_a.json",
                 "clip_url": None},
                {"gloss": "B", "landmark_clip_url": "",
                 "clip_url": "/clips/u.mp4"},
            ],
            "resolved_tokens": ["A", "B"],
            "unresolved_tokens": [],
            "unsupported_tokens": [],
            "suggestions": {},
            "retrieval_detail": {"level": "composed"},
        }

    # NOTE: stage_generate runs in an executor (sync); only the animation
    # stage is awaited directly, so this mock must stay sync.
    def fake_generate(*a, **k):
        return {"glosses": "A B", "gloss_sequence": ["A", "B"],
                "method": "word_by_word", "matched_sentence": None,
                "similarity": None, "landmark_file": "", "landmark_url": ""}

    monkeypatch.setattr(main, "stage_generate", fake_generate)
    monkeypatch.setattr(main, "stage_resolve_animation", fake_animation)

    with client.websocket_connect("/api/avatar/ws") as avatar:
        with client.websocket_connect("/api/pipeline/ws") as frontend:
            frontend.send_json({"input_mode": "text", "text": "a b"})
            while True:
                msg = frontend.receive_json()
                if msg.get("stage") == "complete":
                    break
        update = avatar.receive_json()
    assert update["landmark_file"] == ""
    # Only motion-carrying entries ship; reference-only clips stay out.
    assert update["playlist"] == [
        {"gloss": "A", "landmark_clip_url": "/landmarks/gloss_a.json"}
    ]
