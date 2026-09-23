"""Endpoints: /api/translate schema, /api/animate, /api/health, validation."""


def test_translate_exact_lookup_schema(client, rag_module):
    resp = client.post(
        "/api/translate",
        json={"text": "hello how are you", "use_rag": True, "top_k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["input_text"] == "hello how are you"
    assert body["glosses"] == "HI HOW ARE_YOU"
    assert body["gloss_sequence"] == ["HI", "HOW", "ARE_YOU"]
    assert body["method"] == "direct_lookup"
    assert body["similarity"] == 1.0
    assert body["landmark_file"] == "hello_landmarks.json"
    assert body["landmark_url"] == "/landmarks/hello_landmarks.json"
    assert body["use_rag"] is True
    assert body["top_k"] == 5
    assert isinstance(body["retrieved_examples"], list)
    assert isinstance(body["animation"], dict)
    assert body["animation"]["clip_playlist"]


def test_translate_rag_disabled_skips_retrieval(client, rag_module):
    resp = client.post(
        "/api/translate",
        json={"text": "hello how are you", "use_rag": False},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert rag_module.calls == []
    assert body["retrieved_examples"] == []
    assert body["use_rag"] is False
    # exact dictionary hit still works without RAG
    assert body["method"] == "direct_lookup"


def test_translate_retrieves_when_rag_enabled(client, rag_module):
    resp = client.post(
        "/api/translate",
        json={"text": "hello how are you", "use_rag": True, "top_k": 3},
    )
    assert resp.status_code == 200
    assert rag_module.calls == [{"query": "hello how are you", "top_k": 3}]
    assert len(resp.json()["retrieved_examples"]) <= 3


def test_translate_top_k_bounds(client):
    assert (
        client.post("/api/translate", json={"text": "hi", "top_k": 0}).status_code
        == 422
    )
    assert (
        client.post("/api/translate", json={"text": "hi", "top_k": 21}).status_code
        == 422
    )


def test_translate_empty_text_rejected(client):
    resp = client.post("/api/translate", json={"text": "   "})
    assert resp.status_code == 400


def test_animate_resolves_known_and_reports_unknown(client):
    resp = client.post(
        "/api/animate",
        json={"gloss_sequence": ["HI", "NOPE", "ARE_YOU"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["resolved_tokens"] == ["HI", "ARE_YOU"]
    assert body["unresolved_tokens"] == ["NOPE"]
    assert len(body["clip_playlist"]) == 2
    assert body["clip_playlist"][0]["clip_id"] == "uid-hi"
    assert body["clip_playlist"][1]["landmark_file"] == "hello_landmarks.json"


def test_animate_empty_sequence_never_crashes(client):
    resp = client.post("/api/animate", json={"gloss_sequence": []})
    assert resp.status_code == 200
    assert resp.json()["clip_playlist"] == []


def test_health_endpoint_rich(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["database"] in ("connected", "disconnected")
    assert body["database_error"] or body["database"] == "connected"
    assert body["sentence_mappings"] == 3
    assert body["landmark_files"] >= 1
    assert "paths" in body
    assert body["paths"]["data_dir"].endswith("data")
    assert "faiss_index" in body
    # T5 weights are not in the repo
    assert body["t5_model"] in ("loaded", "not loaded")


def test_landmark_static_mount(client):
    resp = client.get("/landmarks/hello_landmarks.json")
    assert resp.status_code == 200
    assert "frames" in resp.json()
