"""Background DB persistence must be non-fatal when the DB is unreachable."""

import asyncio


def test_log_translation_background_swallows_connection_error():
    from crud import log_translation_background

    # DATABASE_URL points at 127.0.0.1:1 (nothing listening).
    # Must not raise — log failure is non-fatal for the API.
    asyncio.run(
        log_translation_background(
            input_text="hello",
            output_glosses="HI",
            method="test",
        )
    )


def test_translate_endpoint_succeeds_when_db_down(client, rag_module):
    resp = client.post("/api/translate", json={"text": "hello how are you"})
    assert resp.status_code == 200
    assert resp.json()["glosses"] == "HI HOW ARE_YOU"


def test_history_endpoint_degrades_gracefully(client):
    resp = client.get("/api/history")
    assert resp.status_code == 200
    body = resp.json()
    assert "history" in body
    # either connected (unlikely with test URL) or note/error payload
    if body.get("note"):
        assert "error" in body or body["total"] == 0
