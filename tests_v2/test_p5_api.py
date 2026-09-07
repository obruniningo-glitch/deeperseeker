"""P5 tests: API endpoints, health, admin dashboard (synchronous)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from starlette.testclient import TestClient

from v2.api.app import create_app


@pytest.fixture
def app():
    app = create_app()
    # Override settings for testing
    os.environ["DEEPSEEKER_API_KEY"] = "test-key"
    os.environ["DEEPSEEKER_DB_PATH"] = "test_p5.db"
    yield create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def cleanup():
    yield
    db_path = "test_p5.db"
    import time
    time.sleep(0.1)  # allow connections to close
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass  # Windows file locking


def test_health_endpoint(client):
    """Health endpoint returns token pool and session info."""
    response = client.get("/health", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert "tokens" in data
    assert "cached_sessions" in data
    assert "cookie_valid" in data
    assert data["tokens"]["total"] >= 1  # default token


def test_models_endpoint(client):
    """Models endpoint returns available models."""
    response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "instant" in model_ids
    assert "expert" in model_ids
    assert "vision" in model_ids


def test_chat_completions_non_stream(client):
    """Non-streaming chat completion returns valid response."""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "fake",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_stream(client):
    """Streaming chat completion returns SSE."""
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "fake",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_admin_tokens_crud(client):
    """Admin token CRUD operations."""
    # List (should have default token)
    response = client.get("/admin/tokens", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    tokens = response.json()
    assert len(tokens) >= 1

    # Add
    response = client.post(
        "/admin/tokens",
        headers={"Authorization": "Bearer test-key"},
        json={"alias": "test-token", "secret": "sk-test-123", "provider": "deepseek"},
    )
    assert response.status_code == 200
    token = response.json()
    assert token["alias"] == "test-token"
    assert token["status"] == "ACTIVE"
    token_id = token["id"]

    # Delete
    response = client.delete(f"/admin/tokens/{token_id}", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200


def test_admin_sessions(client):
    """Admin sessions list."""
    response = client.get("/admin/sessions", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_admin_usage(client):
    """Admin usage stats."""
    response = client.get("/admin/usage", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)


def test_admin_prune_sessions(client):
    """Admin prune sessions."""
    response = client.post(
        "/admin/prune-sessions",
        headers={"Authorization": "Bearer test-key"},
        json={"ttl_days": 0.001},  # very short TTL
    )
    assert response.status_code == 200
    data = response.json()
    assert "deleted" in data


def test_unauthorized(client):
    """Endpoints reject invalid auth."""
    response = client.get("/health")
    assert response.status_code == 401

    response = client.get("/health", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401