import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from main import app
    return TestClient(app)


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert "version" in data


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_docs_available(client):
    resp = client.get("/docs")
    assert resp.status_code == 200


EXPECTED_PREFIXES = [
    "/api/jacobs/auth",
    "/api/jacobs/rag",
    "/api/jacobs/summary",
    "/api/jacobs/flashcards",
    "/api/jacobs/mindmap",
    "/api/jacobs/podcast",
    "/api/jacobs/presentation",
    "/api/jacobs/infographics",
    "/api/jacobs/parser",
    "/api/jacobs/table",
]


def test_all_routers_registered(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    all_paths = " ".join(paths.keys())
    for prefix in EXPECTED_PREFIXES:
        assert prefix in all_paths, f"Router prefix {prefix} not found in API paths"
