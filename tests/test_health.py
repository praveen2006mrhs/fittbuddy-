"""Unit and integration tests for FitBuddy health and home routes."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a FastAPI test client instance."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient):
    """Test that GET /health returns 200 and valid JSON status."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app"] == "FitBuddy"
    assert "environment" in data
    assert "database" in data
    assert data["database"]["type"] == "sqlite"
    assert "gemini" in data
    assert "workout_model" in data["gemini"]
    assert "tip_model" in data["gemini"]
    assert "configured" in data["gemini"]
    assert "mock_enabled" in data["gemini"]
    assert "mode" in data["gemini"]
    assert "timeout_seconds" in data["gemini"]
    assert "max_retries" in data["gemini"]


def test_home_page(client: TestClient):
    """Test that GET / returns 200 and renders FitBuddy HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "FitBuddy" in response.text
    assert "Personalized Workout Plans" in response.text
    assert "id=\"main-heading\"" in response.text


def test_static_css_accessible(client: TestClient):
    """Test that static CSS assets are served correctly."""
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
