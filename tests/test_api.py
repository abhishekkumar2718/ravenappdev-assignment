import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns expected response"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Raven Chat API", "version": "1.0.0"}


def test_chat_endpoint_success():
    """Test chat endpoint returns success response"""
    response = client.post(
        "/chat",
        json={"query": "Can you pull up the comparison of actuator types?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["content"] == "Hello"
    assert data["citations"] == []


def test_chat_invalid_request():
    """Test chat endpoint with invalid request"""
    response = client.post("/chat", json={})
    assert response.status_code == 422  # Validation error


def test_chat_response_structure():
    """Test that chat response has the correct structure"""
    response = client.post(
        "/chat",
        json={"query": "Show me actuator types"}
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check required fields exist
    assert "status" in data
    assert "content" in data
    assert "citations" in data
    
    # Check field types
    assert isinstance(data["status"], str)
    assert isinstance(data["content"], str)
    assert isinstance(data["citations"], list)