import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from src.main import app
from langchain.schema import Document

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns expected response"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Raven Chat API", "version": "1.0.0"}


@patch('src.main.presenter')
@patch('src.main.retriever')
def test_chat_endpoint_success(mock_retriever, mock_presenter):
    """Test chat endpoint returns success response"""
    # Mock retriever results
    mock_results = [
        (Document(page_content="Actuator types comparison", metadata={"chunk_index": 0}), 0.85),
    ]
    mock_retriever.retrieve.return_value = mock_results
    
    # Mock presenter response
    mock_presenter.present.return_value = "The actuator types comparison shows [pneumatic and electric types]([1])."
    
    response = client.post(
        "/chat",
        json={"query": "Can you pull up the comparison of actuator types?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "actuator types comparison" in data["content"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["confidence"] == 0.85


def test_chat_invalid_request():
    """Test chat endpoint with invalid request"""
    response = client.post("/chat", json={})
    assert response.status_code == 422  # Validation error


@patch('src.main.presenter')
@patch('src.main.retriever')
def test_chat_response_structure(mock_retriever, mock_presenter):
    """Test that chat response has the correct structure"""
    # Mock retriever results
    mock_results = [
        (Document(page_content="Sample content", metadata={}), 0.75),
    ]
    mock_retriever.retrieve.return_value = mock_results
    
    # Mock presenter response
    mock_presenter.present.return_value = "Sample response with [citation]([1])."
    
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


@patch('src.main.presenter')
@patch('src.main.retriever')
def test_chat_with_retriever_success(mock_retriever, mock_presenter):
    """Test chat endpoint with successful retrieval"""
    # Mock retriever results
    mock_results = [
        (Document(page_content="Control valve sizing information", metadata={"chunk_index": 0}), 0.85),
        (Document(page_content="Actuator types comparison", metadata={"chunk_index": 1}), 0.75)
    ]
    mock_retriever.retrieve.return_value = mock_results
    
    # Mock presenter response
    mock_presenter.present.return_value = "Control valve sizing depends on [flow coefficient]([1]) and actuator types include [pneumatic]([2])."
    
    response = client.post(
        "/chat",
        json={"query": "Show me actuator types"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "Control valve sizing" in data["content"]
    assert len(data["citations"]) > 0
    assert data["citations"][0]["confidence"] == 0.85


@patch('src.main.retriever')
def test_chat_with_low_confidence(mock_retriever):
    """Test chat endpoint with low confidence results"""
    # Mock retriever results with low scores
    mock_results = [
        (Document(page_content="Some content", metadata={}), 0.3),
        (Document(page_content="Other content", metadata={}), 0.2)
    ]
    mock_retriever.retrieve.return_value = mock_results
    
    response = client.post(
        "/chat",
        json={"query": "Very specific query"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_info"
    assert "No confident matches" in data["content"]


@patch('src.main.retriever')
def test_chat_with_no_results(mock_retriever):
    """Test chat endpoint with no retrieval results"""
    mock_retriever.retrieve.return_value = []
    
    response = client.post(
        "/chat",
        json={"query": "Non-existent topic"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_info"
    assert "No relevant information" in data["content"]


@patch('src.main.presenter', None)
@patch('src.main.retriever', None)
def test_chat_retriever_not_initialized():
    """Test chat endpoint when retriever is not initialized"""
    response = client.post(
        "/chat",
        json={"query": "Test query"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_info"
    assert "Retriever or presenter not initialized" in data["content"]