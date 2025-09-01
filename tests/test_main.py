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
        (Document(
            page_content="Actuator types comparison",
            metadata={
                "chunk_index": 0,
                "page": 15,
                "bbox": {
                    "top_left_x": 100,
                    "top_left_y": 200,
                    "width": 500,
                    "height": 300
                }
            }
        ), 0.85),
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
        (Document(
            page_content="Sample content",
            metadata={
                "chunk_index": 0,
                "page": 10,
                "bbox": {
                    "top_left_x": 150,
                    "top_left_y": 250,
                    "width": 600,
                    "height": 400
                }
            }
        ), 0.75),
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
        (Document(
            page_content="Control valve sizing information",
            metadata={
                "chunk_index": 0,
                "page": 7,
                "bbox": {
                    "top_left_x": 249,
                    "top_left_y": 688,
                    "width": 757,
                    "height": 47
                }
            }
        ), 0.85),
        (Document(
            page_content="Actuator types comparison",
            metadata={
                "chunk_index": 1,
                "page": 9,
                "bbox": {
                    "top_left_x": 249,
                    "top_left_y": 634,
                    "width": 744,
                    "height": 41
                }
            }
        ), 0.75)
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
        (Document(
            page_content="Some content",
            metadata={
                "chunk_index": 0,
                "page": 5,
                "bbox": {
                    "top_left_x": 100,
                    "top_left_y": 100,
                    "width": 400,
                    "height": 200
                }
            }
        ), 0.3),
        (Document(
            page_content="Other content",
            metadata={
                "chunk_index": 1,
                "page": 6,
                "bbox": {
                    "top_left_x": 120,
                    "top_left_y": 150,
                    "width": 450,
                    "height": 250
                }
            }
        ), 0.2)
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


@patch('src.main.presenter')
@patch('src.main.retriever')
def test_chat_citations_with_bbox_and_page(mock_retriever, mock_presenter):
    """Test that citations correctly include page numbers and bounding boxes from metadata"""
    # Mock retriever results with complete metadata
    mock_results = [
        (Document(
            page_content="Test content with bbox",
            metadata={
                "chunk_index": 0,
                "source": "manual_chapter1.mmd",
                "page": 42,
                "bbox": {
                    "top_left_x": 300,
                    "top_left_y": 400,
                    "width": 800,
                    "height": 600
                }
            }
        ), 0.9),
        (Document(
            page_content="Another test content",
            metadata={
                "chunk_index": 1,
                "source": "manual_chapter2.mmd",
                "page": 55,
                "bbox": {
                    "top_left_x": 150,
                    "top_left_y": 200,
                    "width": 900,
                    "height": 700
                }
            }
        ), 0.8)
    ]
    mock_retriever.retrieve.return_value = mock_results
    
    # Mock presenter response
    mock_presenter.present.return_value = "Test response with [citation]([1])."
    
    response = client.post(
        "/chat",
        json={"query": "Test query for bbox and page"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Check first citation
    assert len(data["citations"]) >= 2
    citation1 = data["citations"][0]
    assert citation1["page_no"] == 42
    assert citation1["bbox"]["top_left_x"] == 300
    assert citation1["bbox"]["top_left_y"] == 400
    assert citation1["bbox"]["width"] == 800
    assert citation1["bbox"]["height"] == 600
    assert citation1["confidence"] == 0.9
    assert citation1["chunk_id"] == "manual_chapter1.mmd_chunk_000"
    
    # Check second citation
    citation2 = data["citations"][1]
    assert citation2["page_no"] == 55
    assert citation2["bbox"]["top_left_x"] == 150
    assert citation2["bbox"]["top_left_y"] == 200
    assert citation2["bbox"]["width"] == 900
    assert citation2["bbox"]["height"] == 700
    assert citation2["confidence"] == 0.8
    assert citation2["chunk_id"] == "manual_chapter2.mmd_chunk_001"


@patch('src.main.presenter')
@patch('src.main.retriever')
def test_chat_citations_with_missing_metadata(mock_retriever, mock_presenter):
    """Test that citations handle missing metadata gracefully with defaults"""
    # Mock retriever results with partial/missing metadata
    mock_results = [
        (Document(
            page_content="Content without bbox",
            metadata={
                "chunk_index": 0,
                "page": 25
                # bbox is missing
            }
        ), 0.85),
        (Document(
            page_content="Content without page",
            metadata={
                "chunk_index": 1,
                # page is missing
                "bbox": {
                    "top_left_x": 50,
                    "top_left_y": 100,
                    "width": 200,
                    "height": 150
                }
            }
        ), 0.75),
        (Document(
            page_content="Content with empty metadata",
            metadata={}  # All metadata missing
        ), 0.65)
    ]
    mock_retriever.retrieve.return_value = mock_results
    
    # Mock presenter response
    mock_presenter.present.return_value = "Response with partial metadata."
    
    response = client.post(
        "/chat",
        json={"query": "Test with missing metadata"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    
    # Check citations handle missing data with defaults
    assert len(data["citations"]) >= 3
    
    # First citation: has page but no bbox
    citation1 = data["citations"][0]
    assert citation1["page_no"] == 25
    assert citation1["bbox"]["top_left_x"] == 0
    assert citation1["bbox"]["top_left_y"] == 0
    assert citation1["bbox"]["width"] == 0
    assert citation1["bbox"]["height"] == 0
    
    # Second citation: has bbox but no page
    citation2 = data["citations"][1]
    assert citation2["page_no"] == 1  # Default page
    assert citation2["bbox"]["top_left_x"] == 50
    assert citation2["bbox"]["top_left_y"] == 100
    assert citation2["bbox"]["width"] == 200
    assert citation2["bbox"]["height"] == 150
    
    # Third citation: no page or bbox
    citation3 = data["citations"][2]
    assert citation3["page_no"] == 1  # Default page
    assert citation3["bbox"]["top_left_x"] == 0
    assert citation3["bbox"]["top_left_y"] == 0
    assert citation3["bbox"]["width"] == 0
    assert citation3["bbox"]["height"] == 0