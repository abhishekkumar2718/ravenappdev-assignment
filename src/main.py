from fastapi import FastAPI, HTTPException
from .models import ChatRequest, ChatResponse, Citation, BoundingBox
from .retriever import Retriever
from .presenter import Presenter
from typing import List
import os

app = FastAPI(
    title="Raven Chat API",
    description="API for locating and extracting tables/images from control valve manual",
    version="1.0.0"
)

# Initialize retriever and presenter
retriever = None
presenter = None
try:
    retriever = Retriever()
    presenter = Presenter()
except Exception as e:
    print(f"Warning: Could not initialize retriever/presenter: {e}")


@app.get("/")
async def root():
    return {"message": "Raven Chat API", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process natural language queries to find relevant tables/images in the control valve manual.
    
    Args:
        request: ChatRequest with query string
        
    Returns:
        ChatResponse with markdown content and citations
    """
    if not retriever or not presenter:
        return ChatResponse(
            status="insufficient_info",
            citations=[],
            content="Retriever or presenter not initialized. Please check the vector store and API key."
        )
    
    # Retrieve relevant documents
    results = retriever.retrieve(request.query)
    
    if not results:
        return ChatResponse(
            status="insufficient_info",
            citations=[],
            content="No relevant information found for your query."
        )
    
    # Check confidence threshold (using the best score)
    best_score = results[0][1] if results else 0
    
    if best_score < 0.5:
        return ChatResponse(
            status="insufficient_info",
            citations=[],
            content="No confident matches found for your query."
        )
    
    # Generate response using presenter
    content = presenter.present(request.query, results)
    
    # Create citations from results
    citations = []
    for i, (doc, score) in enumerate(results[:5]):  # Use top 5 results
        # Extract chunk_id from metadata
        chunk_index = doc.metadata.get('chunk_index', i)
        source = doc.metadata.get('source', 'manual')
        chunk_id = f"{source}_chunk_{chunk_index:03d}"
        
        # Extract page number and bounding box from metadata
        page_no = doc.metadata.get('page', 1)  # Default to page 1 if not found
        bbox_data = doc.metadata.get('bbox', {})
        
        # Create BoundingBox from metadata
        bbox = BoundingBox(
            top_left_x=bbox_data.get('top_left_x', 0),
            top_left_y=bbox_data.get('top_left_y', 0),
            width=bbox_data.get('width', 0),
            height=bbox_data.get('height', 0)
        )
        
        citations.append(Citation(
            page_no=page_no,
            bbox=bbox,
            confidence=score,
            chunk_id=chunk_id
        ))
    
    return ChatResponse(
        status="success",
        citations=citations,
        content=content
    )