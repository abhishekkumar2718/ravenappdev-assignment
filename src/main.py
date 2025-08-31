from fastapi import FastAPI, HTTPException
from .models import ChatRequest, ChatResponse, Citation, BoundingBox
from .retriever import Retriever
from typing import List
import os

app = FastAPI(
    title="Raven Chat API",
    description="API for locating and extracting tables/images from control valve manual",
    version="1.0.0"
)

# Initialize retriever
retriever = None
try:
    retriever = Retriever()
except Exception as e:
    print(f"Warning: Could not initialize retriever: {e}")


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
    if not retriever:
        return ChatResponse(
            status="insufficient_info",
            citations=[],
            content="Retriever not initialized. Please check the vector store and API key."
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
    
    # Combine retrieved content
    content_parts = []
    citations = []
    
    for i, (doc, score) in enumerate(results[:5]):  # Use top 5 results
        content_parts.append(f"**Result {i+1} (score: {score:.3f}):**\n{doc.page_content}\n")
        
        # Add placeholder citation
        citations.append(Citation(
            page_no=1,  # Placeholder
            bbox=BoundingBox(
                top_left_x=100,
                top_left_y=100,
                width=500,
                height=300
            ),
            confidence=score
        ))
    
    content = "\n".join(content_parts)
    
    return ChatResponse(
        status="success",
        citations=citations,
        content=content
    )