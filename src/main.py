from fastapi import FastAPI, HTTPException
from .models import ChatRequest, ChatResponse, Citation, BoundingBox
from typing import List

app = FastAPI(
    title="Raven Chat API",
    description="API for locating and extracting tables/images from control valve manual",
    version="1.0.0"
)


@app.get("/")
async def root():
    return {"message": "Raven Chat API", "version": "1.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(_request: ChatRequest):
    """
    Process natural language queries to find relevant tables/images in the control valve manual.
    
    Args:
        request: ChatRequest with query string
        
    Returns:
        ChatResponse with markdown content and citations
    """
    return ChatResponse(
        status="success",
        citations=[],
        content="Hello"
    )