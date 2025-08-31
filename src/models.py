from pydantic import BaseModel
from typing import List, Dict, Optional


class ChatRequest(BaseModel):
    query: str


class BoundingBox(BaseModel):
    top_left_x: int
    top_left_y: int
    width: int
    height: int


class Citation(BaseModel):
    page_no: int
    bbox: BoundingBox
    confidence: Optional[float] = None


class ChatResponse(BaseModel):
    content: str
    citations: List[Citation]
    status: Optional[str] = "success"  # Can be "success" or "insufficient_info"