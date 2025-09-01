from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum


class ChunkType(str, Enum):
    SECTION = "section"
    TABLE = "table"
    IMAGE = "image"


class ChunkMetadata(BaseModel):
    chunk_id: str
    chunk_type: ChunkType
    section_title: str
    section_path: List[str]  # e.g., ["Chapter 1", "Control Valve Selection", "Ball Valves"]
    entity_ids: List[str] = []  # e.g., ["table_1-2", "figure_1-6"]
    page_number: int
    line_numbers: tuple[int, int]  # (start, end)


class EntityRegistry(BaseModel):
    entities: Dict[str, ChunkMetadata] = {}  # Maps entity_id to metadata
    section_entities: Dict[str, List[str]] = {}  # Maps section_id to list of entity_ids


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
    chunk_id: Optional[str] = None
    chunk_type: Optional[ChunkType] = None


class ChatResponse(BaseModel):
    content: str
    citations: List[Citation]
    status: Optional[str] = "success"  # Can be "success" or "insufficient_info"