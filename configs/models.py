from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, Dict, Any
from enum import Enum


class ChatAnswer(BaseModel):
    """Validate chat answer type and length."""
    answer: Annotated[str, Field(min_length=1, max_length=4096)]


class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"


# 🔥 Upload Response (multi-doc aware)
class UploadResponse(BaseModel):
    session_id: str
    indexed: bool
    documents_indexed: int = 0   # 👈 NEW
    message: Optional[str] = None


# 🔥 Chat Request (can support future controls)
class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: Optional[int] = 5   # 👈 optional override for retrieval


# 🔥 Source Metadata (VERY IMPORTANT for RAG)
class SourceDocument(BaseModel):
    source: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


# 🔥 Chat Response (RAG enhanced)
class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDocument]] = None  # 👈 NEW
    metadata: Optional[Dict[str, Any]] = None       # 👈 NEW (debug/info)
