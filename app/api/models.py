from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi import UploadFile, File

class DocumentIn(BaseModel):
    content: str = Field(..., description="The document content to be added")

class DocumentsIn(BaseModel):
    documents: List[DocumentIn] = Field(..., description="List of documents to add")

class QueryIn(BaseModel):
    query: str = Field(..., description="The query to process")
    max_results: Optional[int] = Field(default=5, description="Maximum number of results to return")

class DocumentOut(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    citation_number: int

class QueryResult(BaseModel):
    original_query: str
    expanded_query: str
    results: List[DocumentOut]
    answer: Optional[str] = None

class StatusResponse(BaseModel):
    status: str
    message: str