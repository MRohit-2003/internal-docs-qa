from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DocumentMetadata(BaseModel):
    """Document metadata model"""
    filename: str
    file_type: str
    file_size: int
    upload_time: datetime
    chunk_count: Optional[int] = 0
    
class DocumentChunk(BaseModel):
    """Document chunk model"""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    chunk_index: int
    
class UploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool
    message: str
    document_id: Optional[str] = None
    metadata: Optional[DocumentMetadata] = None