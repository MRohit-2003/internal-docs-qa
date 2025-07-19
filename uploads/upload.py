from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from pathlib import Path
import os
from ..src.vector_store import vector_store

from ..models import UploadResponse, DocumentMetadata
from ..src.utils import document_processor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["upload"])

# Update the upload_document function
@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file
        is_valid, message = document_processor.validate_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Save file
        file_path = await document_processor.save_uploaded_file(file)
        
        # Create metadata
        metadata = {
            "filename": file.filename,
            "file_type": Path(file.filename).suffix.lower(),
            "file_size": file.size or 0,
            "upload_time": datetime.now().isoformat()
        }
        
        # Process document into chunks
        chunks = document_processor.process_document(file_path, metadata)
        
        # Add chunks to vector store
        await vector_store.add_chunks(chunks)
        
        # Update metadata with chunk count
        doc_metadata = DocumentMetadata(
            filename=file.filename,
            file_type=Path(file.filename).suffix.lower(),
            file_size=file.size or 0,
            upload_time=datetime.now(),
            chunk_count=len(chunks)
        )
        
        logger.info(f"Successfully processed document: {file.filename} ({len(chunks)} chunks)")
        
        return UploadResponse(
            success=True,
            message=f"Document '{file.filename}' uploaded and processed successfully. Created {len(chunks)} chunks.",
            document_id=Path(file_path).stem,
            metadata=doc_metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/vector-stats")
async def get_vector_stats():
    """Get vector store statistics"""
    try:
        stats = vector_store.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting vector stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting vector store statistics")
