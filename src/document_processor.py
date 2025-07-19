"""
Document Processing Module
Handles PDF, DOCX, and TXT file processing and text chunking
"""

import os
from pathlib import Path
from typing import List, Dict
import PyPDF2
import docx
import tiktoken

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def process_document(self, file_path: str) -> List[Dict]:
        """Process a document and return text chunks"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_extension == '.docx':
            text = self._extract_docx_text(file_path)
        elif file_extension == '.txt':
            text = self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Split into chunks
        chunks = self._split_text(text)
        
        # Return chunks with metadata
        return [
            {
                "content": chunk,
                "source": os.path.basename(file_path),
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return ""
        return text
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {e}")
            return ""
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count"""
        if not text:
            return []
        
        # Simple paragraph-based splitting for now
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = len(self.encoding.encode(potential_chunk))
            
            if token_count <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk.strip()]
