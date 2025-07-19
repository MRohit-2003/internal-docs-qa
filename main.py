#!/usr/bin/env python3
"""
Internal Docs Q&A Agent - Main Application
FastAPI server with document upload and Q&A functionality
"""

import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules (to be created)

from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.qa_engine import QAEngine
from fastapi.concurrency import run_in_threadpool

# Initialize FastAPI app
app = FastAPI(
    title="Internal Docs Q&A Agent",
    description="AI-powered Q&A system for internal documentation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize components
doc_processor = DocumentProcessor()
vector_store = VectorStore()
qa_engine = QAEngine()

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with upload and chat interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Internal Docs Q&A Agent</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .chat-container { border: 1px solid #ccc; height: 400px; overflow-y: auto; padding: 10px; margin: 20px 0; }
            .input-area { display: flex; margin: 20px 0; }
            .input-area input { flex: 1; padding: 10px; margin-right: 10px; }
            .input-area button { padding: 10px 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Internal Docs Q&A Agent</h1>
            <p>Upload your documents and ask questions!</p>
            
            <div class="upload-area">
                <p>Drag & drop files here or click to upload</p>
                <input type="file" id="fileInput" multiple accept=".pdf,.txt,.docx">
                <button onclick="uploadFiles()">Upload Documents</button>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <p>👋 Hi! Upload some documents and I'll help you find answers to your questions.</p>
            </div>
            
            <div class="input-area">
                <input type="text" id="questionInput" placeholder="Ask a question about your documents...">
                <button onclick="askQuestion()">Ask</button>
            </div>
        </div>
        
        <script>
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput.files;
                
                if (files.length === 0) {
                    alert('Please select files to upload');
                    return;
                }
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    addMessage('system', `✅ Uploaded ${result.processed_files} documents successfully!`);
                } catch (error) {
                    addMessage('system', '❌ Error uploading files: ' + error.message);
                }
            }
            
            async function askQuestion() {
                const input = document.getElementById('questionInput');
                const question = input.value.trim();
                
                if (!question) return;
                
                addMessage('user', question);
                input.value = '';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    
                    const result = await response.json();
                    addMessage('assistant', result.answer);
                    
                    if (result.sources && result.sources.length > 0) {
                        addMessage('system', '📚 Sources: ' + result.sources.join(', '));
                    }
                } catch (error) {
                    addMessage('system', '❌ Error: ' + error.message);
                }
            }
            
            function addMessage(type, message) {
                const container = document.getElementById('chatContainer');
                const div = document.createElement('div');
                div.style.margin = '10px 0';
                div.style.padding = '10px';
                div.style.borderRadius = '5px';
                
                if (type === 'user') {
                    div.style.backgroundColor = '#e3f2fd';
                    div.innerHTML = '👤 <strong>You:</strong> ' + message;
                } else if (type === 'assistant') {
                    div.style.backgroundColor = '#f3e5f5';
                    div.innerHTML = '🤖 <strong>Assistant:</strong> ' + message;
                } else {
                    div.style.backgroundColor = '#fff3e0';
                    div.innerHTML = message;
                }
                
                container.appendChild(div);
                container.scrollTop = container.scrollHeight;
            }
            
            // Allow Enter key to submit question
            document.getElementById('questionInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload and process documents"""
    try:
        processed_files = 0
        for file in files:
            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Process document in threadpool
            text_chunks = await run_in_threadpool(doc_processor.process_document, file_path)

            # Store in vector database in threadpool
            await run_in_threadpool(vector_store.add_documents, text_chunks, file.filename or "unknown")

            processed_files += 1

        return {"message": "Documents uploaded successfully", "processed_files": processed_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question_data: dict):
    """Answer questions based on uploaded documents"""
    try:
        question = question_data.get("question", "")
        
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        
        # Get answer from QA engine
        result = qa_engine.get_answer(question, vector_store)
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
