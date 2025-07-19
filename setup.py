#!/usr/bin/env python3
"""
Setup script for Internal Docs Q&A Agent
Use this script to set up your development environment in VSCode
"""
import os
import sys
import subprocess
from pathlib import Path
import json

def create_directory_structure():
    """Create the required directory structure"""
    directories = [
        'uploads',
        'templates',
        'static',
        'static/css',
        'static/js',
        'src',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_env_file():
    """Create a sample .env file"""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=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

# Application Settings
DEBUG=True
HOST=0.0.0.0
PORT=8000

# File Processing Settings
MAX_FILE_SIZE=52428800
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Storage Settings
VECTOR_DIMENSION=1536
MAX_DOCUMENTS=1000
"""
    
    env_path = Path('.env')
    if not env_path.exists():
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✓ Created .env file - Please add your OpenAI API key!")
    else:
        print("✓ .env file already exists")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
jinja2==3.1.2
python-dotenv==1.0.0
openai==1.3.0
faiss-cpu==1.7.4
numpy==1.24.3
pandas==2.0.3
PyPDF2==3.0.1
python-docx==0.8.11
tiktoken==0.5.1
langchain==0.0.335
langchain-openai==0.0.2
pydantic==2.5.0
httpx==0.25.2
aiofiles==23.2.1
"""
    
    req_path = Path('requirements.txt')
    if not req_path.exists():
        with open(req_path, 'w', encoding='utf-8') as f:
            f.write(requirements)
        print("✓ Created requirements.txt")
    else:
        print("✓ requirements.txt already exists")

def install_dependencies():
    """Install required Python packages"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False
    return True

def create_vscode_settings():
    """Create VSCode settings for the project"""
    vscode_dir = Path('.vscode')
    vscode_dir.mkdir(exist_ok=True)
    
    # Settings
    settings = {
        "python.defaultInterpreterPath": "./venv/bin/python",
        "python.linting.enabled": True,
        "python.linting.pylintEnabled": True,
        "python.formatting.provider": "black",
        "python.analysis.typeCheckingMode": "basic",
        "files.exclude": {
            "**/__pycache__": True,
            "**/*.pyc": True,
            ".env": True,
            "uploads/*": True,
            "*.log": True,
            "faiss_index*": True,
            "embeddings.pkl": True
        },
        "files.watcherExclude": {
            "**/uploads/**": True,
            "**/__pycache__/**": True
        }
    }
    
    with open(vscode_dir / 'settings.json', 'w', encoding='utf-8') as f:
        json.dump(settings, f, indent=2)
    
    # Launch configuration
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Run FastAPI Server",
                "type": "python",
                "request": "launch",
                "program": "main.py",
                "console": "integratedTerminal",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}"
                },
                "args": []
            },
            {
                "name": "Debug FastAPI Server",
                "type": "python",
                "request": "launch",
                "program": "main.py",
                "console": "integratedTerminal",
                "env": {
                    "PYTHONPATH": "${workspaceFolder}",
                    "DEBUG": "True"
                },
                "args": ["--reload"]
            }
        ]
    }
    
    with open(vscode_dir / 'launch.json', 'w', encoding='utf-8') as f:
        json.dump(launch_config, f, indent=2)
    
    # Tasks configuration
    tasks_config = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "Start Development Server",
                "type": "shell",
                "command": "uvicorn",
                "args": ["main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
                "group": {
                    "kind": "build",
                    "isDefault": True
                },
                "presentation": {
                    "echo": True,
                    "reveal": "always",
                    "focus": False,
                    "panel": "shared"
                },
                "problemMatcher": []
            },
            {
                "label": "Install Dependencies",
                "type": "shell",
                "command": "pip",
                "args": ["install", "-r", "requirements.txt"],
                "group": "build"
            }
        ]
    }
    
    with open(vscode_dir / 'tasks.json', 'w', encoding='utf-8') as f:
        json.dump(tasks_config, f, indent=2)
    
    print("✓ Created VSCode configuration files")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Project specific
uploads/
*.log
faiss_index*
embeddings.pkl
static/uploaded_files/
vector_store/

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary files
*.tmp
*.temp
~*
"""
    
    gitignore_path = Path('.gitignore')
    if not gitignore_path.exists():
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore file")
    else:
        print("✓ .gitignore file already exists")

def create_main_app_structure():
    """Create the main application files"""
    
    # Main FastAPI app
    main_py = """#!/usr/bin/env python3
\"\"\"
Internal Docs Q&A Agent - Main Application
FastAPI server with document upload and Q&A functionality
\"\"\"

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
    \"\"\"Home page with upload and chat interface\"\"\"
    return \"\"\"
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
    \"\"\"

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    \"\"\"Upload and process documents\"\"\"
    try:
        processed_files = 0
        
        for file in files:
            # Save uploaded file
            file_path = f"uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Process document
            text_chunks = doc_processor.process_document(file_path)
            
            # Store in vector database
            vector_store.add_documents(text_chunks, file.filename)
            
            processed_files += 1
        
        return {"message": "Documents uploaded successfully", "processed_files": processed_files}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question_data: dict):
    \"\"\"Answer questions based on uploaded documents\"\"\"
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
"""
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(main_py)
    print("✓ Created main.py")

def create_src_modules():
    """Create the source modules"""
    
    # Create __init__.py
    with open('src/__init__.py', 'w', encoding='utf-8') as f:
        f.write('"""Internal Docs Q&A Agent - Source Modules"""\n')
    
    # Document Processor
    doc_processor = '''"""
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
                    text += page.extract_text() + "\\n"
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
                text += paragraph.text + "\\n"
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
        paragraphs = text.split('\\n\\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk + "\\n\\n" + paragraph if current_chunk else paragraph
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
'''
    
    with open('src/document_processor.py', 'w', encoding='utf-8') as f:
        f.write(doc_processor)
    
    # Vector Store
    vector_store = '''"""
Vector Store Module
Handles document embeddings and similarity search using FAISS
"""

import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
import faiss
from openai import OpenAI

class VectorStore:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.dimension = 1536  # OpenAI embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        self.embeddings_cache = "embeddings.pkl"
        self.index_cache = "faiss_index.bin"
        
        # Load existing data if available
        self._load_existing_data()
    
    def add_documents(self, text_chunks: List[Dict], source_name: str):
        """Add documents to the vector store"""
        print(f"Adding {len(text_chunks)} chunks from {source_name}")
        
        for chunk in text_chunks:
            # Generate embedding
            embedding = self._get_embedding(chunk["content"])
            
            # Add to FAISS index
            self.index.add(np.array([embedding]))
            
            # Store document metadata
            self.documents.append({
                "content": chunk["content"],
                "source": source_name,
                "chunk_id": chunk["chunk_id"]
            })
        
        # Save updated data
        self._save_data()
        print(f"Successfully added documents. Total documents: {len(self.documents)}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self._get_embedding(query)
        
        # Search in FAISS
        scores, indices = self.index.search(np.array([query_embedding]), k)
        
        # Return results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                result = self.documents[idx].copy()
                result["score"] = float(score)
                result["rank"] = i + 1
                results.append(result)
        
        return results
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    
    def _save_data(self):
        """Save vector store data to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, self.index_cache)
            
            # Save documents metadata
            with open(self.embeddings_cache, 'wb') as f:
                pickle.dump(self.documents, f)
                
        except Exception as e:
            print(f"Error saving vector store data: {e}")
    
    def _load_existing_data(self):
        """Load existing vector store data from disk"""
        try:
            if os.path.exists(self.index_cache) and os.path.exists(self.embeddings_cache):
                # Load FAISS index
                self.index = faiss.read_index(self.index_cache)
                
                # Load documents metadata
                with open(self.embeddings_cache, 'rb') as f:
                    self.documents = pickle.load(f)
                
                print(f"Loaded existing data: {len(self.documents)} documents")
        except Exception as e:
            print(f"Error loading existing data: {e}")
            # Initialize empty structures
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []
'''
    
    with open('src/vector_store.py', 'w', encoding='utf-8') as f:
        f.write(vector_store)
    
    # Q&A Engine
    qa_engine = '''"""
Q&A Engine Module
Handles question answering using RAG (Retrieval-Augmented Generation)
"""

import os
from typing import Dict, List
from openai import OpenAI

class QAEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-3.5-turbo"
    
    def get_answer(self, question: str, vector_store) -> Dict:
        """Get answer to question using RAG approach"""
        
        # Step 1: Retrieve relevant documents
        relevant_docs = vector_store.search(question, k=5)
        
        if not relevant_docs:
            return {
                "answer": "I don't have any relevant documents to answer your question. Please upload some documents first.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Step 2: Prepare context from retrieved documents
        context = self._prepare_context(relevant_docs)
        
        # Step 3: Generate answer using OpenAI
        answer = self._generate_answer(question, context)
        
        # Step 4: Extract sources
        sources = list(set([doc["source"] for doc in relevant_docs[:3]]))
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": self._calculate_confidence(relevant_docs)
        }
    
    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i+1} (from {doc['source']}):")
            context_parts.append(doc["content"])
            context_parts.append("")  # Empty line for separation
        
        return "\\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using OpenAI GPT"""
        
        prompt = f"""You are a helpful assistant that answers questions based on provided documents. 
        
Use the following context to answer the question. If the context doesn't contain enough information to answer the question, say so clearly.

Context:
{context}

Question: {question}

Answer: """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context. Be concise but comprehensive."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _calculate_confidence(self, documents: List[Dict]) -> float:
        """Calculate confidence score based on retrieval scores"""
        if not documents:
            return 0.0
        
        # Simple confidence based on best match score
        # Lower scores are better in FAISS L2 distance
        best_score = documents[0]["score"]
        
        # Convert to confidence (higher is better)
        # This is a simple heuristic - you might want to improve this
        if best_score < 0.5:
            return 0.9
        elif best_score < 1.0:
            return 0.7
        elif best_score < 2.0:
            return 0.5
        else:
            return 0.3
'''
    
    with open('src/qa_engine.py', 'w', encoding='utf-8') as f:
        f.write(qa_engine)
    
    print("✓ Created source modules")

def create_readme():
    """Create README.md file"""
    readme_content = """# Internal Docs Q&A Agent

A fast AI-powered Q&A system for internal documentation built with FastAPI and OpenAI.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   python setup.py
   ```

2. **Add OpenAI API Key**
   - Edit `.env` file and add your OpenAI API key
   
3. **Run the Application**
   ```bash
   python main.py
   ```
   
4. **Open Browser**
   - Visit `http://localhost:8000`

## 📁 Project Structure

```
internal-docs-qa/
├── main.py              # FastAPI application
├── src/                 # Source modules
│   ├── document_processor.py
│   ├── vector_store.py
│   └── qa_engine.py
├── uploads/             # Uploaded documents
├── static/              # Static files (CSS, JS)
├── templates/           # HTML templates
├── requirements.txt     # Python dependencies
└── .env                # Environment variables
```

## 🛠️ Features

- **Document Upload**: PDF, DOCX, TXT support
- **Smart Q&A**: RAG-powered question answering
- **Source Attribution**: Shows which documents were used
- **Real-time Processing**: Fast document ingestion
- **Web Interface**: Simple, clean UI

## 🔧 Configuration

Edit `.env` file:

```env
OPENAI_API_KEY=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
MAX_FILE_SIZE=52428800
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 📚 Usage

1. Upload your documents using the web interface
2. Ask questions about the content
3. Get AI-powered answers with source citations

## 🧪 Development

- Run with auto-reload: `uvicorn main:app --reload`
- Debug mode: Set `DEBUG=True` in `.env`
- VSCode: Use provided launch configuration

## 📦 Dependencies

- FastAPI for web framework
- OpenAI for embeddings and chat completion
- FAISS for vector similarity search
- PyPDF2, python-docx for document processing

## 🎯 Hackathon Timeline

This project is designed to be built in 3 days:
- Day 1: Core setup and document ingestion
- Day 2: Q&A engine and web interface
- Day 3: Polish and demo preparation

Built for hackathons - optimized for rapid development and deployment!
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✓ Created README.md")

def main():
    """Main setup function"""
    print("🚀 Setting up Internal Docs Q&A Agent...")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Create configuration files
    create_env_file()
    create_requirements_file()
    create_gitignore()
    
    # Create VSCode configuration
    create_vscode_settings()
    
    # Create application files
    create_main_app_structure()
    create_src_modules()
    create_readme()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Add your OpenAI API key to the .env file")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the app: python main.py")
    print("4. Open http://localhost:8000 in your browser")
    print("\n🎯 You're ready for Day 2 development!")
    print("Happy coding! 🚀")

if __name__ == "__main__":
    main()