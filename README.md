# Internal Docs Q&A Agent

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
