"""
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
            # Ensure embedding is a numpy array of shape (1, dimension) and dtype float32
            embedding_np = np.array(embedding, dtype=np.float32).reshape(1, -1)
            # Add to FAISS index
            # FAISS add() expects the argument as a positional parameter named 'x'
            # FAISS add() expects the embedding as the first positional argument (x)
            # Defensive: ensure no shadowing and call as positional
            add_fn = getattr(self.index, "add", None)
            if callable(add_fn):
                add_fn(embedding_np)
            else:
                raise RuntimeError("FAISS index does not have an 'add' method.")
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
        # Ensure query_embedding is a float32 numpy array of shape (1, dimension)
        query_embedding_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        # Defensive: ensure no shadowing and call as positional
        search_fn = getattr(self.index, "search", None)
        if callable(search_fn):
            search_result = search_fn(query_embedding_np, k)
            # For IndexFlatL2, search returns (distances, indices)
            if isinstance(search_result, tuple) and len(search_result) == 2:
                scores, indices = search_result
            else:
                raise RuntimeError("Unexpected return type from FAISS search(); expected a tuple (distances, indices).")
        else:
            raise RuntimeError("FAISS index does not have a 'search' method.")
        
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
