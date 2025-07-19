"""
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
        
        return "\n".join(context_parts)
    
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
            # Defensive: check response structure and content
            if (
                hasattr(response, "choices") and response.choices and
                hasattr(response.choices[0], "message") and
                hasattr(response.choices[0].message, "content")
            ):
                content = response.choices[0].message.content
                if content is not None:
                    return content.strip()
                else:
                    return "Error: OpenAI API returned no content."
            else:
                return "Error: Unexpected response format from OpenAI API."
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
