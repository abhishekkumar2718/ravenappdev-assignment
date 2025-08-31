#!/usr/bin/env python3
"""
Retriever module for searching the FAISS vector store using MMR.
"""

import os
from typing import List, Tuple, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()


class Retriever:
    """Retrieves relevant documents from FAISS vector store using MMR."""
    
    VECTOR_STORE_PATH = "data/vector_store_20250831_150902"
    
    def __init__(self, api_key: str = None):
        """
        Initialize the retriever with FAISS index and embeddings.
        
        Args:
            api_key: Google Gemini API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.api_key
        )
        
        # Load FAISS index
        if not os.path.exists(self.VECTOR_STORE_PATH):
            raise FileNotFoundError(f"Vector store not found at {self.VECTOR_STORE_PATH}")
        
        self.vector_store = FAISS.load_local(
            self.VECTOR_STORE_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents using MMR (Maximal Marginal Relevance).
        
        Args:
            query: The search query
            k: Number of documents to retrieve (default: 10)
            
        Returns:
            List of tuples containing (Document, score)
        """
        try:
            # Use similarity search with score instead of MMR with score (not available)
            # First get MMR results for diversity
            mmr_docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=20,  # Fetch more candidates for MMR to select from
                lambda_mult=0.5  # Balance between relevance and diversity
            )
            
            # Then get similarity scores for the same query
            similarity_results = self.vector_store.similarity_search_with_score(query, k=k*2)
            
            # Create a mapping of content to score
            content_to_score = {doc.page_content: score for doc, score in similarity_results}
            
            # Combine MMR results with scores
            results = []
            for doc in mmr_docs:
                score = content_to_score.get(doc.page_content, 0.5)  # Default score if not found
                results.append((doc, score))
            
            return results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []