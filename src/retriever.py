#!/usr/bin/env python3
"""
Retriever module for searching the FAISS vector store using MMR.
"""

import os
from typing import List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

load_dotenv()


class Retriever:
    """Retrieves relevant documents from FAISS vector store using MMR."""
    
    VECTOR_STORE_PATH = "data/vector_store_20250901_193259"
    
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
        
        # Load Whoosh index if available
        self.whoosh_index = None
        whoosh_path = os.path.join(self.VECTOR_STORE_PATH, "whoosh_index")
        if os.path.exists(whoosh_path):
            try:
                self.whoosh_index = open_dir(whoosh_path)
                print("Loaded Whoosh index for hybrid search")
            except Exception as e:
                print(f"Warning: Could not load Whoosh index: {e}")
                print("Falling back to vector-only search")
        
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents using hybrid search (vector + keyword).
        
        Args:
            query: The search query
            k: Number of documents to retrieve (default: 10)
            
        Returns:
            List of tuples containing (Document, score)
        """
        try:
            # Increase fetch size for re-ranking
            fetch_k = 25
            
            # Use MMR for diversity across all chunk types
            mmr_docs = self.vector_store.max_marginal_relevance_search(
                query,
                k=fetch_k,
                fetch_k=50,  # Fetch more candidates for MMR to select from
                lambda_mult=0.5  # Balance between relevance and diversity
            )
            
            # Get similarity scores for the same query
            similarity_results = self.vector_store.similarity_search_with_score(query, k=fetch_k*2)
            
            # Create a mapping of content to vector score
            content_to_vector_score = {doc.page_content: score for doc, score in similarity_results}
            
            # If Whoosh index is available, perform hybrid scoring
            if self.whoosh_index:
                # Get keyword scores
                keyword_scores = self._get_keyword_scores(query, mmr_docs)
                
                # Combine scores
                results = []
                for doc in mmr_docs:
                    vector_score = content_to_vector_score.get(doc.page_content, 0.5)
                    keyword_score = keyword_scores.get(doc.metadata.get('chunk_id', ''), 0.0)
                    
                    # Hybrid score: 50% vector, 50% keyword
                    hybrid_score = 0.5 * vector_score + 0.5 * keyword_score
                    results.append((doc, hybrid_score))
                
                # Sort by hybrid score
                results.sort(key=lambda x: x[1], reverse=True)
                
                # Take top k results
                results = results[:k]
            else:
                # Fallback to vector-only search
                results = []
                for doc in mmr_docs[:k]:
                    score = content_to_vector_score.get(doc.page_content, 0.5)
                    results.append((doc, score))
            
            # Expand results with related entities
            expanded_results = self._expand_with_entities(results)
            
            return expanded_results
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
    
    def _get_keyword_scores(self, query: str, documents: List[Document]) -> dict:
        """
        Get keyword scores for documents using Whoosh.
        
        Args:
            query: The search query
            documents: List of documents to score
            
        Returns:
            Dictionary mapping chunk_id to keyword score (0-1)
        """
        keyword_scores = {}
        
        try:
            with self.whoosh_index.searcher() as searcher:
                # Parse query
                qp = QueryParser("content", self.whoosh_index.schema)
                q = qp.parse(query)
                
                # Search and get scores
                results = searcher.search(q, limit=100)
                
                # Normalize scores (Whoosh scores can be > 1)
                max_score = max([result.score for result in results]) if results else 1.0
                
                # Create score mapping
                for result in results:
                    chunk_id = result['chunk_id']
                    normalized_score = result.score / max_score if max_score > 0 else 0
                    keyword_scores[chunk_id] = normalized_score
        
        except Exception as e:
            print(f"Error getting keyword scores: {e}")
        
        return keyword_scores
    
    def _expand_with_entities(self, results: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        """Expand results with related entities."""
        expanded = []
        seen_ids = set()
        
        for doc, score in results:
            # Add the main document
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen_ids:
                expanded.append((doc, score))
                seen_ids.add(chunk_id)
            
            # If it's a section with entities, fetch them
            if doc.metadata.get("chunk_type") == "section":
                entity_ids = doc.metadata.get("entity_ids", [])
                for entity_id in entity_ids[:2]:  # Limit to top 2 entities per section
                    # Search for the entity
                    entity_results = self.vector_store.similarity_search_with_score(
                        entity_id, k=1
                    )
                    for entity_doc, _ in entity_results:
                        if entity_doc.metadata.get("chunk_id") == entity_id:
                            if entity_id not in seen_ids:
                                # Give related entities a slightly lower score
                                expanded.append((entity_doc, score * 0.8))
                                seen_ids.add(entity_id)
                            break
        
        # Sort by score
        expanded.sort(key=lambda x: x[1], reverse=True)
        return expanded