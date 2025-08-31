#!/usr/bin/env python3
"""
Presenter module for generating concise responses using Gemini-2.0-flash.
"""

import os
from typing import List, Tuple
from langchain.schema import Document
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()


class Presenter:
    """Generates concise, matter-of-fact responses using Gemini-2.0-flash."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the presenter with Gemini API.
        
        Args:
            api_key: Google Gemini API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def present(self, query: str, results: List[Tuple[Document, float]]) -> str:
        """
        Generate a concise response based on retrieved documents.
        
        Args:
            query: The original user query
            results: List of (Document, score) tuples from retriever
            
        Returns:
            Markdown formatted response with inline citations
        """
        if not results:
            return "No relevant information found for your query."
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(results[:5]):  # Use top 5 results
            context_parts.append(f"[Citation {i+1}] (confidence: {score:.2f}):\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt for Gemini
        prompt = f"""You are a technical assistant helping technicians with control valve queries. 
Based ONLY on the provided citations, answer the user's question concisely and accurately.

CRITICAL INSTRUCTIONS:
1. Use ONLY information present in the citations below
2. Be concise and matter-of-fact
3. Reference citations inline using markdown link syntax like [this information]([1])
4. If information is not in the citations, state clearly that it's not available
5. Do not add any information not present in the citations
6. Focus on accuracy - incorrect answers can be life-harming

User Query: {query}

Citations:
{context}

Provide a concise response using markdown format with inline citation links:"""
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response."