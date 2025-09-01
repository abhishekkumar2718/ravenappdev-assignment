#!/usr/bin/env python3
"""
Presenter module for generating concise responses using Gemini-2.0-flash.
"""

import os
import re
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
        
        # Prepare context from retrieved documents with entity awareness
        context_parts = []
        table_citations = []
        figure_citations = []
        
        for i, (doc, score) in enumerate(results[:5]):  # Use top 5 results
            chunk_type = doc.metadata.get('chunk_type', 'section')
            chunk_id = doc.metadata.get('chunk_id', f'chunk_{i}')
            
            # Format citation label based on entity type
            if chunk_type == 'table':
                # Extract table number if available
                table_match = re.search(r'Table (\d+-\d+)', doc.page_content)
                if table_match:
                    citation_label = f"Table {table_match.group(1)}"
                else:
                    citation_label = chunk_id
                table_citations.append((i+1, citation_label))
            elif chunk_type == 'image':
                # Extract figure number if available
                figure_match = re.search(r'Figure (\d+-\d+)', doc.page_content)
                if figure_match:
                    citation_label = f"Figure {figure_match.group(1)}"
                else:
                    citation_label = chunk_id
                figure_citations.append((i+1, citation_label))
            else:
                section_title = doc.metadata.get('section_title', 'Section')
                citation_label = f"{section_title}"
            
            context_parts.append(
                f"[Citation {i+1}] ({chunk_type}, confidence: {score:.2f}, ID: {citation_label}):\n"
                f"{doc.page_content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Determine if results contain specific entity types
        has_tables = any(doc.metadata.get('chunk_type') == 'table' for doc, _ in results[:5])
        has_figures = any(doc.metadata.get('chunk_type') == 'image' for doc, _ in results[:5])
        
        # Create entity-aware prompt instructions
        entity_instructions = ""
        if has_tables:
            entity_instructions += "\n7. When presenting table data, format it clearly and include the table identifier (e.g., Table 1-2)"
        if has_figures:
            entity_instructions += "\n8. When referencing figures, describe them clearly and include the figure identifier (e.g., Figure 1-6)"
        
        prompt = f"""You are a technical assistant helping technicians with control valve queries. 
Based ONLY on the provided citations, answer the user's question concisely and accurately.

CRITICAL INSTRUCTIONS:
1. Use ONLY information present in the citations below
2. Be concise and matter-of-fact
3. Reference citations inline using markdown link syntax like [this information]([1])
4. If information is not in the citations, state clearly that it's not available
5. Do not add any information not present in the citations
6. Focus on accuracy - incorrect answers can be life-harming{entity_instructions}

User Query: {query}

Citations:
{context}

Note: Citations may include sections, tables, and figures. Pay attention to the citation type and ID.

Provide a concise response using markdown format with inline citation links:"""
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response."