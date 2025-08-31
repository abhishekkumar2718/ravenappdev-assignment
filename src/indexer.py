#!/usr/bin/env python3
"""
Indexer script for building a FAISS vector store from markdown documents.
This script processes markdown files, creates semantic chunks, generates embeddings,
and saves a searchable vector index.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

from dotenv import load_dotenv
from langchain.text_splitter import MarkdownTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()


class Indexer:
    """Indexes markdown documents for semantic search using FAISS and Gemini embeddings."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the indexer with Gemini API key.
        
        Args:
            api_key: Google Gemini API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY environment variable")
        

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=self.api_key
        )
        
    def load_markdown(self, file_path: str) -> str:
        """
        Load markdown content from file.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            Content of the markdown file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_chunks(self, text: str, source_file: str = "manual_chapter1.mmd", chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """
        Split markdown text into chunks using markdown-aware splitting.
        
        Args:
            text: Markdown text to split
            source_file: Name of the source file
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of Document objects containing chunks
        """
        # Use markdown text splitter for better markdown structure preservation
        splitter = MarkdownTextSplitter()
        
        chunks = splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "chunk_index": i,
                    "source": source_file,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
            )
            documents.append(doc)
        
        return documents
    
    def save_chunks(self, documents: List[Document], output_path: str) -> None:
        """
        Save chunks to a file for reuse.
        
        Args:
            documents: List of Document objects to save
            output_path: Path to save the chunks file
        """
        chunks_data = {
            "documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ],
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(documents)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"Chunks saved to: {output_path}")
    
    def load_chunks(self, chunks_path: str) -> Optional[List[Document]]:
        """
        Load previously saved chunks from file.
        
        Args:
            chunks_path: Path to the chunks file
            
        Returns:
            List of Document objects or None if file doesn't exist
        """
        if not os.path.exists(chunks_path):
            return None
        
        try:
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            documents = [
                Document(
                    page_content=doc_data["page_content"],
                    metadata=doc_data["metadata"]
                )
                for doc_data in chunks_data["documents"]
            ]
            
            print(f"Loaded {len(documents)} chunks from: {chunks_path}")
            return documents
            
        except Exception as e:
            print(f"Error loading chunks: {e}")
            return None
    
    def build_index(self, documents: List[Document]) -> FAISS:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            FAISS vector store
        """
        print(f"Building FAISS index from {len(documents)} chunks...")
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        return vectorstore
    
    def save_index(self, vectorstore: FAISS, output_dir: str = None) -> str:
        """
        Save FAISS index to disk with timestamp.
        
        Args:
            vectorstore: FAISS vector store to save
            output_dir: Base directory for saving (default: data/)
            
        Returns:
            Path to the saved index directory
        """
        if output_dir is None:
            output_dir = "data"
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(output_dir, f"vector_store_{timestamp}")
        
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        vectorstore.save_local(save_path)
        
        # Save metadata
        metadata = {
            "created_at": timestamp,
            "source_file": "manual_chapter1.mmd",
            "embedding_model": "models/gemini-embedding-001",
            "num_documents": len(vectorstore.docstore._dict)
        }
        
        with open(os.path.join(save_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Index saved to: {save_path}")
        return save_path
    
    def index_file(self, file_path: str, use_cached_chunks: bool = True, chunk_size: int = 1000, chunk_overlap: int = 200) -> str:
        """
        Complete indexing pipeline for a markdown file.
        
        Args:
            file_path: Path to the markdown file to index
            use_cached_chunks: Whether to use cached chunks if available
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            Path to the saved index
        """
        base_name = os.path.basename(file_path).replace('.mmd', '')
        chunks_cache_path = f"data/chunks_{base_name}.json"
        
        # Try to load cached chunks if requested
        documents = None
        if use_cached_chunks:
            print(f"Checking for cached chunks at: {chunks_cache_path}")
            documents = self.load_chunks(chunks_cache_path)
        
        # If no cached chunks, create new ones
        if documents is None:
            print(f"Loading markdown file: {file_path}")
            text = self.load_markdown(file_path)
            
            print("Creating markdown-based chunks...")
            documents = self.create_chunks(text, source_file=os.path.basename(file_path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print(f"Created {len(documents)} chunks")
            
            # Save chunks for future use
            os.makedirs("data", exist_ok=True)
            self.save_chunks(documents, chunks_cache_path)
        
        print("Building vector index...")
        vectorstore = self.build_index(documents)
        
        print("Saving index...")
        save_path = self.save_index(vectorstore)
        
        return save_path


def main():
    """Main function to run the indexer from command line."""
    parser = argparse.ArgumentParser(description="Index markdown documents for semantic search")
    parser.add_argument(
        "--input", 
        type=str, 
        default="manual_chapter1.mmd",
        help="Path to the markdown file to index"
    )
    parser.add_argument(
        "--no-cache", 
        action="store_true",
        help="Don't use cached chunks, create new ones"
    )
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Target size of each chunk in characters"
    )
    parser.add_argument(
        "--chunk-overlap", 
        type=int, 
        default=200,
        help="Number of characters to overlap between chunks"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Google API key (can also be set via GOOGLE_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    try:
        indexer = Indexer(api_key=args.api_key)
        save_path = indexer.index_file(
            args.input,
            use_cached_chunks=not args.no_cache,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print(f"\nIndexing complete! Index saved to: {save_path}")
        
    except Exception as e:
        print(f"Error during indexing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())