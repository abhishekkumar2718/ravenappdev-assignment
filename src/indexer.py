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
from typing import List, Dict, Any, Optional, Tuple
import pickle
import re

from dotenv import load_dotenv
from langchain.text_splitter import MarkdownTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from .models import ChunkType, ChunkMetadata, EntityRegistry

load_dotenv()


class DocumentLocationMapper:
    """Maps document chunks to their page numbers and bounding boxes."""
    
    def __init__(self, mmd_lines_data: Dict[str, Any]):
        """
        Initialize the mapper with MMD lines data.
        
        Args:
            mmd_lines_data: Dictionary containing pages and line information
        """
        self.mmd_lines_data = mmd_lines_data
        self.reset_position()
    
    def reset_position(self):
        """Reset the search position to the beginning."""
        self.current_page_idx = 0
        self.current_line_idx = 0
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for matching by removing extra whitespace and newlines."""
        return ' '.join(text.split()).strip()
    
    def extract_first_content_line(self, chunk_text: str) -> str:
        """Extract the first meaningful content line, skipping headers."""
        lines = chunk_text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Skip LaTeX commands and headers
            if line.startswith('\\section') or line.startswith('\\begin') or line.startswith('\\caption'):
                continue
            # Return first actual content line
            return line
        
        return ""
    
    def find_location(self, document_text: str) -> Optional[Dict[str, Any]]:
        """
        Find page and bounding box for a document chunk.
        
        Args:
            document_text: The document chunk text
            
        Returns:
            Dictionary with 'page' and 'bbox' or None if not found
        """
        # Special handling for figures/tables
        if document_text.strip().startswith('\\begin{figure}') or document_text.strip().startswith('\\begin{table}'):
            return self._find_figure_or_table_location(document_text)
        
        # Get first content line
        first_line = self.extract_first_content_line(document_text)
        if not first_line:
            return None
            
        # Normalize for matching
        normalized_first_line = self.normalize_text(first_line)
        
        # Take just the first 30-50 characters for matching
        # This helps when OCR splits sentences across lines
        search_text = normalized_first_line[:40] if len(normalized_first_line) > 40 else normalized_first_line
        
        # Two-pointer search starting from last position
        pages = self.mmd_lines_data.get('pages', [])
        
        for page_idx in range(self.current_page_idx, len(pages)):
            page_data = pages[page_idx]
            lines = page_data.get('lines', [])
            
            start_line_idx = self.current_line_idx if page_idx == self.current_page_idx else 0
            
            for line_idx in range(start_line_idx, len(lines)):
                line_data = lines[line_idx]
                line_text = self.normalize_text(line_data.get('text', ''))
                
                # Check if the search text appears in this line
                if search_text in line_text:
                    # Update pointers for next search
                    self.current_page_idx = page_idx
                    self.current_line_idx = line_idx
                    
                    return {
                        'page': page_data.get('page'),
                        'bbox': line_data.get('region')
                    }
                
                # Also check if this line is the start of our text
                # (handles case where our chunk starts mid-line in the OCR)
                if line_text and search_text.startswith(line_text):
                    # Check if the next line continues our text
                    if line_idx + 1 < len(lines):
                        next_line = self.normalize_text(lines[line_idx + 1].get('text', ''))
                        combined = line_text + ' ' + next_line
                        if search_text in combined:
                            self.current_page_idx = page_idx
                            self.current_line_idx = line_idx
                            
                            return {
                                'page': page_data.get('page'),
                                'bbox': line_data.get('region')
                            }
        
        return None
    
    def _find_figure_or_table_location(self, document_text: str) -> Optional[Dict[str, Any]]:
        """Special handling for figures and tables."""
        # Look for Figure X-X or Table X-X patterns
        figure_match = re.search(r'Figure \d+-\d+', document_text)
        table_match = re.search(r'Table \d+-\d+', document_text)
        
        search_text = figure_match.group(0) if figure_match else (table_match.group(0) if table_match else None)
        
        if not search_text:
            return None
            
        # Search for this specific figure/table
        pages = self.mmd_lines_data.get('pages', [])
        
        for page_idx in range(self.current_page_idx, len(pages)):
            page_data = pages[page_idx]
            
            for line_idx, line_data in enumerate(page_data.get('lines', [])):
                if search_text in line_data.get('text', ''):
                    self.current_page_idx = page_idx
                    self.current_line_idx = line_idx
                    
                    return {
                        'page': page_data.get('page'),
                        'bbox': line_data.get('region')
                    }
        
        return None


class SectionEntityChunker:
    """Chunks documents into sections, tables, and images with entity relationships."""
    
    def __init__(self, location_mapper: Optional[DocumentLocationMapper] = None):
        """
        Initialize the chunker with optional location mapper.
        
        Args:
            location_mapper: DocumentLocationMapper for page/bbox information
        """
        self.location_mapper = location_mapper
        self.entity_registry = EntityRegistry()
        self.current_section_path = []
        self.current_section_id = None
        
    def chunk_document(self, text: str, source_file: str = "manual.mmd") -> List[Document]:
        """
        Chunk document into sections, tables, and images.
        
        Args:
            text: The document text to chunk
            source_file: Name of the source file
            
        Returns:
            List of Document objects with entity-aware metadata
        """
        documents = []
        
        # Reset location mapper if available
        if self.location_mapper:
            self.location_mapper.reset_position()
        
        # Split by sections
        sections = self._split_sections(text)
        
        for section_idx, section_content in enumerate(sections):
            # Extract section title and update path
            section_title = self._extract_section_title(section_content)
            self._update_section_path(section_title)
            
            # Generate section ID
            section_id = f"section_{section_idx}"
            self.current_section_id = section_id
            
            # Extract tables and figures from section
            tables = self._extract_tables(section_content)
            figures = self._extract_figures(section_content)
            
            # Create entity IDs
            table_ids = []
            figure_ids = []
            
            # Process tables
            for table_idx, (table_content, table_id_match) in enumerate(tables):
                table_id = table_id_match if table_id_match else f"table_{section_idx}_{table_idx}"
                table_ids.append(table_id)
                
                # Create table document
                table_doc = self._create_entity_document(
                    content=table_content,
                    chunk_id=table_id,
                    chunk_type=ChunkType.TABLE,
                    section_title=section_title,
                    source_file=source_file
                )
                documents.append(table_doc)
                
                # Remove table from section content
                section_content = section_content.replace(table_content, f"[Table {table_id}]")
            
            # Process figures
            for fig_idx, (figure_content, figure_id_match) in enumerate(figures):
                figure_id = figure_id_match if figure_id_match else f"figure_{section_idx}_{fig_idx}"
                figure_ids.append(figure_id)
                
                # Create figure document
                figure_doc = self._create_entity_document(
                    content=figure_content,
                    chunk_id=figure_id,
                    chunk_type=ChunkType.IMAGE,
                    section_title=section_title,
                    source_file=source_file
                )
                documents.append(figure_doc)
                
                # Remove figure from section content
                section_content = section_content.replace(figure_content, f"[Figure {figure_id}]")
            
            # Create section document with entity references
            section_doc = self._create_entity_document(
                content=section_content,
                chunk_id=section_id,
                chunk_type=ChunkType.SECTION,
                section_title=section_title,
                source_file=source_file,
                entity_ids=table_ids + figure_ids
            )
            documents.append(section_doc)
            
            # Update entity registry
            self.entity_registry.section_entities[section_id] = table_ids + figure_ids
        
        return documents
    
    def _split_sections(self, text: str) -> List[str]:
        """Split text by section markers."""
        # Split by \section* markers
        sections = re.split(r'\\section\*\{', text)
        
        # Process each section
        processed_sections = []
        for section in sections:
            if section.strip():
                # Add back the section marker if it was removed
                if not section.startswith('\\section*{'):
                    section = '\\section*{' + section
                processed_sections.append(section)
        
        return processed_sections if processed_sections else [text]
    
    def _extract_section_title(self, section_content: str) -> str:
        """Extract section title from content."""
        match = re.match(r'\\section\*\{([^}]+)\}', section_content)
        if match:
            return match.group(1).strip()
        return "Untitled Section"
    
    def _update_section_path(self, section_title: str) -> None:
        """Update the hierarchical section path."""
        # Simple heuristic: Chapter resets path, other sections append
        if "Chapter" in section_title:
            self.current_section_path = [section_title]
        else:
            # Keep only top 2 levels for simplicity
            if len(self.current_section_path) >= 2:
                self.current_section_path = [self.current_section_path[0], section_title]
            else:
                self.current_section_path.append(section_title)
    
    def _extract_tables(self, content: str) -> List[Tuple[str, Optional[str]]]:
        """Extract tables from content. Returns list of (table_content, table_id)."""
        tables = []
        
        # Find all tables
        table_pattern = r'\\begin\{table\}.*?\\end\{table\}'
        table_matches = re.finditer(table_pattern, content, re.DOTALL)
        
        for match in table_matches:
            table_content = match.group(0)
            
            # Try to extract table ID from caption
            caption_match = re.search(r'Table (\d+-\d+)', table_content)
            table_id = f"table_{caption_match.group(1).replace('-', '_')}" if caption_match else None
            
            tables.append((table_content, table_id))
        
        return tables
    
    def _extract_figures(self, content: str) -> List[Tuple[str, Optional[str]]]:
        """Extract figures from content. Returns list of (figure_content, figure_id)."""
        figures = []
        
        # Find all figures
        figure_pattern = r'\\begin\{figure\}.*?\\end\{figure\}'
        figure_matches = re.finditer(figure_pattern, content, re.DOTALL)
        
        for match in figure_matches:
            figure_content = match.group(0)
            
            # Try to extract figure ID from caption
            caption_match = re.search(r'Figure (\d+-\d+)', figure_content)
            figure_id = f"figure_{caption_match.group(1).replace('-', '_')}" if caption_match else None
            
            figures.append((figure_content, figure_id))
        
        return figures
    
    def _create_entity_document(self, content: str, chunk_id: str, chunk_type: ChunkType,
                              section_title: str, source_file: str,
                              entity_ids: List[str] = None) -> Document:
        """Create a Document with entity-aware metadata."""
        # Get location information if mapper available
        location_info = {}
        if self.location_mapper:
            location = self.location_mapper.find_location(content)
            if location:
                location_info = {
                    "page": location["page"],
                    "bbox": location["bbox"]
                }
        
        # Create metadata
        metadata = {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type.value,
            "section_title": section_title,
            "section_path": self.current_section_path.copy(),
            "entity_ids": entity_ids or [],
            "source": source_file,
            **location_info
        }
        
        # Store in entity registry
        chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            chunk_type=chunk_type,
            section_title=section_title,
            section_path=self.current_section_path.copy(),
            entity_ids=entity_ids or [],
            page_number=location_info.get("page", 1),
            line_numbers=(0, 0)  # Would need to enhance location mapper for this
        )
        self.entity_registry.entities[chunk_id] = chunk_metadata
        
        return Document(page_content=content, metadata=metadata)
    
    def get_entity_registry(self) -> EntityRegistry:
        """Return the entity registry."""
        return self.entity_registry


class Indexer:
    """Indexes markdown documents for semantic search using FAISS and Gemini embeddings."""
    
    def __init__(self, api_key: str = None, mmd_lines_path: str = None):
        """
        Initialize the indexer with Gemini API key.
        
        Args:
            api_key: Google Gemini API key. If not provided, reads from GOOGLE_API_KEY env var.
            mmd_lines_path: Path to mmd_lines_data.json file for location mapping
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        # Only initialize embeddings if we have a real API key
        self.embeddings = None
        if self.api_key and self.api_key != "dummy-key-for-chunk-only":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.api_key
            )
        
        # Load mmd_lines_data if provided
        self.location_mapper = None
        if mmd_lines_path and os.path.exists(mmd_lines_path):
            with open(mmd_lines_path, 'r', encoding='utf-8') as f:
                mmd_lines_data = json.load(f)
                self.location_mapper = DocumentLocationMapper(mmd_lines_data)
        
        # Initialize entity registry
        self.entity_registry = None
        
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
    
    def create_chunks(self, text: str, source_file: str = "manual_chapter1.mmd", chunk_size: int = 1000, chunk_overlap: int = 200, use_entity_chunking: bool = True) -> List[Document]:
        """
        Split markdown text into chunks using entity-aware or markdown-aware splitting.
        
        Args:
            text: Markdown text to split
            source_file: Name of the source file
            chunk_size: Target size of each chunk in characters (for old method)
            chunk_overlap: Number of characters to overlap between chunks (for old method)
            use_entity_chunking: Whether to use new entity-aware chunking
            
        Returns:
            List of Document objects containing chunks
        """
        if use_entity_chunking:
            # Use new section-entity chunker
            chunker = SectionEntityChunker(location_mapper=self.location_mapper)
            documents = chunker.chunk_document(text, source_file)
            
            # Save entity registry for later use
            self.entity_registry = chunker.get_entity_registry()
            
            return documents
        else:
            # Use old markdown text splitter for backward compatibility
            splitter = MarkdownTextSplitter()
            
            chunks = splitter.split_text(text)
            
            # Reset location mapper position for new chunking session
            if self.location_mapper:
                self.location_mapper.reset_position()
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "chunk_index": i,
                    "source": source_file,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
                
                # Add location information if mapper is available
                if self.location_mapper:
                    location = self.location_mapper.find_location(chunk)
                    if location:
                        metadata["page"] = location["page"]
                        metadata["bbox"] = location["bbox"]
                
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
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
        
        # Add entity registry if available
        if self.entity_registry:
            chunks_data["entity_registry"] = {
                "entities": {k: v.model_dump() for k, v in self.entity_registry.entities.items()},
                "section_entities": self.entity_registry.section_entities
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
            
            # Load entity registry if available
            if "entity_registry" in chunks_data:
                registry_data = chunks_data["entity_registry"]
                self.entity_registry = EntityRegistry()
                
                # Reconstruct entities
                for entity_id, entity_dict in registry_data["entities"].items():
                    chunk_metadata = ChunkMetadata(**entity_dict)
                    self.entity_registry.entities[entity_id] = chunk_metadata
                
                # Reconstruct section_entities
                self.entity_registry.section_entities = registry_data["section_entities"]
            
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
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Please provide a valid API key.")
            
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
    
    def chunk_file(self, file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, preview: bool = True) -> List[Document]:
        """
        Create chunks from a markdown file without building the vector index.
        
        Args:
            file_path: Path to the markdown file to chunk
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            preview: Whether to print chunk previews
            
        Returns:
            List of Document objects containing chunks
        """
        print(f"Loading markdown file: {file_path}")
        text = self.load_markdown(file_path)
        
        print("Creating markdown-based chunks...")
        documents = self.create_chunks(text, source_file=os.path.basename(file_path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"Created {len(documents)} chunks")
        
        if preview:
            print("\n" + "="*80)
            print("CHUNK PREVIEW")
            print("="*80 + "\n")
            
            for i, doc in enumerate(documents):
                print(f"--- Chunk {i + 1}/{len(documents)} ---")
                print(f"Size: {doc.metadata['chunk_size']} characters")
                
                # Show location info if available
                if 'page' in doc.metadata:
                    print(f"Page: {doc.metadata['page']}")
                    bbox = doc.metadata.get('bbox', {})
                    print(f"Bounding Box: x={bbox.get('top_left_x')}, y={bbox.get('top_left_y')}, w={bbox.get('width')}, h={bbox.get('height')}")
                
                # Show chunk type if available
                if 'chunk_type' in doc.metadata:
                    print(f"Chunk Type: {doc.metadata['chunk_type']}")
                if 'section_path' in doc.metadata:
                    print(f"Section Path: {' > '.join(doc.metadata['section_path'])}")
                if 'entity_ids' in doc.metadata and doc.metadata['entity_ids']:
                    print(f"Referenced Entities: {', '.join(doc.metadata['entity_ids'])}")
                
                # Show first 500 characters of content
                preview_text = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                print(f"Content preview:\n{preview_text}")
                print()
        
        # Save chunks
        base_name = os.path.basename(file_path).replace('.mmd', '')
        chunks_cache_path = f"data/chunks_{base_name}.json"
        os.makedirs("data", exist_ok=True)
        self.save_chunks(documents, chunks_cache_path)
        
        return documents


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
    parser.add_argument(
        "--mmd-lines",
        type=str,
        default="mmd_lines_data.json",
        help="Path to mmd_lines_data.json file for location mapping"
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Only create chunks without building vector index"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Skip chunk preview in chunk-only mode"
    )
    
    args = parser.parse_args()
    
    try:
        if args.chunk_only:
            # Chunk-only mode doesn't require API key
            indexer = Indexer(api_key=args.api_key or "dummy-key-for-chunk-only", mmd_lines_path=args.mmd_lines)
            indexer.chunk_file(
                args.input,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                preview=not args.no_preview
            )
            print(f"\nChunking complete! Chunks saved to data/chunks_{os.path.basename(args.input).replace('.mmd', '')}.json")
        else:
            # Full indexing mode
            indexer = Indexer(api_key=args.api_key, mmd_lines_path=args.mmd_lines)
            save_path = indexer.index_file(
                args.input,
                use_cached_chunks=not args.no_cache,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap
            )
            print(f"\nIndexing complete! Index saved to: {save_path}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())