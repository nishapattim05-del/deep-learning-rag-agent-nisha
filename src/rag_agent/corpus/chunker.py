"""
chunker.py
==========
Document chunking pipeline for RAG ingestion.
 
Splits raw documents into chunks and attaches metadata.
 
Supports:
- Markdown (.md)
- PDF (.pdf)
"""
 
from __future__ import annotations
 
from pathlib import Path
from typing import List
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
 
from rag_agent.agent.state import ChunkMetadata, DocumentChunk
from rag_agent.vectorstore.store import VectorStoreManager
 
 
class DocumentChunker:
    """
    Splits documents into chunks and attaches metadata.
    """
 
    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )
 
    # -----------------------------------------------------------------------
    # PUBLIC METHODS
    # -----------------------------------------------------------------------
 
    def chunk_files(self, file_paths: List[Path]) -> List[DocumentChunk]:
        """
        Process multiple files.
        """
        all_chunks: List[DocumentChunk] = []
 
        for file_path in file_paths:
            chunks = self.chunk_file(file_path)
            all_chunks.extend(chunks)
 
        return all_chunks
 
    def chunk_file(self, file_path: Path) -> List[DocumentChunk]:
        """
        Process a single file based on its type.
        """
        suffix = file_path.suffix.lower()
 
        if suffix == ".md":
            text = file_path.read_text(encoding="utf-8")
 
        elif suffix == ".pdf":
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
 
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
 
        return self._create_chunks(text, file_path.name)
 
    # -----------------------------------------------------------------------
    # CHUNK CREATION
    # -----------------------------------------------------------------------
 
    def _create_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Split text and attach metadata.
        """
        splits = self._splitter.split_text(text)
 
        chunks: List[DocumentChunk] = []
 
        for chunk_text in splits:
            metadata = ChunkMetadata(
                source=source,
                topic=self._infer_topic(source),
                difficulty=self._infer_difficulty(source),
                type="concept_explanation"
            )
 
            chunk_id = VectorStoreManager.generate_chunk_id(
                source, chunk_text
            )
 
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    metadata=metadata,
                )
            )
 
        return chunks
 
    # -----------------------------------------------------------------------
    # METADATA HELPERS
    # -----------------------------------------------------------------------
 
    def _infer_topic(self, filename: str) -> str:
        name = filename.lower()
 
        if "cnn" in name:
            return "CNN"
        elif "rnn" in name:
            return "RNN"
        elif "ann" in name:
            return "ANN"
 
        return "General"
 
    def _infer_difficulty(self, filename: str) -> str:
        name = filename.lower()
 
        if "beginner" in name:
            return "beginner"
        elif "intermediate" in name:
            return "intermediate"
        elif "advanced" in name:
            return "advanced"
 
        return "unknown"