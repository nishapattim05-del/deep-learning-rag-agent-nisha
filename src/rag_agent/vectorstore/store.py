"""
store.py
========
ChromaDB vector store management.
"""
 
from __future__ import annotations
 
import hashlib
from pathlib import Path
 
import chromadb
from loguru import logger
 
from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import EmbeddingFactory, Settings, get_settings
 
 
class VectorStoreManager:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = EmbeddingFactory(self._settings).create()
        self._client = None
        self._collection = None
        self._initialise()
 
    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------
 
    def _initialise(self) -> None:
        try:
            Path(self._settings.chroma_db_path).mkdir(
                parents=True, exist_ok=True
            )
 
            self._client = chromadb.PersistentClient(
                path=self._settings.chroma_db_path
            )
 
            self._collection = self._client.get_or_create_collection(
                name=self._settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
 
            count = self._collection.count()
            logger.info(
                f"ChromaDB initialised | Collection: {self._settings.chroma_collection_name} | Items: {count}"
            )
 
        except Exception as e:
            raise RuntimeError(f"Failed to initialise ChromaDB: {e}")
 
    # -----------------------------------------------------------------------
    # Duplicate Detection
    # -----------------------------------------------------------------------
 
    @staticmethod
    def generate_chunk_id(source: str, chunk_text: str) -> str:
        content = f"{source}::{chunk_text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
 
    def check_duplicate(self, chunk_id: str) -> bool:
        result = self._collection.get(ids=[chunk_id])
        return len(result.get("ids", [])) > 0
 
    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------
 
    def ingest(self, chunks: list[DocumentChunk]) -> IngestionResult:
        result = IngestionResult()
 
        for chunk in chunks:
            try:
                if self.check_duplicate(chunk.chunk_id):
                    result.skipped += 1
                    continue
 
                embedding = self._embeddings.embed_documents(
                    [chunk.chunk_text]
                )[0]
 
                self._collection.upsert(
                    ids=[chunk.chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.chunk_text],
                    metadatas=[chunk.metadata.to_dict()],
                )
 
                result.ingested += 1
 
            except Exception as e:
                logger.error(f"Ingestion error: {e}")
                result.errors += 1
 
        logger.info(
            f"Ingestion complete | Ingested: {result.ingested}, Skipped: {result.skipped}, Errors: {result.errors}"
        )
 
        return result
 
    # -----------------------------------------------------------------------
    # Retrieval (FIXED)
    # -----------------------------------------------------------------------
 
    def query(
        self,
        query_text: str,
        k: int | None = None,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
 
        k = k or self._settings.retrieval_k
 
        where_filter = {}
        if topic_filter:
            where_filter["topic"] = topic_filter
        if difficulty_filter:
            where_filter["difficulty"] = difficulty_filter
 
        if not where_filter:
            where_filter = None
 
        query_embedding = self._embeddings.embed_query(query_text)
 
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
 
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
 
        retrieved_chunks = []
 
        for i, (doc, meta, dist) in enumerate(
            zip(documents, metadatas, distances)
        ):
            score = 1 - dist
 
            if score < self._settings.similarity_threshold:
                continue
 
            # ✅ FIX: ensure chunk_id is provided
            chunk_id = self.generate_chunk_id(
                meta.get("source", ""), doc
            )
 
            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=doc,
                    metadata=ChunkMetadata(**meta),
                    score=score,
                )
            )
 
        return sorted(
            retrieved_chunks, key=lambda x: x.score, reverse=True
        )
 
    # -----------------------------------------------------------------------
    # Corpus Inspection
    # -----------------------------------------------------------------------
 
    def list_documents(self) -> list[dict]:
        results = self._collection.get(include=["metadatas"])
 
        docs = {}
        for meta in results.get("metadatas", []):
            source = meta.get("source")
 
            if source not in docs:
                docs[source] = {
                    "source": source,
                    "topic": meta.get("topic"),
                    "chunk_count": 0,
                }
 
            docs[source]["chunk_count"] += 1
 
        return list(docs.values())
 
    def get_document_chunks(self, source: str) -> list[DocumentChunk]:
        results = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
 
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
 
        return [
            DocumentChunk(
                chunk_id=self.generate_chunk_id(meta["source"], doc),
                chunk_text=doc,
                metadata=ChunkMetadata(**meta),
            )
            for doc, meta in zip(documents, metadatas)
        ]
 
    def get_collection_stats(self) -> dict:
        results = self._collection.get(include=["metadatas"])
 
        topics = set()
        sources = set()
        bonus = False
 
        for meta in results.get("metadatas", []):
            topics.add(meta.get("topic"))
            sources.add(meta.get("source"))
 
            if meta.get("is_bonus"):
                bonus = True
 
        return {
            "total_chunks": self._collection.count(),
            "topics": list(topics),
            "sources": list(sources),
            "bonus_topics_present": bonus,
        }
 
    def delete_document(self, source: str) -> int:
        results = self._collection.get(where={"source": source})
 
        ids = results.get("ids", [])
 
        if ids:
            self._collection.delete(ids=ids)
 
        return len(ids)