"""
config.py
=========
Centralised configuration and LLM + Embedding factories.
"""
 
from __future__ import annotations
 
import os
import streamlit as st
from enum import Enum
from functools import lru_cache
 
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
# ---------------------------------------------------------------------------
# ENUMS
# ---------------------------------------------------------------------------
 
class LLMProvider(str, Enum):
    GROQ = "groq"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
 
 
class EmbeddingProvider(str, Enum):
    LOCAL = "local"
    OPENAI = "openai"
 
 
# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
 
    # LLM
    llm_provider: LLMProvider = LLMProvider.GROQ
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
 
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.2", alias="OLLAMA_MODEL")
 
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL")
    lmstudio_model: str = Field(default="local-model", alias="LMSTUDIO_MODEL")
 
    # Embeddings
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
 
    # Vector store
    chroma_db_path: str = Field(default="./data/chroma_db", alias="CHROMA_DB_PATH")
    chroma_collection_name: str = Field(default="deep_learning_corpus", alias="CHROMA_COLLECTION_NAME")
 
    # Retrieval
    retrieval_k: int = Field(default=4, alias="RETRIEVAL_K")
    similarity_threshold: float = Field(default=0.2, alias="SIMILARITY_THRESHOLD")
 
    # App
    max_context_tokens: int = Field(default=3000, alias="MAX_CONTEXT_TOKENS")
    app_title: str = Field(default="Deep Learning Interview Prep Agent", alias="APP_TITLE")
 
 
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings(**st.secrets)
    except Exception:
        return Settings()
 
 
# ---------------------------------------------------------------------------
# LLM FACTORY
# ---------------------------------------------------------------------------
 
class LLMFactory:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
 
    def create(self) -> BaseChatModel:
        provider = self._settings.llm_provider
 
        if provider == LLMProvider.GROQ:
            return self._create_groq()
        elif provider == LLMProvider.OLLAMA:
            return self._create_ollama()
        elif provider == LLMProvider.LMSTUDIO:
            return self._create_lmstudio()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
 
    # -------------------- GROQ --------------------
    def _create_groq(self) -> BaseChatModel:
        from langchain_groq import ChatGroq
 
        api_key = os.getenv("GROQ_API_KEY")
 
        # fallback to Streamlit secrets
        if not api_key:
            try:
                api_key = st.secrets.get("GROQ_API_KEY")
            except Exception:
                pass
 
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY is missing")
 
        return ChatGroq(
            api_key=api_key,
            model=self._settings.groq_model,
        )
 
    # -------------------- OLLAMA --------------------
    def _create_ollama(self):
        from langchain_community.chat_models import ChatOllama
 
        return ChatOllama(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_model,
            temperature=0.2,
        )
 
    # -------------------- LM STUDIO --------------------
    def _create_lmstudio(self):
        from langchain_openai import ChatOpenAI
 
        return ChatOpenAI(
            base_url=self._settings.lmstudio_base_url,
            api_key="not-needed",
            model=self._settings.lmstudio_model,
            temperature=0.2,
        )
 
 
# ---------------------------------------------------------------------------
# EMBEDDING FACTORY
# ---------------------------------------------------------------------------
 
class EmbeddingFactory:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
 
    def create(self):
        provider = self._settings.embedding_provider
 
        if provider == EmbeddingProvider.LOCAL:
            return self._create_local()
        elif provider == EmbeddingProvider.OPENAI:
            return self._create_openai()
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
 
    def _create_local(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings
 
        return HuggingFaceEmbeddings(
            model_name=self._settings.embedding_model
        )
 
    def _create_openai(self):
        from langchain_openai import OpenAIEmbeddings
 
        return OpenAIEmbeddings(
            model="text-embedding-3-small"
        )