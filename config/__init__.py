"""Configuration package for the RAG Chatbot Lab."""

from .config import (
    config,
    ModelConfig,
    LLMProviderConfig,
    RAGConfig,
    PathConfig,
    ModeConfig,
    SystemConfig,
    get_active_documents,
    get_document_by_id,
    get_document_by_folder,
    ensure_directories,
)

__all__ = [
    "config",
    "ModelConfig",
    "LLMProviderConfig",
    "RAGConfig",
    "PathConfig",
    "ModeConfig",
    "SystemConfig",
    "get_active_documents",
    "get_document_by_id",
    "get_document_by_folder",
    "ensure_directories",
]
