"""
RAG Chatbot Lab configuration.

This module is the single place where environment-driven settings are parsed.
All system prompts are loaded from text files under ``prompts/`` so students
can customize the agent without touching Python code.

Loading order for prompts:
1. ``prompts/<name>.txt`` on disk (student-editable surface).
2. A small built-in default shipped with this module (used only if the file
   is missing or unreadable; warns at startup).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from config.corpus_registry import get_documents

load_dotenv()
logger = logging.getLogger(__name__)

PROMPTS_DIR_ENV = "PROMPTS_DIR"
KNOWLEDGE_BASE_PATH_ENV = "KNOWLEDGE_BASE_PATH"
PROCESSED_OUTPUT_PATH_ENV = "OCR_DATA_PATH"
SAMPLE_CORPUS_PATH_ENV = "SAMPLE_CORPUS_PATH"
_DEFAULT_PROCESSED_OUTPUT = "./examples/"
_DEFAULT_KNOWLEDGE_BASE = "./knowledge_base/"


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

_DEFAULT_GENERAL_PROMPT = (
    "You are a teaching assistant for the RAG Chatbot Lab. "
    "Discuss the materials in the knowledge base, citing passages with [citation: #]."
)

_DEFAULT_TEXT_SPECIFIC_PROMPT = (
    "You are a teaching assistant for the RAG Chatbot Lab in text-specific mode. "
    "Stay grounded in the selected documents, citing passages with [citation: #]."
)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a teaching assistant for the RAG Chatbot Lab. "
    "Use the retrieved context as your primary source and cite passages with [citation: #]."
)

_DEFAULT_SUMMARY_PROMPT = (
    "Summarize the provided documents for a learner. "
    "Use prose, structure as introduction, content summary, themes, notable features."
)


def _resolve_prompts_dir() -> Path:
    """Locate the prompts directory.

    Resolution order:
    1. ``PROMPTS_DIR`` environment variable (absolute or relative to CWD).
    2. ``<repo_root>/prompts`` — the repo layout ships with prompts there.
    """
    override = os.getenv(PROMPTS_DIR_ENV)
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path
    return Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(name: str, fallback: str) -> str:
    """Load a single prompt file from disk, falling back to a safe default."""
    path = _resolve_prompts_dir() / f"{name}.txt"
    if not path.exists():
        logger.warning(f"Prompts file missing: {path}. Using built-in default.")
        return fallback
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read {path}: {exc}. Using built-in default.")
        return fallback


def _resolve_repo_relative_path(value: str) -> Path:
    """Resolve ``value`` against the package root rather than CWD.

    Relative paths (including ``../``) are anchored to the package root.
    Absolute paths are used as-is.
    """
    path = Path(value)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent.parent / value).resolve()


# ---------------------------------------------------------------------------
# Settings sections
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    """Configuration for embedding + generation models."""
    name: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    device: str = os.getenv("DEVICE", "cpu")
    cache_dir: str = os.getenv("MODEL_CACHE_DIR", "./models/")

    max_context_length: int = int(os.getenv("MAX_CONTEXT_LENGTH", "4096"))
    max_response_length: int = int(os.getenv("MAX_RESPONSE_LENGTH", "8192"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    do_sample: bool = os.getenv("DO_SAMPLE", "true").lower() == "true"


class LLMProviderConfig(BaseModel):
    """Configuration for LLM providers."""
    default_provider: str = os.getenv("DEFAULT_LLM_PROVIDER", "huggingface")

    huggingface: dict = {
        "model_name": os.getenv("HUGGINGFACE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        "api_key": os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"),
        "token": os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN"),
        "temperature": float(os.getenv("TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("MAX_RESPONSE_LENGTH", "8192")),
        "top_p": float(os.getenv("TOP_P", "0.9")),
    }
    openai: dict = {
        "model_name": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("OPENAI_TOP_P", "0.9")),
    }
    anthropic: dict = {
        "model_name": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "temperature": float(os.getenv("ANTHROPIC_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("ANTHROPIC_TOP_P", "0.9")),
    }
    deepseek: dict = {
        "model_name": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "temperature": float(os.getenv("DEEPSEEK_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("DEEPSEEK_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("DEEPSEEK_TOP_P", "0.9")),
    }
    grok: dict = {
        "model_name": os.getenv("GROK_MODEL", "grok-beta"),
        "api_key": os.getenv("GROK_API_KEY"),
        "base_url": os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
        "temperature": float(os.getenv("GROK_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("GROK_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("GROK_TOP_P", "0.9")),
    }
    kimi: dict = {
        "model_name": os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
        "api_key": os.getenv("KIMI_API_KEY"),
        "base_url": os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        "temperature": float(os.getenv("KIMI_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("KIMI_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("KIMI_TOP_P", "0.9")),
    }
    minimax: dict = {
        "model_name": os.getenv("MINIMAX_MODEL", "abab6.5-chat"),
        "api_key": os.getenv("MINIMAX_API_KEY"),
        "base_url": os.getenv("MINIMAX_BASE_URL", "https://api.minimax.chat/v1"),
        "temperature": float(os.getenv("MINIMAX_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("MINIMAX_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("MINIMAX_TOP_P", "0.9")),
    }
    qwen: dict = {
        "model_name": os.getenv("QWEN_MODEL", "qwen-plus"),
        "api_key": os.getenv("QWEN_API_KEY"),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "temperature": float(os.getenv("QWEN_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("QWEN_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("QWEN_TOP_P", "0.9")),
    }
    glm: dict = {
        "model_name": os.getenv("GLM_MODEL", "glm-4"),
        "api_key": os.getenv("GLM_API_KEY"),
        "base_url": os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        "temperature": float(os.getenv("GLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("GLM_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("GLM_TOP_P", "0.9")),
    }
    ollama: dict = {
        "model_name": os.getenv("OLLAMA_MODEL", "llama2"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
        "max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "8192")),
        "top_p": float(os.getenv("OLLAMA_TOP_P", "0.9")),
    }


class RAGConfig(BaseModel):
    """Configuration for the RAG system."""
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


class PathConfig(BaseModel):
    """Configuration for file paths used by the lab.

    ``knowledge_base_path`` is the authored corpus root. When it contains at
    least one ``<doc_id>/meta.yaml``, only the authored layer is used.
    Otherwise the lab falls back to ``processed_output_path`` so the demo
    corpus keeps working unchanged.

    All paths resolve relative to the package root (where this file lives),
    not the current working directory, so the lab starts the same way
    regardless of where the user runs ``python -m`` from.
    """
    base_dir: Path = Path(__file__).parent
    package_root: Path = Path(__file__).resolve().parent.parent

    base_data_path: Path = Path(__file__).resolve().parent.parent

    prompts_dir: Path = _resolve_prompts_dir()
    knowledge_base_path: Path = _resolve_repo_relative_path(
        os.getenv(KNOWLEDGE_BASE_PATH_ENV, _DEFAULT_KNOWLEDGE_BASE)
    )
    sample_corpus_path: Path = _resolve_repo_relative_path(
        os.getenv(SAMPLE_CORPUS_PATH_ENV)
        or os.getenv(PROCESSED_OUTPUT_PATH_ENV)
        or _DEFAULT_PROCESSED_OUTPUT
    )
    # Backwards-compatible alias for code that referenced ``processed_output_path``.
    @property
    def processed_output_path(self) -> Path:
        return self.sample_corpus_path

    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/embeddings/")
    processed_texts_path: str = os.getenv("PROCESSED_TEXTS_PATH", "./data/processed_texts/")
    logs_path: str = "./logs/"

    def get_absolute_path(self, relative_path: str) -> Path:
        return _resolve_repo_relative_path(relative_path)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # No HF-Spaces branch — CLI-only build. Paths are anchored to the
        # package root so the lab runs identically regardless of CWD.


class ModeConfig(BaseModel):
    """Configuration for student interaction modes."""
    default_mode: str = os.getenv("DEFAULT_MODE", "general")
    available_modes: List[str] = ["general", "text_specific"]

    general_mode_prompt: str = ""
    text_specific_mode_prompt: str = ""
    system_prompt: str = ""
    summary_mode_prompt: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        # Prompt strings are loaded from prompts/*.txt so students can edit
        # them without touching this module. They are not part of the
        # constructor signature.
        prompts = {
            "general_mode_prompt": _load_prompt("general_mode", _DEFAULT_GENERAL_PROMPT),
            "text_specific_mode_prompt": _load_prompt(
                "text_specific_mode", _DEFAULT_TEXT_SPECIFIC_PROMPT
            ),
            "system_prompt": _load_prompt("system", _DEFAULT_SYSTEM_PROMPT),
            "summary_mode_prompt": _load_prompt("summary_mode", _DEFAULT_SUMMARY_PROMPT),
        }
        prompts.update(kwargs)
        super().__init__(**prompts)


class CLIConfig(BaseModel):
    """Configuration for the lab CLI runner."""
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


class SystemConfig(BaseModel):
    """Main system configuration."""
    model: ModelConfig = ModelConfig()
    llm_providers: LLMProviderConfig = LLMProviderConfig()
    rag: RAGConfig = RAGConfig()
    paths: PathConfig = PathConfig()
    cli: CLIConfig = CLIConfig()
    modes: ModeConfig = ModeConfig()

    model_config = {"arbitrary_types_allowed": True}


# Global configuration instance. Imported as `from config import config`.
config = SystemConfig()


# ---------------------------------------------------------------------------
# Corpus helpers (the lab's data-driven layer)
# ---------------------------------------------------------------------------

def get_active_documents() -> List[dict]:
    """Return the active document registry from corpus_registry.

    This replaces the legacy ``LITERATURE_WORKS`` constant. Callers that need
    the doc list should import this function (or `from config.corpus_registry
    import get_documents` directly).
    """
    return get_documents(
        knowledge_base_path=config.paths.knowledge_base_path,
        sample_corpus_path=config.paths.sample_corpus_path,
    )


def get_document_by_id(doc_id: str) -> Optional[dict]:
    """Look up a document by id in the active registry."""
    from config.corpus_registry import find_by_id  # local import for clarity

    return find_by_id(get_active_documents(), doc_id)


def get_document_by_folder(folder_name: str) -> Optional[dict]:
    """Look up a document by on-disk folder name in the active registry."""
    from config.corpus_registry import find_by_folder

    return find_by_folder(get_active_documents(), folder_name)


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    paths = config.paths
    dirs_to_create = [
        paths.get_absolute_path(paths.vector_store_path),
        paths.get_absolute_path(paths.processed_texts_path),
        paths.get_absolute_path(paths.logs_path),
        paths.get_absolute_path(config.model.cache_dir),
        paths.knowledge_base_path,
        paths.sample_corpus_path,
    ]
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
