"""
Doctor Checks
Dependency-light diagnostic checks for each of the 5 STUDENT_GUIDE.md setup
steps. These must work even when the rest of the system (embeddings, vector
store) is broken -- that's the whole point of `doctor`.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix_hint: str = ""


REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PACKAGES = [
    "faiss",
    "sentence_transformers",
    "openai",
    "pydantic",
    "dotenv",
    "yaml",
]

PROVIDER_ENV_VARS = {
    "HuggingFace": ["HF_TOKEN", "HUGGINGFACE_API_KEY"],
    "OpenAI": ["OPENAI_API_KEY"],
    "Anthropic": ["ANTHROPIC_API_KEY"],
    "DeepSeek": ["DEEPSEEK_API_KEY"],
    "Grok": ["GROK_API_KEY"],
    "Kimi": ["KIMI_API_KEY"],
    "MiniMax": ["MINIMAX_API_KEY"],
    "Qwen": ["QWEN_API_KEY"],
    "GLM": ["GLM_API_KEY"],
    "Ollama": ["OLLAMA_BASE_URL"],
}

KEY_PREFIX_HINTS = {
    "HF_TOKEN": "hf_",
    "HUGGINGFACE_API_KEY": "hf_",
    "OPENAI_API_KEY": "sk-",
    "ANTHROPIC_API_KEY": "sk-ant-",
}


def check_install() -> List[CheckResult]:
    results = []
    py_ver = sys.version_info
    results.append(CheckResult(
        name="Python version",
        ok=py_ver >= (3, 9),
        detail=f"Running Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
        fix_hint="Install Python 3.9 or newer.",
    ))
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            results.append(CheckResult(
                name=f"import {package}",
                ok=True,
                detail=f"{package} imports cleanly",
            ))
        except ImportError as exc:
            results.append(CheckResult(
                name=f"import {package}",
                ok=False,
                detail=f"Could not import {package}: {exc}",
                fix_hint="Run: pip install -r requirements.txt",
            ))
    return results


def check_api_key() -> List[CheckResult]:
    results = []
    env_path = REPO_ROOT / ".env"
    results.append(CheckResult(
        name=".env file exists",
        ok=env_path.exists(),
        detail=str(env_path) if env_path.exists() else f"{env_path} not found",
        fix_hint="Run: cp .env.example .env",
    ))

    configured = []
    for provider, var_names in PROVIDER_ENV_VARS.items():
        for var_name in var_names:
            value = os.getenv(var_name)
            if value:
                configured.append((provider, var_name, value))
                break

    results.append(CheckResult(
        name="At least one provider key configured",
        ok=bool(configured),
        detail=(
            f"Configured: {', '.join(p for p, _, _ in configured)}"
            if configured
            else "No provider environment variables are set"
        ),
        fix_hint="Edit .env and uncomment one provider line, e.g. HF_TOKEN=hf_...",
    ))

    for provider, var_name, value in configured:
        prefix = KEY_PREFIX_HINTS.get(var_name)
        if prefix and not value.startswith(prefix):
            results.append(CheckResult(
                name=f"{provider} key format",
                ok=False,
                detail=f"{var_name} doesn't start with the expected '{prefix}' prefix",
                fix_hint=f"Double-check you copied the full {provider} key into .env",
            ))

    return results


def check_build_index() -> List[CheckResult]:
    results = []
    try:
        from config import get_active_documents
        docs = get_active_documents()
    except Exception as exc:  # noqa: BLE001
        return [CheckResult(
            name="Corpus loads",
            ok=False,
            detail=f"Could not load active corpus: {exc}",
            fix_hint="Add a knowledge_base/<doc_id>/meta.yaml (see STUDENT_GUIDE.md step 3a).",
        )]

    results.append(CheckResult(
        name="Corpus loads",
        ok=bool(docs),
        detail=f"{len(docs)} document(s) in the active corpus" if docs else "No active corpus found",
        fix_hint="Add a knowledge_base/<doc_id>/meta.yaml (see STUDENT_GUIDE.md step 3a).",
    ))

    from config import config
    index_path = config.paths.get_absolute_path(config.paths.vector_store_path) / "faiss_index.idx"
    metadata_path = config.paths.get_absolute_path(config.paths.vector_store_path) / "faiss_metadata.json"

    results.append(CheckResult(
        name="FAISS index file exists",
        ok=index_path.exists(),
        detail=str(index_path) if index_path.exists() else f"{index_path} not found",
        fix_hint="Run: python tests/quick_test.py --rebuild",
    ))

    if index_path.exists():
        try:
            from models.vector_store import VectorStore
            store = VectorStore()
            loaded = store.load(str(index_path), str(metadata_path))
            results.append(CheckResult(
                name="FAISS index loads with vectors",
                ok=loaded and store.count > 0,
                detail=f"{store.count} vectors loaded" if loaded else "Index failed to load",
                fix_hint="Run: python tests/quick_test.py --rebuild",
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(
                name="FAISS index loads with vectors",
                ok=False,
                detail=f"Error loading index: {exc}",
                fix_hint="Run: python tests/quick_test.py --rebuild",
            ))

    return results


def check_chat(run_live_test: bool = False) -> List[CheckResult]:
    results = []
    try:
        from services.chatbot_service import ChatbotService
        service = ChatbotService()
        results.append(CheckResult(
            name="ChatbotService initializes",
            ok=True,
            detail="ChatbotService() constructed without raising",
        ))
    except Exception as exc:  # noqa: BLE001
        return [CheckResult(
            name="ChatbotService initializes",
            ok=False,
            detail=f"ChatbotService() raised: {exc}",
            fix_hint="Run: python tests/quick_test.py --rebuild, then check .env for a valid provider key.",
        )]

    try:
        session_id, _ = service.start_chat_session()
        results.append(CheckResult(
            name="Chat session starts",
            ok=bool(session_id),
            detail=f"Session id: {session_id}",
        ))
    except Exception as exc:  # noqa: BLE001
        results.append(CheckResult(
            name="Chat session starts",
            ok=False,
            detail=f"start_chat_session() raised: {exc}",
        ))
        return results

    if run_live_test:
        try:
            response = service.process_message("Hello", session_id)
            results.append(CheckResult(
                name="Live provider round-trip",
                ok=bool(response.message),
                detail=response.message[:120],
            ))
        except Exception as exc:  # noqa: BLE001
            results.append(CheckResult(
                name="Live provider round-trip",
                ok=False,
                detail=f"process_message() raised: {exc}",
                fix_hint="Check the active provider's API key and network access.",
            ))

    return results


def check_customize() -> List[CheckResult]:
    results = []
    from config import config

    for prompt_name in ["general_mode", "text_specific_mode", "system", "summary_mode"]:
        path = config.paths.prompts_dir / f"{prompt_name}.txt"
        non_empty = path.exists() and path.read_text(encoding="utf-8").strip() != ""
        results.append(CheckResult(
            name=f"prompts/{prompt_name}.txt",
            ok=non_empty,
            detail=str(path) if non_empty else f"{path} missing or empty",
            fix_hint=f"Create/edit {path} with your agent's voice for this mode.",
        ))

    kb_has_docs = any(config.paths.knowledge_base_path.glob("*/meta.yaml"))
    sample_exists = config.paths.sample_corpus_path.exists()
    results.append(CheckResult(
        name="Knowledge base or sample corpus present",
        ok=kb_has_docs or sample_exists,
        detail=(
            "knowledge_base/*/meta.yaml found" if kb_has_docs
            else "sample corpus found" if sample_exists
            else "Neither knowledge_base/ nor the sample corpus exist"
        ),
        fix_hint="See STUDENT_GUIDE.md section 5b for how to add knowledge_base/<doc_id>/meta.yaml.",
    ))

    return results


STEPS: List[Tuple[str, Callable[..., List[CheckResult]]]] = [
    ("1. Install", check_install),
    ("2. Add an API key", check_api_key),
    ("3. Build the FAISS index", check_build_index),
    ("4. Chat", check_chat),
    ("5. Customize (optional)", check_customize),
]
