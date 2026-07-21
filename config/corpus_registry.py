"""
Corpus registry for the RAG Chatbot Lab.

Auto-discovers documents in two layers:

1. Authored layer: ``knowledge_base/<doc_id>/meta.yaml`` files. Each YAML
   declares the document's id, title, author, year, genre, source, and the
   on-disk folder that contains its processed text. This is the canonical
   authoring surface for students customizing the agent.

2. Demo fallback: if ``knowledge_base/`` is empty or missing, fall back to
   the ``processed_output/`` folder shipped with the repo. Each subdirectory
   becomes a synthesized record. This keeps the existing demo corpus working
   without forcing students to author 19 YAML files by hand.

Every record returned is a dict with at least:
    id, title, author, year, genre, source, folder, ocr_file

Callers should treat extra keys as optional. The legacy ``LITERATURE_WORKS``
list lived in ``config/config.py`` and is now superseded by ``get_documents``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _normalize(name: str) -> str:
    """Normalize a folder name for matching.

    Mirrors the helper previously embedded in ``services.document_processor``
    so the registry is self-contained.
    """
    normalized = unicodedata.normalize("NFD", name)
    ascii_quote = chr(0x22)
    ascii_single = chr(0x27)
    normalized = (
        normalized.replace(chr(0x201C), ascii_quote)
        .replace(chr(0x201D), ascii_quote)
        .replace(chr(0x2018), ascii_single)
        .replace(chr(0x2019), ascii_single)
    )
    return normalized


def _slug_from_folder(folder_name: str) -> str:
    """Build a stable id from a folder name when no authored meta exists.

    Lowercases, replaces non-word characters with underscores, collapses
    repeated separators. Good enough as an internal identifier.
    """
    base = _normalize(folder_name).lower()
    base = re.sub(r"[^a-z0-9]+", "_", base)
    base = base.strip("_")
    return base or "doc"


def _synthesize_record(folder: Path) -> dict:
    """Synthesize a corpus record from an existing demo folder."""
    folder_name = folder.name
    return {
        "id": _slug_from_folder(folder_name),
        "title": folder_name.replace("_", " ").strip(),
        "author": "",
        "year": "",
        "genre": "",
        "source": "demo",
        "folder": folder_name,
        "ocr_file": "ocr_output.json",
    }


def _load_authored(knowledge_base_path: Path) -> List[dict]:
    """Scan ``knowledge_base/<id>/meta.yaml`` for student-authored records."""
    if not knowledge_base_path.exists():
        return []
    records: List[dict] = []
    for meta_file in sorted(knowledge_base_path.glob("*/meta.yaml")):
        try:
            import yaml  # local import so PyYAML is optional at runtime if user only runs demo
        except ImportError:
            logger.error(
                "PyYAML is required to read knowledge_base/*/meta.yaml. "
                "Add `PyYAML` to requirements.txt and reinstall, or remove "
                "the knowledge_base/ directory to fall back to the demo corpus."
            )
            return []
        try:
            data = yaml.safe_load(meta_file.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Skipping {meta_file}: {exc}")
            continue
        if "id" not in data or "folder" not in data:
            logger.warning(f"Skipping {meta_file}: missing required keys (id, folder)")
            continue
        data.setdefault("source", "authored")
        data.setdefault("ocr_file", "ocr_output.json")
        data.setdefault("author", "")
        data.setdefault("title", data["id"])
        data.setdefault("year", "")
        data.setdefault("genre", "")
        records.append(data)
    return records


def _load_demo_fallback(sample_corpus_path: Path) -> List[dict]:
    """Fall back to the bundled sample corpus if the knowledge base is empty."""
    if not sample_corpus_path.exists():
        return []
    records: List[dict] = []
    for folder in sorted(sample_corpus_path.iterdir()):
        if folder.is_dir():
            records.append(_synthesize_record(folder))
    return records


def get_documents(
    knowledge_base_path: Optional[Path] = None,
    sample_corpus_path: Optional[Path] = None,
    processed_output_path: Optional[Path] = None,
) -> List[dict]:
    """Return the active document list, honoring the authored layer first.

    ``processed_output_path`` is kept as a back-compat alias for
    ``sample_corpus_path`` — older code paths can still pass either name.
    """
    kb = knowledge_base_path
    sample = sample_corpus_path or processed_output_path
    if kb is not None:
        authored = _load_authored(kb)
        if authored:
            return authored
    if sample is not None:
        return _load_demo_fallback(sample)
    return []


def find_by_id(documents: List[dict], doc_id: str) -> Optional[dict]:
    """Look up a document record by id, or ``None`` if absent."""
    for doc in documents:
        if doc.get("id") == doc_id:
            return doc
    return None


def find_by_folder(documents: List[dict], folder_name: str) -> Optional[dict]:
    """Look up a document record by on-disk folder name, or ``None`` if absent."""
    normalized = _normalize(folder_name)
    for doc in documents:
        if _normalize(doc.get("folder", "")) == normalized:
            return doc
    return None
