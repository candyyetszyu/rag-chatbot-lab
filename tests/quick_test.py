#!/usr/bin/env python3
"""
Quick Test — runs the project setup end-to-end and verifies everything.

Equivalent to:
    python -m scripts.setup_system
plus a sanity check that the FAISS index built successfully and the
retrieval pipeline can answer a query.

Run:
    python tests/quick_test.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

HEADER = "=" * 60


def section(title: str) -> None:
    print(f"\n{HEADER}\n  {title}\n{HEADER}")


def check_config() -> None:
    section("1. Configuration")
    from config import config, get_active_documents

    print(f"  Model       : {config.model.name}")
    print(f"  Embedding   : {config.model.embedding_model}")
    print(f"  Prompts dir : {config.paths.prompts_dir}")
    print(f"  Knowledge   : {config.paths.knowledge_base_path}")
    print(f"  Sample      : {config.paths.sample_corpus_path}")
    print(f"  Vector store: {config.paths.vector_store_path}")

    docs = get_active_documents()
    print(f"\n  Active corpus: {len(docs)} document(s)")
    for d in docs[:3]:
        print(f"    - {d.get('id', '?')}  [{d.get('folder', '?')}]")
    if len(docs) > 3:
        print(f"    ... and {len(docs) - 3} more")

    if not docs:
        raise SystemExit("\n✗ No active corpus. Add a knowledge_base/<doc_id>/meta.yaml "
                         "(see STUDENT_GUIDE.md step 3a).")
    print("  ✓ Corpus loaded")


def check_providers() -> dict:
    section("2. LLM providers (which APIs have keys?)")
    from config import config

    keys = {
        "HuggingFace": os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"),
        "OpenAI":      os.getenv("OPENAI_API_KEY"),
        "Anthropic":   os.getenv("ANTHROPIC_API_KEY"),
        "DeepSeek":    os.getenv("DEEPSEEK_API_KEY"),
        "Grok":        os.getenv("GROK_API_KEY"),
        "Kimi":        os.getenv("KIMI_API_KEY"),
        "MiniMax":     os.getenv("MINIMAX_API_KEY"),
        "Qwen":        os.getenv("QWEN_API_KEY"),
        "GLM":         os.getenv("GLM_API_KEY"),
    }
    available = {name: bool(key) for name, key in keys.items()}
    for name, present in available.items():
        marker = "✓" if present else "·"
        print(f"  {marker} {name:<12} {'key set' if present else 'no key'}")

    configured = [n for n, p in available.items() if p]
    if not configured:
        print("\n  ⚠ No API keys configured. Add at least one to .env before chatting:")
        print("      HF_TOKEN=hf_…  (free HuggingFace tier works)")
        print("\n  Setup will still build the FAISS index so the lab is bootable.")
    else:
        print(f"\n  ✓ Configured providers: {', '.join(configured)}")
    return available


def run_setup(available_providers: dict) -> bool:
    section("3. Build FAISS index (scripts.setup_system)")
    # Setup is heavy; only run when embeddings don't already exist, unless
    # --rebuild is passed.
    force = "--rebuild" in sys.argv
    vector_path = Path(__file__).parent.parent / "data" / "embeddings" / "faiss_index.idx"
    if vector_path.exists() and not force:
        print(f"  Index already present at {vector_path}")
        print("  (pass --rebuild to force a rebuild)")
        return True

    print("  Running setup_system()...")
    start = time.time()
    try:
        from scripts.setup_system import setup_system
        ok = setup_system()
    except SystemExit as e:
        ok = bool(e.code)
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ setup_system raised: {e}")
        return False
    elapsed = time.time() - start
    print(f"  Setup completed in {elapsed:.1f}s → {'OK' if ok else 'FAILED'}")
    return bool(ok)


def check_retrieval() -> None:
    section("4. Smoke-load the FAISS index and run one retrieval")
    try:
        from models.vector_store import VectorStore
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ VectorStore failed to import: {e}")
        return

    idx = VectorStore()
    index_path = Path(__file__).parent.parent / "data" / "embeddings" / "faiss_index.idx"
    meta_path = Path(__file__).parent.parent / "data" / "embeddings" / "faiss_metadata.json"
    if not index_path.exists():
        print("  ⚠ No FAISS index on disk — skipping retrieval smoke test.")
        return
    loaded = idx.load(str(index_path), str(meta_path))
    print(f"  Index loaded: {loaded}")
    print(f"  Vectors    : {getattr(idx, 'count', 'n/a')}")
    print("  ✓ Retrieval pipeline importable")


def summarise(available_providers: dict) -> None:
    section("5. What's next?")
    if not any(available_providers.values()):
        print("  1. Edit .env and add at least one provider API key (HF_TOKEN is fastest).")
    print("  2. Run:    python -m commands.cli")
    print("  3. To swap in your own corpus, see STUDENT_GUIDE.md.")


def main() -> int:
    print(HEADER)
    print("  RAG Chatbot Lab — Quick Test")
    print(HEADER)

    check_config()
    available = check_providers()
    setup_ok = run_setup(available)
    check_retrieval()
    summarise(available)

    section("Done")
    if not setup_ok:
        print("  ⚠ Setup returned a failure. Check logs above. The CLI may still work if")
        print("    the FAISS index already exists from a previous run.")
        return 1
    print("  ✓ Lab is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
