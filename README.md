# RAG Chatbot Lab

A local, CLI-driven **RAG chatbot**. Customize the agent by editing prompts and swapping in your own knowledge base. No web server. No database. Everything lives on your filesystem.

## What you'll do

```
1. Install              pip install -r requirements.txt
2. Add an API key       copy .env.example → .env, uncomment one provider line
3. Add a corpus         create knowledge_base/<doc_id>/meta.yaml + text
4. Build the index      python tests/quick_test.py
5. Chat                 python -m commands.cli
```

Full walkthrough with troubleshooting in **[`STUDENT_GUIDE.md`](./STUDENT_GUIDE.md)**.

## Mental model

```
corpus text ──[ingest]──▶ chunks ──[embed]──▶ vectors (FAISS index)
                                                │
user query ────────────────────────────────────▶│── retrieve top-k similar chunks
                                                │
                                prompt + context + question ──▶ LLM ──▶ answer
```

Three student-editable surfaces:

| Surface | Where | What changes |
|---|---|---|
| **Voice** | `prompts/*.txt` | How the agent talks and what scope it allows. |
| **Knowledge** | `knowledge_base/<doc_id>/meta.yaml` + text | What the agent knows about. |
| **Keys & tuning** | `.env` | API keys, default provider, paths, retrieval settings. |

## Quick start

```bash
# 1. Install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Add an API key (Hugging Face free tier works)
cp .env.example .env
# open .env → section 1: uncomment one provider line, e.g. HF_TOKEN=hf_...

# 3. Add a corpus (no bundled sample — bring at least one document)
mkdir -p knowledge_base/my_doc
# add knowledge_base/my_doc/meta.yaml + knowledge_base/my_doc/ocr_output.json
# see knowledge_base/README.md for the format

# 4. Build the index + smoke test
python tests/quick_test.py

# 5. Chat
python -m commands.cli
```

`quick_test.py` checks imports, lists configured providers, builds the FAISS index (skips if one already exists), and loads it. A green run means the lab is ready.

After changing the corpus or prompts, rebuild with `python tests/quick_test.py --rebuild` and restart the CLI.

## Corpus

There's no bundled sample corpus — you provide the documents. The lab reads `knowledge_base/<doc_id>/meta.yaml` files; if none exist, the active corpus is empty and setup will tell you so.

See [`knowledge_base/README.md`](./knowledge_base/README.md) for the `meta.yaml` format and file layout.

## Layout

```
.
├── STUDENT_GUIDE.md              ← five-step walkthrough
├── README.md
├── .env.example                  ← copy to .env
├── requirements.txt
│
├── prompts/                      ← student-editable: agent voice
│   ├── general_mode.txt
│   ├── text_specific_mode.txt
│   ├── system.txt
│   └── summary_mode.txt
├── knowledge_base/               ← student-editable: your documents (no bundled sample)
│
├── commands/
│   ├── cli.py                    ← menu-driven chatbot
│   ├── doctor.py                 ← interactive doctor: menu + LLM-explained failures
│   └── doctor_checks.py          ← the PASS/FAIL checks doctor.py runs (no UI)
├── scripts/setup_system.py       ← builds the FAISS index
├── tests/quick_test.py           ← setup + verify in one command
│
├── config/                       ← config.py + corpus_registry.py
├── services/                     ← chatbot, retrieval, sessions, logger
├── models/                       ← FAISS store, answer generator, LLM providers
├── utils/                        ← embedding generator
└── data/                         ← runtime output (auto-created, gitignored):
                                    embeddings/, processed_texts/, conversations/
```

## LLM providers

Ten providers are pre-wired. Uncomment **one** key in `.env` — no other config required:

| Provider | Env var |
|---|---|
| Hugging Face | `HF_TOKEN` (or `HUGGINGFACE_API_KEY`) |
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Grok (xAI) | `GROK_API_KEY` |
| Kimi | `KIMI_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| Qwen | `QWEN_API_KEY` |
| GLM | `GLM_API_KEY` |
| Ollama | `OLLAMA_BASE_URL` |

Optional: set `DEFAULT_LLM_PROVIDER` in `.env`, or switch at runtime from the CLI menu. See `.env.example` for path overrides and retrieval tuning (`RETRIEVAL_TOP_K`, `CHUNK_SIZE`, etc.).

## CLI menu

`python -m commands.cli` opens an arrow-key menu:

```text
> Start chatting
> Switch provider
> Switch mode (general / text_specific)
> Pick a document to focus on
> Toggle RAG on/off
> Session info / stats
> View corpus (and how to add your own document)
> Save conversation
> Run doctor (diagnose a setup step)
> Quit
```

Inside **Start chatting**, type your message. Shortcuts: `/menu` (back to menu), `/works` (corpus + add-a-document guide), `/quit` (save and exit), `/help`.

Stuck on setup? Pick **Run doctor** or run `python -m commands.doctor` — it checks each setup step and explains failures using your configured LLM. `commands/doctor.py` is the interactive menu and LLM-explain layer; `commands/doctor_checks.py` holds the actual PASS/FAIL check functions and has no UI, so it keeps working even when the rest of the system is broken.

## What's intentionally not here

This is a teaching lab, not a product:

- Web server, REST API, dashboard, or UI
- Database (no Supabase, Postgres, or SQLite runtime)
- Authentication, user accounts, or cookie sessions
- Background jobs, queues, or workers
- Docker, HuggingFace Spaces, or cloud deployment configs
- Question-generation harness, studio tools, or analytics dashboards

## License

MIT
