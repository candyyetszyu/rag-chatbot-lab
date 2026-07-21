# Student Guide — Set up your RAG chatbot in five steps

This walkthrough takes you from a fresh clone to a working chatbot. Follow it once; then refer back whenever you swap in a new corpus or change prompts.

For a high-level overview, see [`README.md`](./README.md).

---

## What you'll do

```
1. Install
2. Add an API key
3. Add a corpus and build the FAISS index
4. Chat
5. (Optional) Customize prompts and retrieval
```

There's no bundled sample corpus — step 3 has you add at least one document before building the index. Total time: about five minutes once you have a document ready.

---

## How the lab works

```
corpus text ──[chunk]──▶ pieces ──[embed]──▶ vectors (FAISS index)
                                                  │
your question ───────────────────────────────────▶│── retrieve top-k chunks
                                                  │
                          prompt + context + question ──▶ LLM ──▶ answer
```

You customize three things:

| Surface | Where | What it changes |
|---|---|---|
| **Voice** | `prompts/*.txt` | Tone, scope, citation style |
| **Knowledge** | `knowledge_base/<doc_id>/` | What the agent can retrieve |
| **Keys & tuning** | `.env` | API keys, provider, retrieval settings |

---

## Step 1 — Install

```bash
cd avatar_backend_training

python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Requires **Python 3.9+**. That's all for this step.

---

## Step 2 — Add an LLM API key

Copy the env template and add **one** API key:

```bash
cp .env.example .env
```

Open `.env`. In **section 1** ("Pick at least one LLM provider"), uncomment exactly one line and paste your key after the `=`.

The fastest option is **Hugging Face** — the free tier works for this lab:

```text
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

(`HUGGINGFACE_API_KEY` works too; it's the same key.)

Other supported providers (uncomment the matching line in `.env`):

| Provider | Env var |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Grok (xAI) | `GROK_API_KEY` |
| Kimi | `KIMI_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| Qwen | `QWEN_API_KEY` |
| GLM | `GLM_API_KEY` |
| Ollama (local) | `OLLAMA_BASE_URL` |

Optional: set `DEFAULT_LLM_PROVIDER` in section 2 of `.env` (e.g. `huggingface`, `openai`). You can also switch providers at runtime from the CLI menu.

Save the file. That's all the config you need for steps 1–4.

> The `.env` file is in `.gitignore` on purpose. Don't commit it.

---

## Step 3 — Add a corpus and build the FAISS index

There's no bundled sample corpus. Before you can build an index, add at least one document under `knowledge_base/`.

### 3a. Add a document

Each document needs a folder with two files:

```
knowledge_base/
└── my_doc/
    ├── meta.yaml         ← required
    └── ocr_output.json   ← your text, chunked by page
```

`meta.yaml`:

```yaml
id: my_doc                         # unique slug — shown in "Pick a document"
title: "My Document"
author: "Your Name"
year: "2024"
genre: "notes"                     # free-form
source: "course_pack"              # free-form
folder: "my_doc"                   # must match the folder name
ocr_file: "ocr_output.json"        # default if omitted
```

`ocr_output.json` is a flat JSON object mapping page keys to page text:

```json
{
  "page_1": "First page of your document text goes here...",
  "page_2": "Second page continues here..."
}
```

One page is fine to start — split longer documents into as many `page_N` keys as you like. See [`knowledge_base/README.md`](./knowledge_base/README.md) for the full format.

### 3b. Build the index

```bash
python tests/quick_test.py
```

This single command:

1. Checks that project modules import cleanly.
2. Prints config paths and lists the active corpus.
3. Shows which providers have keys configured.
4. Runs the setup pipeline (`scripts.setup_system`) to chunk + embed the corpus — **skips if an index already exists**.
5. Loads the FAISS index and reports vector count.
6. Tells you what to do next.

You'll see something like:

```text
  Active corpus: 1 document(s)
    - my_doc  [my_doc]
  ✓ Corpus loaded

  ✓ HuggingFace  key set

  Setup completed in 4.1s → OK
  Index loaded: True
  Vectors    : 12
  ✓ Lab is ready.
```

If you see `✗ No active corpus`, double-check `knowledge_base/my_doc/meta.yaml` exists and has `id` and `folder` set.

> First run is slow (~1–3 minutes) because the embedding model (`all-MiniLM-L6-v2`, ~90 MB) downloads once. Later runs are much faster.

After adding or changing documents, re-embed with:

```bash
python tests/quick_test.py --rebuild
```

---

## Step 4 — Chat

```bash
python -m commands.cli
```

You'll land on the **main menu** first — not directly in chat. That's intentional: switch provider, pin a document, toggle RAG, and more are one arrow-key selection away.

```text
Start chatting
Switch provider
Switch mode (general / text_specific)
Pick a document to focus on
Toggle RAG on/off
Session info / stats
View corpus (and how to add your own document)
Save conversation
Run doctor (diagnose a setup step)
Quit
```

| Menu item | What it does |
|---|---|
| **Start chatting** | Enter the chat loop — type a message and press Enter. |
| **Switch provider** | Change the active LLM (only providers with a key in `.env` are listed). |
| **Switch mode** | `general` (whole corpus) ↔ `text_specific` (focused on pinned document(s)). |
| **Pick a document to focus on** | Pin the session to one document, or clear the pin. |
| **Toggle RAG on/off** | Answer with retrieved context vs. the model alone. |
| **Session info / stats** | Session details (id, mode, RAG, filter, provider) + lab stats. |
| **View corpus** | Lists active documents and shows how to add your own. |
| **Save conversation** | Save the transcript to `data/conversations/`. |
| **Run doctor** | Step-by-step diagnostics — see [Troubleshooting](#troubleshooting). |
| **Quit** | Save and exit. |

Select **Start chatting** and try:

```text
You: Hello
```

You should get a context-grounded answer about the loaded corpus.

### Chat shortcuts

Inside the chat loop:

| Shortcut | Action |
|---|---|
| `/menu` | Back to the main menu |
| `/works` | Corpus list + add-a-document guide |
| `/quit` or `/exit` | Save and exit |
| `/help` | Reminder of shortcuts |

Responses show source citations and which provider answered. Conversations are also logged automatically to `data/conversations/sessions/*.json`.

### Modes and RAG

- **General mode** uses `prompts/general_mode.txt` and retrieves across the whole corpus.
- **Text-specific mode** uses `prompts/text_specific_mode.txt` and works best with a document pinned via **Pick a document to focus on**.
- **RAG off** sends your question to the LLM without retrieved context — useful to compare grounded vs. ungrounded answers.

---

## Step 5 — Customize (optional)

### 5a. Edit the prompts

Four prompt files live in `prompts/`:

| File | What it controls |
|---|---|
| `prompts/general_mode.txt` | Voice in *general* mode. **Edit this first.** |
| `prompts/text_specific_mode.txt` | Voice when focused on specific document(s). |
| `prompts/system.txt` | Default system-layer voice. |
| `prompts/summary_mode.txt` | Voice for structured summaries. |

Open `prompts/general_mode.txt` in any text editor. Replace its body with whatever you want the agent to sound like. Keep the `[citation: #]` format if you want the agent to cite retrieved passages.

**Tutor for an intro CS class**

```text
You are a friendly teaching assistant for an introductory CS course.
Answer using the course notes loaded in your knowledge base.
When citing, use the format [citation: #].
If the answer isn't in the loaded material, say so — don't guess.
```

**Internal help-desk bot**

```text
You are Acme Co.'s help-desk assistant.
Use the FAQ documents in the knowledge base to answer.
Cite the FAQ entry with [citation: #].
If the user asks something outside the FAQ, suggest they email
support@acme.example rather than guessing.
```

Save, quit the CLI, and restart (`python -m commands.cli`). Prompts load at startup.

### 5b. Add more documents

Repeat the pattern from [step 3a](#3a-add-a-document): create another `knowledge_base/<doc_id>/` folder with its own `meta.yaml` and `ocr_output.json`, then rebuild:

```bash
python tests/quick_test.py --rebuild
python -m commands.cli
```

Every folder with a valid `meta.yaml` becomes part of the active corpus and shows up in **Pick a document to focus on**.

### 5c. Tune retrieval (optional)

Section 4 of `.env.example` exposes settings you can adjust without touching Python:

| Variable | Default | What it does |
|---|---|---|
| `RETRIEVAL_TOP_K` | 5 | How many chunks to retrieve per query |
| `SIMILARITY_THRESHOLD` | 0.4 | Minimum similarity score to include a chunk |
| `CHUNK_SIZE` | 512 | Characters per chunk during indexing |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive chunks |
| `TEMPERATURE` | 0.7 | LLM randomness |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer used for embeddings |

After changing chunk or embedding settings, run `python tests/quick_test.py --rebuild`.

---

## Where things live

```
.
├── prompts/                              # student-editable: agent voice
├── knowledge_base/                       # student-editable: your documents (no bundled sample)
├── tests/quick_test.py                   # setup + smoke test in one command
├── commands/
│   ├── cli.py                            # menu-driven chatbot
│   ├── doctor.py                         # interactive doctor: menu + LLM-explained failures
│   └── doctor_checks.py                  # the PASS/FAIL checks doctor.py runs (no UI)
├── scripts/setup_system.py               # builds the FAISS index
├── config/                               # loaders (edit prompts in prompts/, not here)
├── services/                             # chatbot, retrieval, sessions, logger
├── models/                               # FAISS store, answer generator, LLM providers
├── utils/                                # embedding generator
└── data/                                 # runtime output — auto-created, gitignored:
                                            #   embeddings/, processed_texts/, conversations/
```

`data/` is created on first setup. You never need to `mkdir` it by hand.

---

## Troubleshooting

Stuck on one of the five steps? Run the doctor:

```bash
python -m commands.doctor
```

Pick the step you're on. It runs PASS/FAIL checks and — if you have a provider key — asks that LLM to explain failures. You can also reach it from the CLI via **Run doctor**.

Two files split the work:

- **`commands/doctor.py`** — the interactive part: the step-picker menu, PASS/FAIL printing, and (if a provider key is set) asking the LLM to explain failures. This is what `python -m commands.doctor` runs.
- **`commands/doctor_checks.py`** — the actual checks, as plain functions with no UI or LLM calls. Each returns a list of `CheckResult`s (name, ok, detail, fix hint). It's dependency-light on purpose, so it still works to diagnose problems even when the rest of the system (embeddings, vector store, etc.) is broken.

| Doctor step | What it checks |
|---|---|
| 1. Install | Python 3.9+, required packages import |
| 2. Add an API key | `.env` exists, at least one provider key set |
| 3. Build the FAISS index | Corpus loads (from `knowledge_base/`), index file exists and has vectors |
| 4. Chat | `ChatbotService` initializes, session starts |
| 5. Customize | Prompt files exist and are non-empty, corpus present |

### Common issues

**No API keys configured**
Edit `.env` and uncomment one provider line with your key.

**Setup is slow on first run**
The embedding model downloads once (~90 MB). Subsequent runs are seconds.

**No active corpus**
There's no bundled sample — add `knowledge_base/<doc_id>/meta.yaml` with a matching `ocr_output.json` (see [step 3a](#3a-add-a-document)).

**FAISS index missing or empty**
Run `python tests/quick_test.py --rebuild`.

**Changed corpus but answers still reference old material**
Rebuild the index (`--rebuild`) and restart the CLI.

**Wrong LLM provider at startup**
Set `DEFAULT_LLM_PROVIDER` in `.env`, or use **Switch provider** in the menu.

**Provider listed but chat fails**
Check the key is valid, you have network access, and the provider isn't rate-limited. The doctor's step 4 can run a live round-trip test.

**I broke something**
Run `git diff`. The only files you should have changed are `prompts/*.txt`, `knowledge_base/*`, and `.env`. Everything else should be untouched.

---

## Quick reference

```bash
# Setup
cp .env.example .env          # add a key, then:
python tests/quick_test.py      # build index (first time)
python tests/quick_test.py --rebuild   # after corpus changes

# Run
python -m commands.cli          # chat
python -m commands.doctor       # diagnose a setup step
```
