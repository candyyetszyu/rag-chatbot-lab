# prompts/

All editable prompt files for the RAG Chatbot Lab. The CLI loads these at startup; restart the CLI to pick up changes.

| File | Used for |
|---|---|
| `system.txt` | Default system-layer voice (rarely overridden) |
| `general_mode.txt` | Voice + scope for `mode = general` (whole-corpus discussion) |
| `text_specific_mode.txt` | Voice + scope for `mode = text_specific` (focused documents) |
| `summary_mode.txt` | Voice for full-text structured summaries |

Replace any file with prose that fits your domain. Keep it plain UTF-8.
