# Simple Chatbot Module

This folder contains two entry points for interacting with a Gemini (Google Generative AI) model via LangChain:

| File | Purpose |
|------|---------|
| `simple_chatbot.py` | Console (CLI) chat using a prompt → LLM → output parser chain with manual history. |
| `streamlit_app.py`  | Web UI (Streamlit) that reuses the same chain logic for a richer chat interface. |

## 1. Requirements
- Python 3.12+
- An API key set as `GEMINI_API_KEY` (preferred) or `GOOGLE_API_KEY`.
- Dependencies installed (managed by `pyproject.toml`).

Set your key (PowerShell, persistent for new terminals):
```powershell
setx GEMINI_API_KEY "YOUR_KEY"
```
(Then open a new terminal.)

Or create a project‐root `.env` file (one level above this folder):
```
GEMINI_API_KEY=YOUR_KEY
```

Install dependencies (from project root):
```powershell
uv sync
```

## 2. CLI Chat (`simple_chatbot.py`)
Run from project root:
```powershell
uv run python -m src.simple_chatbot.simple_chatbot
```
Or (inside the `src` directory):
```powershell
uv run python -m simple_chatbot.simple_chatbot
```
Direct execution (not recommended) will also work:
```powershell
uv run python src/simple_chatbot/simple_chatbot.py
```

### CLI Commands
- `/reset` – Clear conversation history
- `/trim` – Keep only the last 10 messages
- `exit`, `quit`, `q` – Exit the program

## 3. Streamlit Web UI (`streamlit_app.py`)
Launch from project root:
```powershell
uv run streamlit run src/simple_chatbot/streamlit_app.py
```
Or from inside `src`:
```powershell
uv run streamlit run simple_chatbot/streamlit_app.py
```

### Features
- In‑session chat history (not persisted)
- Reset button in sidebar
- Reuses the same chain logic as the CLI (single source of truth)

## 4. Internal Architecture
Both interfaces share the same chain built in `build_chain()`:
```
ChatPromptTemplate (system + history + human)
   |-> ChatGoogleGenerativeAI (Gemini model: gemini-2.0-flash)
        |-> StrOutputParser (returns plain string)
```
History is a Python list of LangChain message objects (`HumanMessage` / `AIMessage`) manually appended each turn.

## 5. Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| ImportError on `simple_chatbot` | Running from unexpected cwd | Use project root + module form (`python -m ...`) |
| API key error | Env var not loaded | Add to `.env` or run `setx` and reopen terminal |
| No responses / quota errors | Rate limits | Wait or switch plan/model |

## 6. Extending
Ideas:
- Add streaming token output (Streamlit placeholder updates)
- Persist chat to `logs/` as JSONL
- Add model + temperature selectors in the sidebar
- Add `/save` command in CLI

---
Maintained under `src/simple_chatbot/`. Keep both interfaces in sync by modifying only `build_chain()` logic when changing prompt/model.
