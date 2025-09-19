RAG PDF QA (Gemini + LangChain)
================================

This module provides a Retrieval-Augmented Generation (RAG) system over your PDFs using:

- Embeddings: Google Gemini text-embedding-004
- Vector store: FAISS (in-memory)
- LLM: Google Gemini chat models
- Framework: LangChain

Two ways to use it:

- Streamlit UI: `streamlit_rag.py`
- Terminal CLI: `rag_pipeline.py`

Prerequisites
-------------

- Python 3.12+
- API key: set either `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment or in a project-level `.env` file.

Example `.env` (place in project root):

```bash
GEMINI_API_KEY=your_api_key_here
```

Install
-------

From the project root directory:

Using pip (recommended):

```bash
pip install -e .
```

Or using uv:

```bash
uv pip install -e .
```

If you hit missing packages later, install explicitly:

```bash
pip install pypdf faiss-cpu
```

Streamlit App (UI)
------------------

Launch the app from project root:

```bash
streamlit run src/rag_qa/streamlit_rag.py
```

In the app:

1. Upload one or more PDFs from the left sidebar.
2. Adjust chunking (size/overlap), Top-K, and model settings if needed.
3. Click "Build / Rebuild Vector Store".
4. Use the chat input to ask questions. Answers include an expandable Sources section.

Notes:

- The index is in-memory; rebuilding will clear previous data.
- Model and temperature can be configured in the sidebar.

Terminal CLI
------------

Run the CLI from project root:

```bash
python src/rag_qa/rag_pipeline.py
```

You will be prompted to provide PDF paths (comma-separated). After building, type questions repeatedly. Commands:

- `/reload` to load different PDFs and rebuild
- `/exit` or `/quit` to end

CLI flags (optional):

```bash
python src/rag_qa/rag_pipeline.py \
  --pdf "D:/docs/one.pdf" "D:/docs/two.pdf" \
  --embed-model "models/text-embedding-004" \
  --chat-model "gemini-2.0-flash" \
  --temperature 0.2 \
  --k 4 \
  --chunk-size 1000 \
  --chunk-overlap 150 \
  --no-sources
```

Architecture Overview
---------------------

- `load_pdfs(paths)`: Loads PDFs into `Document` objects (via `PyPDFLoader`).
- `chunk_documents(docs, chunk_size, chunk_overlap)`: Recursively splits text.
- `build_vector_store(chunks, embed_model)`: Embeds with Gemini and builds a FAISS index.
- `get_retriever(store, k)`: Wraps FAISS for top-k retrieval.
- `build_qa_chain(chat_model, temperature)`: Creates a simple prompt + Gemini chat chain.
- `answer_question(question, retriever, qa_builder)`: Retrieves chunks and generates an answer + sources.

Configuration
-------------

- Embeddings model (default): `models/text-embedding-004`
- Chat model (default): `gemini-2.0-flash`
- Temperature (default): `0.2`
- Chunk size/overlap: configurable per use case
- Top-K: number of chunks retrieved per question

Troubleshooting
---------------

- Missing `pypdf` or `faiss-cpu`:

  ```bash
  pip install pypdf faiss-cpu
  ```

- Import error in Streamlit regarding `rag_qa`:
  - Launch exactly with: `streamlit run src/rag_qa/streamlit_rag.py`
  - The app includes fallbacks to add `src` to `sys.path`.
- "There is no current event loop" error:
  - Handled internally by creating a loop in the thread; ensure you’re on the updated code and restart the app.
- FAISS on Windows:
  - Use `faiss-cpu` wheels (`pip install faiss-cpu`). If installation fails, consider using an alternate vector store (e.g., Chroma) as a fallback.
- PDF parsing warnings (from `pypdf`):
  - Some PDFs contain malformed objects. Usually warnings are harmless; if loading fails, try a different copy of the file.

Caveats
-------

- The FAISS index is not persisted by default. Rebuilding is required after restart. You can extend this with `FAISS.save_local`/`FAISS.load_local` if desired.
- The model is instructed to answer from context only, but hallucinations are still possible.
- Costs and token usage are not shown; you can add callbacks to track them.

Project Files
-------------

- `streamlit_rag.py`: Streamlit user interface for RAG over PDFs.
- `rag_pipeline.py`: Core pipeline and a terminal-friendly CLI entry point.

License
-------

Add your license here.
