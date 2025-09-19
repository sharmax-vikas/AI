AI Project
==========

This repository includes:

1. Simple Gemini chat (`simple_chatbot`)
2. PDF Retrieval-Augmented Generation (RAG) Question Answering app (`rag_qa`)
3. LangChain Agents demo with tools (`agents_demo`)

Both use Google Gemini models via `langchain`.

Prerequisites
--------------

* Python 3.12+
* Google / Gemini API key set as either `GEMINI_API_KEY` or `GOOGLE_API_KEY` in your environment or an `.env` file in the project root.

Example `.env`:

```bash
GEMINI_API_KEY=your_api_key_here
```

Install
--------

This project uses a `pyproject.toml`. You can install dependencies with your preferred tool.

Using pip:

```bash
pip install -e .
```

Or with `uv` (fast installer):

```bash
uv pip install -e .
```

Simple Chat CLI
----------------

```bash
python -m src.simple_chatbot.simple_chatbot
```

Exit with `exit`, `quit`, `/quit`. Use `/reset` to clear history.

Streamlit Chat App
-------------------

```bash
streamlit run src/simple_chatbot/streamlit_app.py
```

PDF RAG App
------------

Launch:

```bash
streamlit run src/rag_qa/streamlit_rag.py
```

Steps inside the UI:

1. Upload one or more PDF files.
2. Adjust chunk size / overlap / Top-K if desired.
3. Click "Build / Rebuild Vector Store".
4. Ask a question in the input field and click "Ask".
5. Expand "Sources" to view retrieved chunks.

Agents Demo
------------

Launch:

```bash
streamlit run src/agents_demo/streamlit_agents.py
```

Features:

* Tool-calling agent using Google Gemini via LangChain
* Conversational memory across turns
* Built-in tools:
  * `web_search` (DuckDuckGo via ddgs, with a LangChain fallback)
  * `calculator` (numexpr)
  * `http_get` (requests)

Usage inside the UI:

1. Set model and temperature in the sidebar and click "Initialize Agent".
2. Ask questions; the agent will decide when to use tools.

Notes:

* Internet access is required for `web_search` and `http_get`.
* If you change dependencies while Streamlit is running, restart Streamlit and re-initialize the agent.

Architecture (RAG)
-------------------

`rag_pipeline.py` functions:

* `load_pdfs` – uses `PyPDFLoader` to turn PDFs into `Document`s.
* `chunk_documents` – splits text with a recursive splitter.
* `build_vector_store` – embeds with `text-embedding-004` (Gemini) into a FAISS index.
* `get_retriever` – returns a retriever wrapper.
* `build_qa_chain` – builds a lightweight prompt + Gemini chat model chain.
* `answer_question` – orchestrates retrieval + generation, returns answer and source docs.

Environment Variables
----------------------

* `GEMINI_API_KEY` or `GOOGLE_API_KEY` – authentication.

Adjusting Models
-----------------

Change the default embedding or chat model names in the Streamlit sidebar (RAG) or in the source constants if needed.

Limitations / Notes
--------------------

* FAISS index is in-memory only; every rebuild clears previous data.
* No persistence layer implemented yet.
* Basic prompt instructs the model not to hallucinate; still may happen.
* Token counting / cost tracking not included (could be added with callbacks).
* Agents demo excludes a shell tool for safety/stability.

Future Ideas
-------------

* Persist FAISS index to disk (`FAISS.save_local`).
* Add multi-query or re-ranking retrieval.
* Support other file types (Markdown, text, HTML).
* Add evaluation scripts for retrieval quality.

