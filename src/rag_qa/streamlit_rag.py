import os
import sys
from pathlib import Path
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document


from rag_pipeline import (
        load_pdfs,
        chunk_documents,
        build_vector_store,
        get_retriever,
        build_qa_chain,
        answer_question,
        get_api_key,
    )


# Load env in case not already
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

st.set_page_config(page_title="RAG PDF QA", page_icon="📄", layout="wide")
st.title("📄 PDF Question & Answer ")

with st.sidebar:
    st.header("Settings & Data")
    st.markdown("### Models")
    embed_model = st.text_input("Embedding Model", value="models/text-embedding-004", help="Gemini embedding model name")
    chat_model = st.text_input("Chat Model", value="gemini-2.0-flash", help="Gemini chat model name")
    temperature = st.slider("Chat Temperature", 0.0, 1.0, 0.2, 0.05, help="Creativity of the model response")
    st.markdown("### Chunking")
    chunk_size = st.number_input("Chunk Size", min_value=200, max_value=4000, value=1000, step=100, help="Max characters per chunk")
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=1000, value=150, step=10, help="Overlap between chunks")
    k = st.slider("Top-K Chunks", 1, 10, 4, help="How many chunks to retrieve")

    st.markdown("### PDF Upload")
    uploaded_files = st.file_uploader("Select PDF(s)", type=["pdf"], accept_multiple_files=True, help="You can upload multiple PDFs")

    if st.button("Clear Vector Store"):
        for key in ["vector_store", "retriever", "qa_builder"]:
            st.session_state.pop(key, None)
        st.success("Cleared in-memory store")

    st.markdown("---")
    st.caption("After uploading, click the build button in main area.")

api_status = st.empty()
try:
    _ = get_api_key()
    api_status.success("API key loaded")
except Exception as e:  # pragma: no cover
    api_status.error(str(e))

 # no sample preload

build_col, _ = st.columns([1, 3])
with build_col:
    if st.button("Build / Rebuild Vector Store", type="primary"):
        pdf_paths = []
        temp_files = []
        for up in uploaded_files or []:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(up.read())
            tmp.flush()
            temp_files.append(tmp)
            pdf_paths.append(Path(tmp.name))
        # sample disabled

        if not pdf_paths:
            st.warning("Upload or select a sample PDF first.")
        else:
            with st.spinner("Loading & chunking PDFs ..."):
                docs = load_pdfs(pdf_paths)
                chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            with st.spinner("Building vector store ..."):
                try:
                    store = build_vector_store(chunks, embed_model=embed_model)
                except Exception as e:
                    st.error(f"Failed building store: {e}")
                else:
                    st.session_state.vector_store = store
                    st.session_state.retriever = get_retriever(store, k=k)
                    st.session_state.qa_builder = build_qa_chain(chat_model=chat_model, temperature=temperature)
                    st.success(f"Vector store ready with {len(chunks)} chunks")
        # Cleanup temps
        for t in temp_files:
            try:
                t.close()
                os.unlink(t.name)
            except OSError:
                pass

if "vector_store" not in st.session_state:
    st.info("Upload PDFs and click 'Build / Rebuild Vector Store' to start.")
    st.stop()

# Initialize chat history structure
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # list of dicts: {question, answer, sources}

st.markdown("### Chat")

# Render past conversation
for entry in st.session_state.qa_history:
    with st.chat_message("user"):
        st.markdown(entry["question"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])
        with st.expander("Sources", expanded=False):
            for i, d in enumerate(entry["sources"], 1):
                meta = d.metadata
                st.markdown(f"**Chunk {i}** - page: {meta.get('page', 'n/a')} source: {meta.get('source', 'unknown')}")
                st.code(d.page_content[:800] + ("..." if len(d.page_content) > 800 else ""))

user_question = st.chat_input("Ask about the loaded documents...")
if user_question:
    q = user_question.strip()
    if q:
        retriever = st.session_state.get("retriever")
        qa_builder = st.session_state.get("qa_builder")
        if not retriever or not qa_builder:
            st.error("Vector store not ready yet.")
        else:
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, source_docs = answer_question(q, retriever, qa_builder)
                    except Exception as e:  # pragma: no cover
                        st.error(str(e))
                        answer = f"Error: {e}"
                        source_docs = []
                st.markdown(answer)
                with st.expander("Sources", expanded=False):
                    for i, d in enumerate(source_docs, 1):
                        meta = d.metadata
                        st.markdown(f"**Chunk {i}** - page: {meta.get('page', 'n/a')} source: {meta.get('source', 'unknown')}")
                        st.code(d.page_content[:800] + ("..." if len(d.page_content) > 800 else ""))
            # Persist turn
            st.session_state.qa_history.append({
                "question": q,
                "answer": answer,
                "sources": source_docs,
            })
