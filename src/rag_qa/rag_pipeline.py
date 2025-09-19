import os
from pathlib import Path
from typing import List, Tuple, Optional
import asyncio
from dataclasses import dataclass
from dotenv import load_dotenv
import argparse
from typing import Iterable

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Load environment variables once
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

DEFAULT_EMBED_MODEL = "models/text-embedding-004"
DEFAULT_CHAT_MODEL = "gemini-2.0-flash"

SYSTEM_RAG = """You are a helpful AI assistant. Use ONLY the provided context to answer the user's question. If the answer is not in the context, say you don't have enough information. Be concise.
""".strip()

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RAG),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
])


@dataclass
class RAGResources:
    vector_store: FAISS
    embeddings_model_name: str
    chunk_size: int
    chunk_overlap: int


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment or .env")
    return key


def load_pdfs(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        if not p.exists():
            continue
        try:
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        except Exception as e:  # pragma: no cover - robust ingestion
            print(f"Failed to load {p}: {e}")
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ".", "?", "!", " "]
    )
    return splitter.split_documents(docs)


def _ensure_event_loop():  # Handles cases where library lazily expects a running loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Create and set a new loop for the current (Streamlit) thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    return None


def build_vector_store(
    docs: List[Document],
    *,
    api_key: Optional[str] = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> FAISS:
    if not docs:
        raise ValueError("No documents provided to build vector store")
    key = api_key or get_api_key()
    _ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=key)
    return FAISS.from_documents(docs, embedding=embeddings)


def get_retriever(store: FAISS, k: int = 4):
    return store.as_retriever(search_kwargs={"k": k})


def format_docs(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta_line = f"[Doc {i} source={d.metadata.get('source', 'unknown')}]"
        parts.append(meta_line + "\n" + d.page_content.strip())
    return "\n\n".join(parts)


def build_qa_chain(*, api_key: Optional[str] = None, chat_model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.2):
    key = api_key or get_api_key()
    _ensure_event_loop()
    llm = ChatGoogleGenerativeAI(model=chat_model, google_api_key=key, temperature=temperature)

    def _combine(inputs):
        question = inputs["question"]
        docs = inputs["docs"]
        return {"question": question, "context": format_docs(docs)}

    # Parallel branch: retrieve docs & passthrough question
    # (retriever will be added dynamically outside when invoked)
    def chain_with_retriever(retriever):
        retrieval = RunnableParallel(docs=retriever, question=RunnablePassthrough()) | _combine | RAG_PROMPT | llm
        return retrieval

    return chain_with_retriever


def answer_question(question: str, retriever, qa_builder) -> Tuple[str, List[Document]]:
    # Retrieve documents (invoke is the modern API; fallback to legacy if needed)
    try:
        docs: List[Document] = retriever.invoke(question)  # type: ignore
    except Exception:
        # Legacy fallback
        docs = retriever.get_relevant_documents(question)  # type: ignore
    chain = qa_builder(retriever)
    response = chain.invoke(question)
    return response.content if hasattr(response, 'content') else str(response), docs


__all__ = [
    "RAGResources",
    "load_pdfs",
    "chunk_documents",
    "build_vector_store",
    "get_retriever",
    "build_qa_chain",
    "answer_question",
]


# --------------------------
# CLI runner for terminal use
# --------------------------

def _parse_args(argv: Optional[Iterable[str]] = None):
    parser = argparse.ArgumentParser(
        description="RAG PDF QA (Gemini) — load PDFs, build index, then ask questions."
    )
    parser.add_argument(
        "--pdf",
        nargs="*",
        help="Path(s) to PDF files (space-separated). If omitted, you'll be prompted interactively.",
    )
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Embedding model (Gemini)")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Chat model (Gemini)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Chat temperature (0.0-1.0)")
    parser.add_argument("--k", type=int, default=4, help="Top-K chunks to retrieve per question")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitter")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Chunk overlap for splitter")
    parser.add_argument("--no-sources", action="store_true", help="Do not print sources after answers")
    return parser.parse_args(argv)


def _prompt_for_pdfs() -> List[Path]:
    print("Enter one or more PDF file paths (comma-separated) or drag-and-drop paths here, then press Enter.")
    print("Example: C:/docs/a.pdf, D:/stuff/b.pdf")
    while True:
        raw = input("PDF path(s): ").strip().strip('"')
        if not raw:
            print("Please provide at least one path.")
            continue
        candidates = [s.strip().strip('"') for s in raw.split(",") if s.strip()]
        paths: List[Path] = [Path(p) for p in candidates]
        missing = [str(p) for p in paths if not p.exists()]
        wrong = [str(p) for p in paths if p.suffix.lower() != ".pdf"]
        if missing:
            print("These paths do not exist:")
            for m in missing:
                print(" -", m)
            continue
        if wrong:
            print("These are not PDF files:")
            for w in wrong:
                print(" -", w)
            continue
        return paths


def _build_resources(paths: List[Path], *, embed_model: str, chat_model: str, temperature: float, k: int, chunk_size: int, chunk_overlap: int):
    key = get_api_key()  # early check
    print("Loading and splitting PDFs ...")
    docs = load_pdfs(paths)
    if not docs:
        raise SystemExit("No documents were loaded from the provided PDFs.")
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Built {len(chunks)} chunks.")
    print("Building vector store (FAISS) ...")
    store = build_vector_store(chunks, api_key=key, embed_model=embed_model)
    retriever = get_retriever(store, k=k)
    qa_builder = build_qa_chain(api_key=key, chat_model=chat_model, temperature=temperature)
    return retriever, qa_builder


def _print_sources(docs: List[Document], max_chars: int = 800):
    print("\nSources:")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "n/a")
        text = d.page_content.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        print(f"[{i}] page={page} source={src}\n{text}\n")


def main(argv: Optional[Iterable[str]] = None):  # pragma: no cover - CLI path
    args = _parse_args(argv)
    try:
        get_api_key()
    except Exception as e:
        print("ERROR:", e)
        return 1

    # Collect PDF paths
    pdf_paths: List[Path]
    if args.pdf:
        pdf_paths = [Path(p) for p in args.pdf]
        bad = [str(p) for p in pdf_paths if not (p.exists() and p.suffix.lower() == ".pdf")]
        if bad:
            print("Some provided paths are invalid or not PDFs:")
            for b in bad:
                print(" -", b)
            pdf_paths = []
    else:
        pdf_paths = []

    if not pdf_paths:
        pdf_paths = _prompt_for_pdfs()

    # Build retrieval resources
    try:
        retriever, qa_builder = _build_resources(
            pdf_paths,
            embed_model=args["embed_model"] if isinstance(args, dict) else args.embed_model,
            chat_model=args["chat_model"] if isinstance(args, dict) else args.chat_model,
            temperature=args["temperature"] if isinstance(args, dict) else args.temperature,
            k=args["k"] if isinstance(args, dict) else args.k,
            chunk_size=args["chunk_size"] if isinstance(args, dict) else args.chunk_size,
            chunk_overlap=args["chunk_overlap"] if isinstance(args, dict) else args.chunk_overlap,
        )
    except Exception as e:
        print("Failed to build resources:", e)
        return 2

    print("\nRAG PDF QA ready. Type your questions. Commands: /exit, /quit, /reload to load new PDFs.")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not q:
            continue
        low = q.lower()
        if low in {"/exit", "/quit", "exit", "quit"}:
            print("Bye!")
            break
        if low in {"/reload", "/files"}:
            pdf_paths = _prompt_for_pdfs()
            try:
                retriever, qa_builder = _build_resources(
                    pdf_paths,
                    embed_model=args.embed_model,
                    chat_model=args.chat_model,
                    temperature=args.temperature,
                    k=args.k,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )
            except Exception as e:
                print("Failed to rebuild resources:", e)
            continue

        try:
            answer, docs = answer_question(q, retriever, qa_builder)
        except Exception as e:
            print("Error answering:", e)
            continue
        print("\nAssistant:")
        print(answer)
        if not args.no_sources:
            _print_sources(docs)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
