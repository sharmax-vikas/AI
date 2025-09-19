import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load env
load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"))

DEFAULT_MODEL = "gemini-2.0-flash"


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY in environment or .env")
    return key


# --- Custom Tools ---

@tool("calculator", return_direct=False)
def calculator(expr: str) -> str:
    """Evaluate a simple math expression using numexpr. Input: string expression like '2+2*5'."""
    import numexpr as ne
    try:
        val = ne.evaluate(expr)
        return str(val)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool("web_search", return_direct=False)
def web_search(query: str) -> str:
    """Run a DuckDuckGo search and return a compact list of results.
    Tries the duckduckgo_search library first; falls back to LangChain's DuckDuckGoSearchRun.
    """
    q = (query or "").strip()
    if not q:
        return "Search error: empty query"
    # First try: duckduckgo_search (more control, returns structured results)
    try:
        from duckduckgo_search import DDGS  # type: ignore
        lines = []
        with DDGS() as ddgs:
            # Prefer news for freshness; fallback to general text search
            results = list(ddgs.news(q, max_results=5))
            if not results:
                results = list(ddgs.text(q, max_results=5, region="wt-wt", safesearch="moderate"))
        for i, r in enumerate(results, 1):
            title = r.get("title") or r.get("source") or "Untitled"
            href = r.get("url") or r.get("href") or ""
            snippet = r.get("body") or r.get("excerpt") or r.get("content") or ""
            if snippet and len(snippet) > 240:
                snippet = snippet[:240] + "..."
            piece = f"{i}. {title}\n{href}\n{snippet}".strip()
            lines.append(piece)
        output = "\n\n".join(lines).strip()
        if not output:
            output = f"No results found for: {q}"
        # if len(output) > 3000:
        #     output = output[:3000] + "..."
        return output
    except Exception:
        pass
    # Fallback: LangChain tool wrapper
    try:
        from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
        runner = DuckDuckGoSearchRun()
        out = runner.run(q)
        return out or f"No results found for: {q}"
    except Exception as e:
        return f"Search error: {e}"


@tool("http_get", return_direct=False)
def http_get(url: str) -> str:
    """Fetch a URL using requests and return text (truncated)."""
    import requests
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        txt = r.text
        if len(txt) > 3000:
            txt = txt[:3000] + "..."
        return txt
    except Exception as e:
        return f"HTTP error: {e}"


 


@dataclass
class AgentConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.2


def build_agent(config: Optional[AgentConfig] = None):
    cfg = config or AgentConfig()
    llm = ChatGoogleGenerativeAI(model=cfg.model, google_api_key=get_api_key(), temperature=cfg.temperature)

    # Assemble tools
    tools = [calculator, web_search, http_get]

    # ReAct-style prompt with memory support (chat_history) and intermediate steps (agent_scratchpad)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful agent. You can use tools when needed. Think step-by-step. Format: Thought:, Action:, Action Input:, Observation:."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # Build an AgentExecutor-like simple loop using function calling
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    return executor


__all__ = [
    "AgentConfig",
    "build_agent",
]
