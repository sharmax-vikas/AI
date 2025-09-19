import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Prefer relative import; fallback to package import if running as a script
try:
    from agents import AgentConfig, build_agent  # type: ignore
except Exception:
    # Ensure project root/src is on sys.path, then import via package name
    try:
        project_root = Path(__file__).resolve().parents[2]  # <repo>/
        src_dir = project_root / "src"
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        from agents_demo.agents import AgentConfig, build_agent  # type: ignore
    except Exception as _e:
        raise

# Load env
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

st.set_page_config(page_title="LangChain Agents Demo", page_icon="🧠", layout="wide")
st.title("🧠 LangChain Agents — Tools & Reasoning")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value="gemini-2.0-flash")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    if st.button("Initialize Agent", type="primary"):
        st.session_state.agent = build_agent(AgentConfig(model=model, temperature=temperature))
        st.success("Agent ready")

    with st.expander("Diagnostics", expanded=False):
        st.caption("Useful if imports or tools don't show up as expected.")
        st.code(f"Python: {sys.executable}")
        if "agent" in st.session_state:
            tools = getattr(st.session_state.agent, "tools", [])
            st.write({
                "tool_names": [getattr(t, "name", type(t).__name__) for t in tools],
            })

if "agent" not in st.session_state:
    st.info("Configure and initialize the agent from the sidebar.")
    st.stop()

# Chat history
if "agent_history" not in st.session_state:
    st.session_state.agent_history = []

# Render chat
from langchain_core.messages import HumanMessage, AIMessage
for turn in st.session_state.agent_history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])  # intermediate steps are printed to server logs via verbose=True

user_input = st.chat_input("Ask the agent... (it can use web_search, calculator, http_get)")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking with tools..."):
            try:
                # Convert prior turns to LangChain chat history messages
                chat_history = []
                for t in st.session_state.agent_history:
                    chat_history.append(HumanMessage(content=t["question"]))
                    chat_history.append(AIMessage(content=t["answer"]))
                result = st.session_state.agent.invoke({"input": user_input, "chat_history": chat_history})
                answer = result.get("output", str(result))
            except Exception as e:
                answer = f"Error: {e}"
        st.markdown(answer)
    st.session_state.agent_history.append({"question": user_input, "answer": answer})
