import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# Import chain builder from existing CLI module
try:
    # If run as a module: python -m simple_chatbot.streamlit_app
    from .simple_chatbot import build_chain, MAX_HISTORY  # type: ignore
except ImportError:
    try:
        # If run as streamlit run src/simple_chatbot/streamlit_app.py (cwd = src)
        from simple_chatbot import build_chain, MAX_HISTORY  # type: ignore
    except ImportError:
        # If run as streamlit run simple_chatbot/streamlit_app.py (cwd = project root)
        from simple_chatbot.simple_chatbot import build_chain, MAX_HISTORY  # type: ignore

# Load env once (in case app started directly)
_env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv(_env_path)

st.set_page_config(page_title="Gemini Chat", page_icon="💬", layout="centered")
st.title("💬 Gemini Chatbot")

# Session state initialization
if "chain" not in st.session_state:
    st.session_state.chain = build_chain()
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Controls")
    if st.button("Reset Conversation", type="primary"):
        st.session_state.history.clear()
        st.success("History cleared")
    st.caption("Model + Prompt source: simple_chatbot.build_chain()")

# Render previous messages
for msg in st.session_state.history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Input box
user_input = st.chat_input("Ask something...")
if user_input:
    # Add user message
    st.session_state.history.append(HumanMessage(content=user_input))

    # Trim if exceeding max
    if len(st.session_state.history) > MAX_HISTORY:
        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare chain input and invoke
    try:
        output = st.session_state.chain.invoke({
            "history": st.session_state.history,
            "input": user_input,
        })
    except Exception as e:  # broad for UI friendliness
        output = f"Error: {e}"  # display error back to user

    with st.chat_message("assistant"):
        st.markdown(output)

    # Append AI response
    st.session_state.history.append(AIMessage(content=output))
