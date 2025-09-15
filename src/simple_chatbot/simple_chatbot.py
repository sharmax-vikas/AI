
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "../../", ".env"))

SYSTEM_PROMPT = "You are a concise, helpful AI assistant."
MODEL_NAME = "gemini-2.0-flash"
TEMPERATURE = 0.7
MAX_HISTORY = 40


def get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Set GEMINI_API_KEY or GOOGLE_API_KEY (env or .env file)")
    return key


def build_chain():
    """Build and return the runnable chain: prompt | llm | parser."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=get_api_key(), temperature=TEMPERATURE)
    parser = StrOutputParser()
    return prompt | llm | parser


def chat_loop(chain) -> None:
    history: List[BaseMessage] = []  # we do not resend system each turn; prompt supplies it
    print("Simple Gemini Chat (chain). Type 'exit' to quit. Commands: /reset /trim /quit")
    while True:
        try:
            user = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not user:
            continue
        lower = user.lower()
        if lower in {"exit", "quit", "q", "/quit"}:
            print("Bye!")
            break
        if lower == "/reset":
            history.clear()
            print("[History cleared]")
            continue
        if lower == "/trim":
            if len(history) > 10:
                history[:] = history[-10:]
            print(f"[History length: {len(history)}]")
            continue

        history.append(HumanMessage(content=user))
        # Trim if exceeding max (keep most recent N)
        if len(history) > MAX_HISTORY:
            history[:] = history[-MAX_HISTORY:]

        # Invoke chain with current history
        output = chain.invoke({"history": history, "input": user})
        # output is a string (parser result); add to history as AIMessage
        print(f"Bot: {output}")
        history.append(AIMessage(content=output))


def main():  # pragma: no cover
    chain = build_chain()
    chat_loop(chain)


if __name__ == "__main__":  # pragma: no cover
    main()
