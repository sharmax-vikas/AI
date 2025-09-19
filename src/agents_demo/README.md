LangChain Agents — Demo
=======================

This demo shows how to build and run LangChain agents that use tools with reasoning patterns.

Features
--------

- Agent types and reasoning (tool-calling agent with ReAct-like prompt)
- Tool integration and custom tools
- Agent executors and callbacks (verbose mode prints to server logs)
- Built-in-like tools (web search, HTTP GET, calculator)

Tools
-----

- calculator: evaluate math expressions using numexpr
- web_search: DuckDuckGo search via langchain-community tool
- http_get: fetch a URL via requests
- (optional) ShellTool: disabled by default for safety

Run (Streamlit)
---------------

From project root:

```bash
streamlit run src/agents_demo/streamlit_agents.py
```

Usage
-----

1. Configure model and temperature (and whether to allow shell tool) in the sidebar.
2. Click "Initialize Agent".
3. Ask questions in the chat box; the agent will use tools when necessary.

Notes
-----

- Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable (.env supported).
- Tools run server-side. Shell tool can execute commands on the server; keep it disabled unless needed.
- For more complex reasoning, consider additional tools, structured outputs, or memory.
