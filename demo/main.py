from __future__ import annotations

import asyncio
import json
import random
import sys
from typing import Any

from cowsay_mcp import run_cowsay
from cowsay_mcp.server import server

from .llm import chat_once, load_model
from .prompting import THEMES, build_initial_messages, explain_poem, parse_tool_call

"""CLI entrypoint for running the cowsay tool-calling demo."""


def fetch_primary_tool() -> Any:
    """Return the first registered tool from the FastMCP server."""

    tools = asyncio.run(server.get_tools())
    try:
        return next(iter(tools.values()))
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("No tools registered on the cowsay MCP server.") from exc


def main() -> None:
    """LLM selects and executes a tool via MCP."""

    bundle = load_model("mlx-community/Qwen3-8B-4bit")
    tool_spec = fetch_primary_tool()

    theme = random.choice(THEMES)
    messages = build_initial_messages(theme, tool_spec)
    raw_response = chat_once(bundle, messages, temperature=1.0)

    print(f"LLM raw response: {raw_response!r}", file=sys.stderr)

    try:
        poem_text = parse_tool_call(raw_response, tool_spec.name)
    except ValueError as exc:
        print(f"LLM output: {raw_response!r}", file=sys.stderr)
        sys.exit(f"Invalid tool call: {exc}")

    tool_call_json = json.dumps({"tool": tool_spec.name, "args": {"text": poem_text}})
    result = run_cowsay(poem_text)

    print("LLM selected tool:", tool_call_json)
    print("\nTool executed result:")
    print(result)

    explanation = explain_poem(bundle, poem_text)
    print("\nPoem explanation:")
    print(explanation)


if __name__ == "__main__":
    main()
