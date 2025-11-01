from __future__ import annotations

import asyncio
import json
import random
import sys
from typing import Any

from cowsay_mcp import run_cowsay
from cowsay_mcp.server import server

from .llm import chat_once, load_model
from .prompting import POEM_ANALYST_PROMPT, THEMES, build_initial_messages

"""CLI entrypoint for running the cowsay tool-calling demo."""


def parse_tool_call(raw_response: str, expected_tool: str) -> str:
    """Extract the poem text from the assistant's JSON tool call."""

    start = raw_response.find("{")
    if start == -1:
        raise ValueError("No JSON found in LLM response")

    data, _ = json.JSONDecoder().raw_decode(raw_response[start:])
    if data.get("tool") != expected_tool:
        raise ValueError(f"Unexpected tool requested: {data.get('tool')}")

    text = data.get("args", {}).get("text", "").strip()
    if not text:
        raise ValueError("Tool call did not include text")
    return text


def normalize_explanation(text: str) -> str:
    """Remove protocol markers and trim whitespace."""

    cleaned = text.strip()
    for marker in ("SYSTEM:", "USER:", "ASSISTANT:"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()
    return cleaned


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

    explanation = normalize_explanation(
        chat_once(
            bundle,
            [
                {"role": "system", "content": POEM_ANALYST_PROMPT},
                {"role": "user", "content": poem_text},
            ],
            temperature=0.0,
        )
    )
    print("\nPoem explanation:")
    print(explanation)


if __name__ == "__main__":
    main()
