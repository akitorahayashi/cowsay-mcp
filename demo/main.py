from __future__ import annotations

import asyncio
import json
import random
import re
import sys
import textwrap
from typing import Any

from cowsay_mcp import run_cowsay
from cowsay_mcp.server import server

from .llm import chat_once, load_model

"""CLI entrypoint for running the cowsay tool-calling demo."""

THEMES = ["nature", "technology", "emotions", "adventure", "creativity"]

POEM_ANALYST_PROMPT = textwrap.dedent(
    """
    あなたは詩の解説者です。ユーザが渡す詩を題材に、テーマ・イメージ・感情的な余韻を3文以内の自然な日本語で丁寧にまとめてください。
    推論過程や指示は書かず、箇条書きも使わず、各文を番号や記号で始めないでください。最初の文は必ず「この詩は」で始めます。
    完成した文章だけを返してください。
    """
).strip()


def fetch_primary_tool() -> Any:
    """Return the first registered tool from the FastMCP server."""

    tools = asyncio.run(server.get_tools())
    try:
        return next(iter(tools.values()))
    except StopIteration as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("No tools registered on the cowsay MCP server.") from exc


def summarise_tags(tool: Any) -> str:
    """Generate a comma separated list of tags for display."""

    tags = getattr(tool, "tags", None)
    return ", ".join(sorted(tags)) if tags else "none"


def summarise_parameters(schema: dict[str, Any] | None) -> str:
    """Turn a JSON schema into a compact human-friendly description."""

    if not schema:
        return "No arguments required"

    properties = schema.get("properties", {}) or {}
    if not properties:
        return "No arguments required"

    parts: list[str] = []
    for name, spec in properties.items():
        details = spec or {}
        type_hint = details.get("type", "any")
        description = details.get("description")
        if description:
            parts.append(f"{name} ({type_hint}) - {description}")
        else:
            parts.append(f"{name} ({type_hint})")
    return "; ".join(parts)


def build_system_prompt(tool: Any) -> str:
    """Create the system prompt that reflects the actual tool metadata."""

    tags = summarise_tags(tool)
    args_summary = summarise_parameters(getattr(tool, "parameters", None))
    properties = (getattr(tool, "parameters", {}) or {}).get("properties", {}) or {}
    arg_keys = list(properties.keys())
    example_args = {name: "..." for name in arg_keys} or {"text": "..."}
    example_json = json.dumps({"tool": tool.name, "args": example_args}, ensure_ascii=False)

    prompt = f"""
    You are an AI assistant that can invoke external tools to enhance responses.

    Tool specification:
    - Name: {tool.name}
    - Description: {tool.description}
    - Tags: {tags}
    - Arguments: {args_summary}

    When you decide to use this tool, you MUST respond with ONLY a single JSON object in the form:
    {example_json}

    Rules:
    1. The first character of your reply must be '{{'.
    2. Do NOT add commentary, apologies, planning notes, or markdown.
    3. Do NOT write analysis before the JSON. No "Assistant:" prefixes.
    4. If you cannot comply, return {{"tool": null, "args": {{}}}}.

    Any violation causes the run to fail, so comply exactly.
    """
    return textwrap.dedent(prompt).strip()


def build_initial_messages(theme: str, tool: Any) -> list[dict[str, str]]:
    """Assemble the conversation history for the first LLM turn."""

    system_prompt = build_system_prompt(tool)
    user_prompt = (
        f"Write exactly one short poem about {theme}, add an appropriate emoji at the beginning, "
        f"and display it using {tool.name}. Respond with the JSON tool call only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_tool_call(raw_response: str, expected_tool: str) -> str:
    """Extract the poem text from the assistant's JSON tool call."""

    start = raw_response.find("{")
    if start == -1:
        raise ValueError("No JSON found in LLM response")

    decoder = json.JSONDecoder()
    payload = raw_response[start:].lstrip()
    try:
        data, _ = decoder.raw_decode(payload)
    except json.JSONDecodeError as first_error:
        if payload.startswith("{{"):
            data, _ = decoder.raw_decode(payload[1:])
        else:
            raise ValueError(f"Invalid JSON tool call: {first_error}") from first_error
    if data.get("tool") != expected_tool:
        raise ValueError(f"Unexpected tool requested: {data.get('tool')}")

    text = data.get("args", {}).get("text", "").strip()
    if not text:
        raise ValueError("Tool call did not include text")
    return text


def normalize_explanation(text: str) -> str:
    """Remove protocol noise and enforce a compact 3-sentence summary."""

    cleaned = text.strip()
    for marker in ("SYSTEM:", "USER:", "ASSISTANT:"):
        idx = cleaned.find(marker)
        if idx != -1:
            cleaned = cleaned[:idx].strip()

    fragments: list[str] = []
    for chunk in cleaned.splitlines():
        stripped = chunk.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[0-9０-９]+[\.\)\-、． ]*", "", stripped).strip()
        stripped = re.sub(r"^[\-・]+", "", stripped).strip()
        if stripped:
            fragments.append(stripped)

    if fragments:
        cleaned = " ".join(fragments)

    sentences = [s.strip() for s in cleaned.replace("\n", "").split("。") if s.strip()]
    if sentences:
        cleaned = "。".join(sentences[:3])
        if cleaned and not cleaned.endswith("。"):
            cleaned += "。"

    if cleaned and not cleaned.startswith("この詩は"):
        cleaned = f"この詩は{cleaned}"

    return cleaned


def explain_poem(bundle: Any, poem_text: str) -> str:
    """Ask the LLM to provide a short Japanese explanation of the poem."""

    messages = [
        {"role": "system", "content": POEM_ANALYST_PROMPT},
        {"role": "user", "content": poem_text},
    ]
    response = chat_once(bundle, messages, temperature=0.0)
    return normalize_explanation(response)


def main() -> None:
    """LLM selects and executes a tool via MCP."""

    bundle = load_model("mlx-community/Qwen3-8B-4bit")
    tool_spec = fetch_primary_tool()

    theme = random.choice(THEMES)
    messages = build_initial_messages(theme, tool_spec)
    raw_response = chat_once(bundle, messages, temperature=0.0, max_tokens=160)

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
