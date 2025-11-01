from __future__ import annotations

import json
import textwrap
from typing import Any

from .llm import chat_once

"""Helper utilities for prompts, parsing, and explanations in the demo."""

THEMES = ["nature", "technology", "emotions", "adventure", "creativity"]

POEM_ANALYST_PROMPT = textwrap.dedent(
    """
    あなたは詩の解説者です。ユーザが渡す詩を題材に、テーマ・イメージ・感情的な余韻を3文以内の自然な日本語の文章でまとめてください。
    完成した文章だけを返してください。
    ~です、~ますなどの丁寧語を使って解説してください。
    """
).strip()


def summarise_tags(tool: Any) -> str:
    """Return a descriptive list of tags for display."""

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
    example_json = json.dumps(
        {"tool": tool.name, "args": example_args}, ensure_ascii=False
    )

    prompt = f"""
    You are an AI assistant that can use tools to enhance your responses.

    Available tool: {tool.name} - {tool.description}
    Tool format: {example_json}
    Tool tags: {tags}
    Arguments: {args_summary}

    Use this tool when it will make the user's message more playful or expressive.

    IMPORTANT: Respond ONLY with a JSON object in the exact format above.
    Do not write explanations or any other text. Just JSON.
    """
    return textwrap.dedent(prompt).strip()


def build_initial_messages(theme: str, tool: Any) -> list[dict[str, str]]:
    """Assemble the conversation history for the first LLM turn."""

    system_prompt = build_system_prompt(tool)
    user_prompt = (
        f"Write exactly one short poem about {theme}, add an appropriate emoji at the beginning, "
        f"and display it using {tool.name}."
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


def explain_poem(bundle: Any, poem_text: str) -> str:
    """Ask the LLM to provide a short Japanese explanation of the poem."""

    messages = [
        {"role": "system", "content": POEM_ANALYST_PROMPT},
        {"role": "user", "content": poem_text},
    ]
    response = chat_once(bundle, messages, temperature=0.0)
    return normalize_explanation(response)


__all__ = [
    "THEMES",
    "build_initial_messages",
    "parse_tool_call",
    "explain_poem",
]
