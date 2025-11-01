from __future__ import annotations

import json
import textwrap
from typing import Any

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
    example_args = {name: "..." for name in arg_keys}
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


__all__ = [
    "THEMES",
    "build_initial_messages",
]
