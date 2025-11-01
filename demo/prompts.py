from __future__ import annotations

"""Prompt construction helpers for the demo conversation flow."""

from typing import Tuple

from .llm import Message


def build_system_prompt() -> str:
    """Describe the available tool and strict JSON schema instructions."""
    return (
        "You are an assistant that can call exactly one tool when needed.\n"
        'Available tool: csay with args schema {"text": "<string>"}.\n'
        'When you want to use a tool, respond with ONLY a JSON object with keys "tool" and "args".\n'
        "Do not add commentary or markdown, just valid JSON.\n"
        'Example: {"tool": "csay", "args": {"text": "hello"}}'
    )


def first_turn_messages() -> Tuple[Message, Message]:
    """Return the system/user messages for the first model call."""
    return (
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": "Please say 'good evening human' using cowsay."},
    )


def final_turn_messages(
    tool_call_json: str, tool_result: str
) -> Tuple[Message, Message, Message]:
    """Compose the second-round messages including tool feedback."""
    return (
        {"role": "system", "content": build_system_prompt()},
        {"role": "assistant", "content": tool_call_json},
        {
            "role": "user",
            "content": (
                "tool_result:\n"
                f"{tool_result}\n\n"
                "Now produce a helpful final answer for the user in natural language (in Japanese)."
            ),
        },
    )


__all__ = [
    "build_system_prompt",
    "first_turn_messages",
    "final_turn_messages",
]
