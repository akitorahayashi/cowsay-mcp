from __future__ import annotations

"""High-level orchestration for the cowsay tool-calling demo."""

import json
import sys
from typing import Literal, Tuple

from csay import run_cowsay

from .llm import Message, ModelBundle, chat_once, load_model
from .prompts import final_turn_messages, first_turn_messages

ToolName = Literal["csay"]


def request_tool_call(bundle: ModelBundle) -> str:
    """Ask the model to produce a JSON tool invocation request."""
    messages: Tuple[Message, Message] = first_turn_messages()
    return chat_once(bundle, messages)


def obtain_final_answer(
    bundle: ModelBundle, tool_call_json: str, tool_result: str
) -> str:
    """Send tool feedback to the model and capture the natural-language reply."""
    messages = final_turn_messages(tool_call_json, tool_result)
    return chat_once(bundle, messages, max_tokens=256, temperature=0.0)


def validate_tool_call(raw_json: str) -> tuple[ToolName, str]:
    """Parse and validate the tool call JSON."""
    decoder = json.JSONDecoder()
    trimmed = raw_json.lstrip()
    offset = len(raw_json) - len(trimmed)

    try:
        candidate, end = decoder.raw_decode(trimmed)
    except (
        json.JSONDecodeError
    ) as exc:  # pragma: no cover - requires malformed model output
        print(
            f"Invalid JSON from model: {exc}\nRaw output: {raw_json}", file=sys.stderr
        )
        sys.exit(1)

    remainder = raw_json[offset + end :].strip()
    if remainder:
        print("Warning: ignoring trailing text after JSON tool call.", file=sys.stderr)

    if not isinstance(candidate, dict):
        print(f"Expected a JSON object, received: {candidate!r}", file=sys.stderr)
        sys.exit(1)

    tool = candidate.get("tool")
    if tool != "csay":
        print(f"Unsupported tool requested: {tool}", file=sys.stderr)
        sys.exit(1)

    args = candidate.get("args")
    if not isinstance(args, dict):
        print(f"Expected args to be a JSON object, received: {args!r}", file=sys.stderr)
        sys.exit(1)

    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        print(
            f"Expected args['text'] to be a non-empty string, received: {text!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    return "csay", text


def run_demo(model_id: str = "mlx-community/Qwen3-8B-4bit") -> tuple[str, str]:
    """Execute the full demo round trip and return the intermediate/final results."""
    bundle = load_model(model_id)

    raw_tool_call = request_tool_call(bundle)
    _, text_argument = validate_tool_call(raw_tool_call)

    tool_output = run_cowsay(text_argument)
    final_answer = obtain_final_answer(bundle, raw_tool_call, tool_output)

    return raw_tool_call, final_answer


__all__ = [
    "ToolName",
    "obtain_final_answer",
    "request_tool_call",
    "run_demo",
    "validate_tool_call",
]
