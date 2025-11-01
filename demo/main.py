from __future__ import annotations

import json
import sys

from cowsay_mcp import run_cowsay

from .llm import chat_once, load_model

"""CLI entrypoint for running the cowsay tool-calling demo."""


def main() -> None:
    """LLM selects and executes a tool via MCP."""
    bundle = load_model("mlx-community/Qwen3-8B-4bit")

    # Ask LLM to select a tool
    messages = (
        {
            "role": "system",
            "content": (
                "You are an AI assistant that can use tools.\n"
                "Available tool: cowsay-mcp - displays text in ASCII art\n"
                "Tool format: {'tool': 'cowsay-mcp', 'args': {'text': 'your message'}}\n\n"
                "IMPORTANT: Respond ONLY with a JSON object in the exact format above.\n"
                "Do not write explanations or any other text. Just JSON."
            ),
        },
        {
            "role": "user",
            "content": "Write a short poem, add an appropriate emoji at the beginning, and display it using cowsay-mcp.",
        },
    )
    raw_response = chat_once(bundle, messages)

    # Parse and validate tool call
    try:
        # Find the first complete JSON object (LLM might output text before JSON)
        start = raw_response.find("{")
        if start == -1:
            raise ValueError("No JSON found")

        brace_count = 0
        end = start
        for i, char in enumerate(raw_response[start:], start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        json_str = raw_response[start:end]
        data = json.loads(json_str)

        if data.get("tool") != "cowsay-mcp":
            sys.exit(f"Wrong tool: {data.get('tool')}")

        text = data.get("args", {}).get("text", "").strip()
        if not text:
            sys.exit("No text provided")

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"LLM output: {raw_response!r}", file=sys.stderr)
        sys.exit(f"Invalid tool call: {e}")

    # Execute tool
    tool_call_json = json.dumps({"tool": "cowsay-mcp", "args": {"text": text}})
    result = run_cowsay(text)

    print("LLM selected tool:", tool_call_json)
    print("\nTool executed result:")
    print(result)


if __name__ == "__main__":
    main()
