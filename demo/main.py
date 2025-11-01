from __future__ import annotations

import json
import random
import sys

from cowsay_mcp import run_cowsay

from .llm import chat_once, load_model

"""CLI entrypoint for running the cowsay tool-calling demo."""


def main() -> None:
    """LLM selects and executes a tool via MCP."""
    bundle = load_model("mlx-community/Qwen3-8B-4bit")

    # Ask LLM to select a tool
    themes = ["nature", "technology", "emotions", "adventure", "creativity"]
    random_theme = random.choice(themes)

    messages = (
        {
            "role": "system",
            "content": (
                "You are an AI assistant that can use tools to enhance your responses.\n"
                "Available tool: cowsay-mcp - Generate fun ASCII art speech bubbles with a cow\n"
                "Tool format: {'tool': 'cowsay-mcp', 'args': {'text': 'your message'}}\n\n"
                "Use this tool when:\n"
                "- You want to make a message more fun and engaging\n"
                "- The user asks for ASCII art or visual elements\n"
                "- You want to add humor or personality to your response\n"
                "- The message would benefit from visual presentation\n\n"
                "IMPORTANT: Respond ONLY with a JSON object in the exact format above.\n"
                "Do not write explanations or any other text. Just JSON."
            ),
        },
        {
            "role": "user",
            "content": f"Write exactly one short poem about {random_theme}, add an appropriate emoji at the beginning, and display it using cowsay-mcp.",
        },
    )
    raw_response = chat_once(bundle, messages, temperature=1.0)

    print(f"LLM raw response: {raw_response!r}", file=sys.stderr)

    # Parse and validate tool call
    try:
        # Find the start of the JSON object
        start = raw_response.find("{")
        if start == -1:
            raise ValueError("No JSON found")

        # Use the built-in JSON decoder to robustly parse the object from the string.
        # This correctly handles nested structures and special characters.
        data, _ = json.JSONDecoder().raw_decode(raw_response[start:])

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
