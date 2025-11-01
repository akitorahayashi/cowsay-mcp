from __future__ import annotations

from typing import Final

import cowsay
from fastmcp import FastMCP

"""This MCP server exposes a single tool `cowsay-mcp` backed by the Python `cowsay` package so that local LLMs can request ASCII-art speech bubbles."""


SERVER_NAME: Final[str] = "cowsay-mcp"


server = FastMCP(SERVER_NAME)


def run_cowsay(text: str) -> str:
    """Generate ASCII art speech bubble with a cow using the provided text.
    
    This tool creates fun ASCII art where a cow appears to be speaking
    the given text in a speech bubble. Perfect for adding humor and
    personality to messages.
    
    Args:
        text: The message to display in the cow's speech bubble
        
    Returns:
        ASCII art string containing the speech bubble and cow
    """
    try:
        return cowsay.get_output_string("cow", text)
    except Exception as exc:  # pragma: no cover - defensive catch for library errors
        return f"cowsay error: {exc}"


server.tool(
    name="cowsay-mcp",
    description="Generate fun ASCII art speech bubbles with a cow. Use this tool when you want to make messages more engaging and humorous by displaying them as if a cow is speaking.",
    tags={"text", "art", "fun", "ascii"}
)(run_cowsay)


def main() -> None:
    """Start the FastMCP server."""
    server.run()


if __name__ == "__main__":
    main()
