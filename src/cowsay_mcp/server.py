from __future__ import annotations

from typing import Final

import cowsay
from fastmcp import FastMCP

"""This MCP server exposes a single tool `cowsay-mcp` backed by the Python `cowsay` package so that local LLMs can request ASCII-art speech bubbles."""


SERVER_NAME: Final[str] = "cowsay-mcp"


server = FastMCP(SERVER_NAME)


def run_cowsay(text: str) -> str:
    """Return the ASCII-art cow for the provided text using the cowsay library."""
    try:
        return cowsay.get_output_string("cow", text)
    except Exception as exc:  # pragma: no cover - defensive catch for library errors
        return f"cowsay error: {exc}"


server.tool(name="cowsay-mcp")(run_cowsay)


def main() -> None:
    """Start the FastMCP server."""
    server.run()


if __name__ == "__main__":
    main()
