from __future__ import annotations

"""This MCP server exposes a single tool `csay` backed by the Python `cowsay` package so that local LLMs can request ASCII-art speech bubbles."""

from typing import Final

import cowsay
from fastmcp import FastMCP

SERVER_NAME: Final[str] = "csay"
server = FastMCP(SERVER_NAME)


def run_cowsay(text: str) -> str:
    """Return the ASCII-art cow for the provided text using the cowsay library."""
    try:
        return cowsay.get_output_string("cow", text)
    except Exception as exc:  # pragma: no cover - defensive catch for library errors
        return f"cowsay error: {exc}"


def csay_action(text: str) -> str:
    """Expose the cowsay helper as a plain callable for reuse and testing."""
    return run_cowsay(text)


server.tool(name="csay")(csay_action)


def main() -> None:
    """Start the FastMCP server."""
    server.run()


if __name__ == "__main__":
    main()
