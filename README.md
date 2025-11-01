# csay

## Overview
- Demonstrates how a local MLX model can perform tool-calling by coordinating with a FastMCP server that wraps the Python `cowsay` library.
- Provides a single `csay` tool implemented as an MCP server plus a demo script that walks through the two-turn tool-calling loop.

## Requirements
- Python 3.12 or newer.
- [uv](https://github.com/astral-sh/uv) installed and on `PATH` for dependency management.
- Apple Silicon machine with [MLX](https://github.com/ml-explore/mlx) support; the demo loads `mlx-community/Qwen3-8B-4bit` via `mlx-lm`.

## Setup
- `uv sync` to install the core project dependencies.
- `uv sync --group dev` if you want local linting/formatting helpers.
- `uv sync --group demo` before running the MLX demo so that `mlx-lm` is available.
- Optional: `uv run csay-server` (or `python -m csay.server`) to launch the FastMCP server manually for inspection.

## Demo Run
- Execute `uv run --group demo python -m demo.main`.
- The script will:
	- prompt the model to request the `csay` tool via strict JSON,
	- render the ASCII-art response via the bundled Python `cowsay` dependency,
	- send the tool call and output back to the model, requesting a final Japanese response.
- Expect to see the raw JSON tool call printed first, followed by the model's final answer.

## Testing
- `uv run pytest tests/unit` to execute fast unit tests.
- `uv run pytest tests/intg` to run integration coverage (patched MLX + cowsay flow).
- `just test` runs both suites if you rely on the justfile helper.

## Notes / Future Work
- The demo imports the tool helper directly; a production agent would speak MCP over stdio or sockets to the running FastMCP server.
- Additional tools (filesystem, HTTP, etc.) can be added to `csay.server` without changing the demo structure.
- You can replace `mlx-community/Qwen3-8B-4bit` with other MLX-compatible models or quantization levels to experiment with different behaviors.