from __future__ import annotations

"""CLI entrypoint for running the cowsay tool-calling demo."""

from .tool_flow import run_demo


def main() -> None:
    tool_call_json, final_answer = run_demo()

    print("Tool call JSON:")
    print(tool_call_json)
    print("\nFinal answer:")
    print(final_answer)


if __name__ == "__main__":
    main()
