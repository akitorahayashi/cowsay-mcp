import json
import sys

from cowsay_mcp.server import server


def main() -> None:
    """Start the FastMCP server or handle stdin tool call."""
    if not sys.stdin.isatty():
        # Input from pipe, handle as tool call
        try:
            input_data = sys.stdin.read().strip()
            if not input_data:
                raise ValueError("No input data")

            tool_call = json.loads(input_data)
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})

            if tool_name == "cowsay-mcp":
                from cowsay_mcp.server import run_cowsay
                result = run_cowsay(args.get("text", ""))
                response = {"result": result}
                print(json.dumps(response))
            else:
                response = {"error": f"Unknown tool: {tool_name}"}
                print(json.dumps(response), file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            response = {"error": str(e)}
            print(json.dumps(response), file=sys.stderr)
            sys.exit(1)
    else:
        # No pipe input, run as MCP server
        server.run()


if __name__ == "__main__":
    main()
