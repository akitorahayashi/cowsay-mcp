from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from cowsay_mcp.main import main as server_main


class TestServerMain:
    """Test the MCP server main function with stdin handling."""

    @patch("sys.stdin")
    @patch("cowsay_mcp.main.server.run")
    def test_server_main_normal_mode(self, mock_server_run, mock_stdin):
        """Test server runs normally when stdin is a tty (interactive)."""
        mock_stdin.isatty.return_value = True

        server_main()

        mock_server_run.assert_called_once()
        mock_stdin.read.assert_not_called()

    @patch("sys.stdin")
    @patch("cowsay_mcp.main.server.run")
    def test_server_main_stdin_mode_valid_tool_call(self, mock_server_run, mock_stdin):
        """Test server handles valid tool call from stdin."""
        mock_stdin.isatty.return_value = False
        tool_call = {"tool": "cowsay-mcp", "args": {"text": "Hello world"}}
        mock_stdin.read.return_value = json.dumps(tool_call)

        with patch(
            "cowsay_mcp.server.run_cowsay", return_value="Mocked ASCII art"
        ) as mock_run_cowsay:
            with patch("builtins.print") as mock_print:
                server_main()

        mock_server_run.assert_not_called()
        mock_run_cowsay.assert_called_once_with("Hello world")
        # Check that JSON response was printed
        printed_calls = [call for call in mock_print.call_args_list if len(call[0]) > 0]
        assert len(printed_calls) == 1
        response = json.loads(printed_calls[0][0][0])
        assert response == {"result": "Mocked ASCII art"}

    @patch("sys.stdin")
    @patch("cowsay_mcp.main.server.run")
    def test_server_main_stdin_mode_unknown_tool(self, mock_server_run, mock_stdin):
        """Test server handles unknown tool from stdin."""
        mock_stdin.isatty.return_value = False
        tool_call = {"tool": "unknown-tool", "args": {"text": "test"}}
        mock_stdin.read.return_value = json.dumps(tool_call)

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit):
                server_main()

        mock_server_run.assert_not_called()
        # Check that error JSON was printed to stderr
        stderr_calls = [
            call
            for call in mock_print.call_args_list
            if len(call) > 1 and call[1].get("file") is not None
        ]
        assert len(stderr_calls) >= 1
        # The last stderr call should contain the error
        error_output = stderr_calls[-1][0][0]
        error_data = json.loads(error_output)
        assert "Unknown tool" in error_data["error"]

    @patch("sys.stdin")
    @patch("cowsay_mcp.main.server.run")
    def test_server_main_stdin_mode_invalid_json(self, mock_server_run, mock_stdin):
        """Test server handles invalid JSON from stdin."""
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = "invalid json"

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit):
                server_main()

        mock_server_run.assert_not_called()
        # Check that error JSON was printed to stderr
        stderr_calls = [
            call
            for call in mock_print.call_args_list
            if len(call) > 1 and call[1].get("file") is not None
        ]
        assert len(stderr_calls) >= 1
        error_output = stderr_calls[-1][0][0]
        error_data = json.loads(error_output)
        assert "error" in error_data

    @patch("sys.stdin")
    @patch("cowsay_mcp.main.server.run")
    def test_server_main_stdin_mode_empty_input(self, mock_server_run, mock_stdin):
        """Test server handles empty input from stdin."""
        mock_stdin.isatty.return_value = False
        mock_stdin.read.return_value = ""

        with patch("builtins.print") as mock_print:
            with pytest.raises(SystemExit):
                server_main()

        mock_server_run.assert_not_called()
        # Check that error JSON was printed to stderr
        stderr_calls = [
            call
            for call in mock_print.call_args_list
            if len(call) > 1 and call[1].get("file") is not None
        ]
        assert len(stderr_calls) >= 1
        error_output = stderr_calls[-1][0][0]
        error_data = json.loads(error_output)
        assert "No input data" in error_data["error"]

    @patch("sys.stdin")
    @patch("cowsay_mcp.main.server.run")
    def test_server_main_stdin_mode_missing_args(self, mock_server_run, mock_stdin):
        """Test server handles tool call with missing args."""
        mock_stdin.isatty.return_value = False
        tool_call = {"tool": "cowsay-mcp"}  # No args
        mock_stdin.read.return_value = json.dumps(tool_call)

        with patch(
            "cowsay_mcp.server.run_cowsay", return_value="ASCII art"
        ) as mock_run_cowsay:
            with patch("builtins.print"):
                server_main()

        mock_run_cowsay.assert_called_once_with(
            ""
        )  # Should pass empty string for missing text

    @patch("sys.stdin")
    @patch("cowsay_mcp.main.server.run")
    def test_server_main_stdin_mode_tool_execution_error(
        self, mock_server_run, mock_stdin
    ):
        """Test server handles tool execution errors."""
        mock_stdin.isatty.return_value = False
        tool_call = {"tool": "cowsay-mcp", "args": {"text": "test"}}
        mock_stdin.read.return_value = json.dumps(tool_call)

        with patch(
            "cowsay_mcp.server.run_cowsay", side_effect=Exception("Tool failed")
        ) as mock_run_cowsay:
            with patch("builtins.print"):
                with pytest.raises(SystemExit):
                    server_main()

        mock_run_cowsay.assert_called_once_with("test")
