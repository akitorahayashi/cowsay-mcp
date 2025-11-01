from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from demo import main as demo_main


def test_demo_main_happy_path(monkeypatch, capsys):
    """Test the complete demo flow with real MCP communication."""
    dummy_bundle = (object(), object())

    def fake_load_model(model_id):
        return dummy_bundle

    def fake_chat_once(bundle, messages, **kwargs):
        # Return a valid tool call that will trigger the MCP subprocess
        return '{"tool": "cowsay-mcp", "args": {"text": "Hello world"}}'

    # Mock the LLM functions
    monkeypatch.setattr("demo.main.load_model", fake_load_model)
    monkeypatch.setattr("demo.main.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.llm.chat_once", fake_chat_once)

    # Mock random.choice to return a consistent theme
    monkeypatch.setattr("demo.main.random.choice", lambda x: "nature")

    demo_main.main()

    captured = capsys.readouterr()
    stdout_output = captured.out
    stderr_output = captured.err

    # Check that the tool call was logged to stderr
    assert '"tool": "cowsay-mcp"' in stderr_output
    assert '"args": {"text": "Hello world"}' in stderr_output

    # Check that the ASCII art result appears in stdout
    assert "Hello world" in stdout_output  # The cowsay output should contain the input text
    assert "|" in stdout_output  # cowsay uses | characters around the text
    assert "Tool executed result:" in stdout_output

    # Check that the poem explanation was generated
    assert "Poem explanation:" in stdout_output


def test_demo_main_invalid_tool_call(monkeypatch, capsys):
    """Test demo handles invalid tool calls gracefully."""
    dummy_bundle = (object(), object())

    def fake_load_model(model_id):
        return dummy_bundle

    def fake_chat_once(bundle, messages, **kwargs):
        # Return invalid JSON
        return 'invalid json tool call'

    monkeypatch.setattr("demo.main.load_model", fake_load_model)
    monkeypatch.setattr("demo.main.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.llm.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.main.random.choice", lambda x: "nature")

    # Should exit with error
    with pytest.raises(SystemExit) as exc_info:
        demo_main.main()

    assert "Invalid tool call" in str(exc_info.value)


def test_demo_main_wrong_tool_name(monkeypatch, capsys):
    """Test demo handles calls to non-existent tools."""
    dummy_bundle = (object(), object())

    def fake_load_model(model_id):
        return dummy_bundle

    def fake_chat_once(bundle, messages, **kwargs):
        # Return call to wrong tool
        return '{"tool": "nonexistent-tool", "args": {"text": "test"}}'

    monkeypatch.setattr("demo.main.load_model", fake_load_model)
    monkeypatch.setattr("demo.main.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.llm.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.main.random.choice", lambda x: "nature")

    # Should exit with error
    try:
        demo_main.main()
        assert False, "Should have exited with error"
    except SystemExit as e:
        assert "Unexpected tool requested" in str(e)


def test_demo_main_mcp_communication_failure(monkeypatch, capsys):
    """Test demo handles MCP server failures."""
    dummy_bundle = (object(), object())

    def fake_load_model(model_id):
        return dummy_bundle

    def fake_chat_once(bundle, messages, **kwargs):
        return '{"tool": "cowsay-mcp", "args": {"text": "test"}}'

    # Mock subprocess to simulate server failure
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("", "MCP server error")
    mock_proc.stderr = "error"

    monkeypatch.setattr("demo.main.load_model", fake_load_model)
    monkeypatch.setattr("demo.main.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.llm.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.main.random.choice", lambda x: "nature")
    monkeypatch.setattr("demo.main.subprocess.Popen", lambda *args, **kwargs: mock_proc)

    with pytest.raises(SystemExit) as exc_info:
        demo_main.main()

    assert "MCP communication error" in str(exc_info.value)

    captured = capsys.readouterr()
    assert "MCP Server Error" in captured.err


def test_demo_main_malformed_mcp_response(monkeypatch, capsys):
    """Test demo handles malformed MCP responses."""
    dummy_bundle = (object(), object())

    def fake_load_model(model_id):
        return dummy_bundle

    def fake_chat_once(bundle, messages, **kwargs):
        return '{"tool": "cowsay-mcp", "args": {"text": "test"}}'

    # Mock subprocess to return invalid JSON
    mock_proc = MagicMock()
    mock_proc.communicate.return_value = ("invalid json response", "")

    monkeypatch.setattr("demo.main.load_model", fake_load_model)
    monkeypatch.setattr("demo.main.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.llm.chat_once", fake_chat_once)
    monkeypatch.setattr("demo.main.random.choice", lambda x: "nature")
    monkeypatch.setattr("demo.main.subprocess.Popen", lambda *args, **kwargs: mock_proc)

    with pytest.raises(SystemExit) as exc_info:
        demo_main.main()

    assert "MCP communication error" in str(exc_info.value)
