from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from demo.main import parse_tool_call, normalize_explanation, fetch_primary_tool


class TestParseToolCall:
    """Test LLM response parsing functionality."""

    def test_parse_valid_tool_call(self):
        """Test parsing a well-formed tool call."""
        response = 'Here is your poem: {"tool": "cowsay-mcp", "args": {"text": "Hello world"}}'
        result = parse_tool_call(response, "cowsay-mcp")
        assert result == "Hello world"

    def test_parse_tool_call_with_extra_text(self):
        """Test parsing when JSON is embedded in other text."""
        response = 'I will use the tool: {"tool": "cowsay-mcp", "args": {"text": "Test poem"}} and that\'s it.'
        result = parse_tool_call(response, "cowsay-mcp")
        assert result == "Test poem"

    def test_parse_tool_call_multiline(self):
        """Test parsing multiline JSON."""
        response = '''Let me create a poem:
        {"tool": "cowsay-mcp",
         "args": {"text": "Multiline\\npoem"}}'''
        result = parse_tool_call(response, "cowsay-mcp")
        assert result == "Multiline\npoem"

    def test_parse_tool_call_wrong_tool(self):
        """Test error when wrong tool is requested."""
        response = '{"tool": "wrong-tool", "args": {"text": "test"}}'
        with pytest.raises(ValueError, match="Unexpected tool requested"):
            parse_tool_call(response, "cowsay-mcp")

    def test_parse_tool_call_no_json(self):
        """Test error when no JSON found."""
        response = "Just some text without JSON"
        with pytest.raises(ValueError, match="No JSON found"):
            parse_tool_call(response, "cowsay-mcp")

    def test_parse_tool_call_missing_text(self):
        """Test error when text argument is missing."""
        response = '{"tool": "cowsay-mcp", "args": {}}'
        with pytest.raises(ValueError, match="Tool call did not include text"):
            parse_tool_call(response, "cowsay-mcp")

    def test_parse_tool_call_empty_text(self):
        """Test error when text is empty."""
        response = '{"tool": "cowsay-mcp", "args": {"text": ""}}'
        with pytest.raises(ValueError, match="Tool call did not include text"):
            parse_tool_call(response, "cowsay-mcp")

    def test_parse_tool_call_whitespace_text(self):
        """Test error when text is only whitespace."""
        response = '{"tool": "cowsay-mcp", "args": {"text": "   "}}'
        with pytest.raises(ValueError, match="Tool call did not include text"):
            parse_tool_call(response, "cowsay-mcp")

    def test_parse_tool_call_malformed_json(self):
        """Test error with malformed JSON."""
        response = '{"tool": "cowsay-mcp", "args": {"text": "test"'
        with pytest.raises(ValueError):  # JSONDecoder will raise
            parse_tool_call(response, "cowsay-mcp")


class TestNormalizeExplanation:
    """Test text normalization functionality."""

    def test_normalize_basic_text(self):
        """Test basic text normalization."""
        text = "This is a normal explanation."
        assert normalize_explanation(text) == "This is a normal explanation."

    def test_normalize_with_system_marker(self):
        """Test removing SYSTEM marker."""
        text = "SYSTEM: This is the explanation"
        assert normalize_explanation(text) == "This is the explanation"

    def test_normalize_with_user_marker(self):
        """Test removing USER marker."""
        text = "USER: Please explain this"
        assert normalize_explanation(text) == "Please explain this"

    def test_normalize_with_assistant_marker(self):
        """Test removing ASSISTANT marker."""
        text = "ASSISTANT: Here's my analysis"
        assert normalize_explanation(text) == "Here's my analysis"

    def test_normalize_with_marker_in_middle(self):
        """Test marker removal when not at start - removes from marker onward."""
        text = "Some text SYSTEM: and more"
        assert normalize_explanation(text) == "and more"

    def test_normalize_whitespace_only(self):
        """Test handling of whitespace-only text."""
        text = "   \n\t   "
        assert normalize_explanation(text) == ""

    def test_normalize_multiple_markers(self):
        """Test handling multiple markers (removes first one)."""
        text = "SYSTEM: USER: ASSISTANT: Final text"
        assert normalize_explanation(text) == "USER: ASSISTANT: Final text"


class TestFetchPrimaryTool:
    """Test server tool fetching functionality."""

    def test_fetch_primary_tool_success(self, monkeypatch):
        """Test successful tool fetching."""
        from cowsay_mcp.server import server

        # Mock the server.get_tools() to return a tool
        mock_tool = MagicMock()
        mock_tool.name = "cowsay-mcp"

        mock_server = MagicMock()
        mock_server.get_tools = AsyncMock(return_value={"cowsay-mcp": mock_tool})

        monkeypatch.setattr("demo.main.server", mock_server)

        result = fetch_primary_tool()
        assert result == mock_tool

    def test_fetch_primary_tool_empty(self, monkeypatch):
        """Test error when no tools available."""
        from cowsay_mcp.server import server

        mock_server = MagicMock()
        mock_server.get_tools = AsyncMock(return_value={})

        monkeypatch.setattr("demo.main.server", mock_server)

        with pytest.raises(RuntimeError, match="No tools registered"):
            fetch_primary_tool()