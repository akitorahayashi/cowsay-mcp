from __future__ import annotations

import json
from unittest.mock import MagicMock

from demo.prompting import (
    summarise_tags,
    summarise_parameters,
    build_system_prompt,
    build_initial_messages,
    THEMES,
)


class TestSummariseTags:
    """Test tag summarization functionality."""

    def test_summarise_tags_with_tags(self):
        """Test summarization when tool has tags."""
        mock_tool = MagicMock()
        mock_tool.tags = ["text", "art", "fun"]
        result = summarise_tags(mock_tool)
        assert result == "art, fun, text"  # Should be sorted

    def test_summarise_tags_empty(self):
        """Test summarization when tool has no tags."""
        mock_tool = MagicMock()
        mock_tool.tags = None
        result = summarise_tags(mock_tool)
        assert result == "none"

    def test_summarise_tags_empty_list(self):
        """Test summarization when tool has empty tags list."""
        mock_tool = MagicMock()
        mock_tool.tags = []
        result = summarise_tags(mock_tool)
        assert result == "none"


class TestSummariseParameters:
    """Test parameter summarization functionality."""

    def test_summarise_parameters_with_properties(self):
        """Test summarization with parameter properties."""
        schema = {
            "properties": {
                "text": {"type": "string", "description": "The text to process"},
                "count": {"type": "integer", "description": "Number of items"}
            }
        }
        result = summarise_parameters(schema)
        expected = "text (string) - The text to process; count (integer) - Number of items"
        assert result == expected

    def test_summarise_parameters_no_properties(self):
        """Test summarization with no properties."""
        schema = {"properties": {}}
        result = summarise_parameters(schema)
        assert result == "No arguments required"

    def test_summarise_parameters_none_schema(self):
        """Test summarization with None schema."""
        result = summarise_parameters(None)
        assert result == "No arguments required"

    def test_summarise_parameters_no_description(self):
        """Test summarization when parameters lack descriptions."""
        schema = {
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"}
            }
        }
        result = summarise_parameters(schema)
        expected = "text (string); count (integer)"
        assert result == expected


class TestBuildSystemPrompt:
    """Test system prompt building functionality."""

    def test_build_system_prompt_complete(self):
        """Test building complete system prompt."""
        mock_tool = MagicMock()
        mock_tool.name = "cowsay-mcp"
        mock_tool.description = "Generate ASCII art speech bubbles"
        mock_tool.tags = ["text", "art"]
        mock_tool.parameters = {
            "properties": {
                "text": {"type": "string", "description": "The message"}
            }
        }

        result = build_system_prompt(mock_tool)

        # Check that key components are included
        assert "cowsay-mcp" in result
        assert "Generate ASCII art speech bubbles" in result
        assert "art, text" in result  # sorted tags
        assert "text (string) - The message" in result
        assert '"tool": "cowsay-mcp"' in result
        assert '"args": {"text": "..."' in result

    def test_build_system_prompt_minimal(self):
        """Test building system prompt with minimal tool info."""
        mock_tool = MagicMock()
        mock_tool.name = "test-tool"
        mock_tool.description = "A test tool"
        mock_tool.tags = None
        mock_tool.parameters = None

        result = build_system_prompt(mock_tool)

        assert "test-tool" in result
        assert "A test tool" in result
        assert "none" in result  # no tags
        assert "No arguments required" in result


class TestBuildInitialMessages:
    """Test initial message building functionality."""

    def test_build_initial_messages(self):
        """Test building initial conversation messages."""
        mock_tool = MagicMock()
        mock_tool.name = "cowsay-mcp"

        theme = "nature"
        messages = build_initial_messages(theme, mock_tool)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert theme in messages[1]["content"]
        assert mock_tool.name in messages[1]["content"]

    def test_build_initial_messages_different_themes(self):
        """Test that different themes are included in user prompt."""
        mock_tool = MagicMock()
        mock_tool.name = "test-tool"

        for theme in ["nature", "technology", "emotions"]:
            messages = build_initial_messages(theme, mock_tool)
            assert theme in messages[1]["content"]


class TestConstants:
    """Test constant values."""

    def test_themes_constant(self):
        """Test that THEMES contains expected values."""
        expected_themes = ["nature", "technology", "emotions", "adventure", "creativity"]
        assert THEMES == expected_themes