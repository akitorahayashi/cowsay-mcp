from __future__ import annotations

import pytest

from demo.tool_flow import validate_tool_call


def test_validate_tool_call_success():
    tool, text = validate_tool_call('{"tool": "csay", "args": {"text": " hello "}}')
    assert tool == "csay"
    assert text == " hello "


def test_validate_tool_call_with_trailing_text(capsys):
    payload = '{"tool": "csay", "args": {"text": "hi"}} extra notes'
    tool, text = validate_tool_call(payload)
    assert tool == "csay"
    assert text == "hi"
    warning = capsys.readouterr().err
    assert "trailing text" in warning


def test_validate_tool_call_invalid_tool():
    with pytest.raises(SystemExit):
        validate_tool_call('{"tool": "other", "args": {"text": "hi"}}')


def test_validate_tool_call_missing_text():
    with pytest.raises(SystemExit):
        validate_tool_call('{"tool": "csay", "args": {"text": "   "}}')
