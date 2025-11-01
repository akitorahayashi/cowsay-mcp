from __future__ import annotations

from cowsay_mcp.server import run_cowsay


def test_run_cowsay_success(monkeypatch):
    monkeypatch.setattr(
        "cowsay_mcp.server.cowsay.get_output_string",
        lambda cow, text: f"{cow}:{text}",
    )

    assert run_cowsay("hello") == "cow:hello"


def test_run_cowsay_handles_exception(monkeypatch):
    def raise_error(cow, text):  # noqa: D401 - helper raising error
        raise ValueError("boom")

    monkeypatch.setattr("cowsay_mcp.server.cowsay.get_output_string", raise_error)

    assert run_cowsay("hello").startswith("cowsay error:")
