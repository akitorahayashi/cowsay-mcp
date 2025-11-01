from __future__ import annotations

from csay.server import csay_action, run_cowsay


def test_run_cowsay_success(monkeypatch):
    monkeypatch.setattr(
        "csay.server.cowsay.get_output_string",
        lambda cow, text: f"{cow}:{text}",
    )

    assert run_cowsay("hello") == "cow:hello"


def test_run_cowsay_handles_exception(monkeypatch):
    def raise_error(cow, text):  # noqa: D401 - helper raising error
        raise ValueError("boom")

    monkeypatch.setattr("csay.server.cowsay.get_output_string", raise_error)

    assert run_cowsay("hello").startswith("cowsay error:")


def test_csay_action_delegates(monkeypatch):
    called_with: list[str] = []

    def fake_run_cowsay(text: str) -> str:
        called_with.append(text)
        return "delegate"

    monkeypatch.setattr("csay.server.run_cowsay", fake_run_cowsay)

    assert csay_action("greetings") == "delegate"
    assert called_with == ["greetings"]
