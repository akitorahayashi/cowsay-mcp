from __future__ import annotations

from typing import Iterator

from demo import tool_flow
from demo.main import main


def test_demo_main_happy_path(monkeypatch, capsys):
    """Simulate the two-turn flow without needing the actual MLX model."""

    dummy_bundle = (object(), object())
    monkeypatch.setattr(tool_flow, "load_model", lambda model_id: dummy_bundle)

    responses: Iterator[str] = iter(
        [
            '{"tool": "csay", "args": {"text": "good evening human"}}',
            "了解しました。楽しい夜をお過ごしください！",
        ]
    )

    def fake_chat_once(bundle, messages, **kwargs):
        return next(responses)

    monkeypatch.setattr(tool_flow, "chat_once", fake_chat_once)
    monkeypatch.setattr(tool_flow, "run_cowsay", lambda text: "<cow>\n(text)")

    main()

    captured = capsys.readouterr().out
    assert '"tool": "csay"' in captured
    assert "Final answer:" in captured
    assert "楽しい夜" in captured
