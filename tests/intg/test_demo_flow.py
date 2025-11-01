from __future__ import annotations

from demo import main as demo_main


def test_demo_main_happy_path(monkeypatch, capsys):
    """Test the complete demo flow."""
    dummy_bundle = (object(), object())

    def fake_load_model(model_id):
        return dummy_bundle

    def fake_chat_once(bundle, messages, **kwargs):
        return '{"tool": "cowsay-mcp", "args": {"text": "🌸 In gardens of the mind, dreams bloom\\nPetals of thought in morning dew"}}'

    monkeypatch.setattr("demo.main.load_model", fake_load_model)
    monkeypatch.setattr("demo.main.chat_once", fake_chat_once)

    demo_main.main()

    captured = capsys.readouterr().out
    assert '"tool": "cowsay-mcp"' in captured
    assert "🌸" in captured
