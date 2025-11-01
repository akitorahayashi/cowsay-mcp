from __future__ import annotations

from typing import Literal, Protocol, Sequence, Tuple, TypedDict

"""Helpers for interacting with MLX local models framed as chat assistants."""

ModelBundle = Tuple[object, object]


class LoadFn(Protocol):
    def __call__(self, model_id: str) -> ModelBundle:
        """Return the loaded model/tokenizer pair."""


class GenerateFn(Protocol):
    def __call__(
        self, model: object, tokenizer: object, prompt: str, /, **kwargs: object
    ) -> str:
        """Produce the assistant response for the supplied prompt."""


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


_MLX_FUNCS: Tuple[LoadFn, GenerateFn] | None = None


def _ensure_mlx_functions() -> Tuple[LoadFn, GenerateFn]:
    """Locate and cache mlx-lm helpers, surfacing a clear error if unavailable."""
    global _MLX_FUNCS
    if _MLX_FUNCS is not None:
        return _MLX_FUNCS

    try:
        from mlx_lm import generate as mlx_generate
        from mlx_lm import load as mlx_load
    except ImportError as exc:  # pragma: no cover - requires missing dependency
        raise RuntimeError(
            "mlx-lm is required for the demo. Install extras via `uv sync --group demo`."
        ) from exc

    _MLX_FUNCS = (mlx_load, mlx_generate)
    return _MLX_FUNCS


def load_model(model_id: str) -> ModelBundle:
    """Load the requested model and tokenizer."""
    load_fn, _ = _ensure_mlx_functions()
    return load_fn(model_id)


def render_messages(messages: Sequence[Message]) -> str:
    """Convert structured chat messages into a text prompt for mlx-lm."""
    parts: list[str] = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"{role}: {message['content']}\n")
    parts.append("ASSISTANT: ")  # steer the model toward a continuation
    return "".join(parts)


def chat_once(
    bundle: ModelBundle,
    messages: Sequence[Message],
    *,
    max_tokens: int = 256,
    temperature: float | None = 0.0,
) -> str:
    """Generate a single assistant response using mlx-lm."""
    model, tokenizer = bundle
    _, generate_fn = _ensure_mlx_functions()
    prompt = render_messages(messages)

    kwargs: dict[str, object] = {"max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        raw_output = generate_fn(model, tokenizer, prompt, **kwargs)
    except TypeError as exc:
        if "temperature" in str(exc) and "temperature" in kwargs:
            kwargs.pop("temperature", None)
            raw_output = generate_fn(model, tokenizer, prompt, **kwargs)
        else:  # pragma: no cover - defensive passthrough for unexpected signatures
            raise

    return raw_output.strip()


__all__ = [
    "ModelBundle",
    "Message",
    "chat_once",
    "load_model",
]
