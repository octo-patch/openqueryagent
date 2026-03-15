"""MiniMax LLM provider.

Implements the LLMProvider protocol using the OpenAI-compatible MiniMax API.
Supports Chat Completions with streaming and JSON response format.

MiniMax offers models like MiniMax-M2.5 with 204K context window.
API docs: https://platform.minimaxi.com/

Requires ``openai`` (install with ``pip install openqueryagent[minimax]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from openqueryagent.core.exceptions import RateLimitError
from openqueryagent.core.types import ChatMessage, TokenUsage
from openqueryagent.llm.base import LLMChunk, LLMResponse, ResponseFormat

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = structlog.get_logger(__name__)

_MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MiniMaxProvider:
    """MiniMax LLM provider via OpenAI-compatible API.

    Args:
        model: Model identifier (e.g., ``MiniMax-M2.5``, ``MiniMax-M2.5-highspeed``).
        api_key: MiniMax API key. Falls back to ``MINIMAX_API_KEY`` env var.
        api_base: Custom API base URL. Defaults to ``https://api.minimax.io/v1``.
        max_retries: Maximum retry attempts for transient errors.
    """

    def __init__(
        self,
        model: str = "MiniMax-M2.5",
        api_key: str | None = None,
        api_base: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._max_retries = max_retries
        self._client: Any = None
        self._init_client(api_key, api_base)

    def _init_client(self, api_key: str | None, api_base: str | None) -> None:
        """Initialize the OpenAI client pointed at MiniMax."""
        try:
            import openai
        except ImportError as e:
            msg = "openai is not installed. Install with: pip install openqueryagent[minimax]"
            raise ImportError(msg) from e

        import os

        resolved_key = api_key or os.environ.get("MINIMAX_API_KEY")

        self._client = openai.AsyncOpenAI(
            api_key=resolved_key,
            base_url=api_base or _MINIMAX_BASE_URL,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @staticmethod
    def _clamp_temperature(temperature: float) -> float:
        """Ensure temperature is within MiniMax's accepted range (0, 1].

        MiniMax rejects temperature=0.0; use a small positive value instead.
        """
        if temperature <= 0.0:
            return 0.01
        return min(temperature, 1.0)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: ResponseFormat | None = None,
    ) -> LLMResponse:
        """Generate a completion via MiniMax Chat API.

        Args:
            messages: Conversation messages.
            temperature: Sampling temperature (clamped to MiniMax's range).
            max_tokens: Maximum response tokens.
            response_format: TEXT or JSON output format.

        Returns:
            LLMResponse with content and token usage.

        Raises:
            RateLimitError: If rate-limited after retries.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._clamp_temperature(temperature),
            "max_tokens": max_tokens,
        }

        if response_format == ResponseFormat.JSON:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(
                    f"MiniMax rate limit exceeded: {e}",
                    provider="minimax",
                    model=self._model,
                ) from e
            raise

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
            ),
            finish_reason=choice.finish_reason or "",
        )

    async def complete_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[LLMChunk]:
        """Generate a streaming completion via MiniMax Chat API.

        Yields:
            LLMChunk objects with incremental content.
        """
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._clamp_temperature(temperature),
            "max_tokens": max_tokens,
            "stream": True,
        }

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    yield LLMChunk(
                        content=delta.content or "",
                        model=chunk.model or self._model,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitError(
                    f"MiniMax rate limit exceeded: {e}",
                    provider="minimax",
                    model=self._model,
                ) from e
            raise


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if exception is a rate limit error."""
    try:
        import openai

        return isinstance(exc, openai.RateLimitError)
    except ImportError:
        return "rate_limit" in str(exc).lower() or "429" in str(exc)
