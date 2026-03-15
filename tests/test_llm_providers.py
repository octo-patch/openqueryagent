"""Tests for Ollama, Bedrock, and MiniMax LLM providers."""

from __future__ import annotations

from openqueryagent.llm.bedrock import BedrockProvider


class TestOllamaProvider:
    def test_model_name(self) -> None:
        # Ollama needs httpx to init; test lazily
        try:
            from openqueryagent.llm.ollama import OllamaProvider
            provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
            assert provider.model_name == "llama3"
        except ImportError:
            pass  # httpx not available in test env


class TestMiniMaxProviderInit:
    def test_model_name(self) -> None:
        try:
            from openqueryagent.llm.minimax import MiniMaxProvider
            provider = MiniMaxProvider(model="MiniMax-M2.5", api_key="test-key")
            assert provider.model_name == "MiniMax-M2.5"
        except ImportError:
            pass  # openai not available in test env

    def test_default_model(self) -> None:
        try:
            from openqueryagent.llm.minimax import MiniMaxProvider
            provider = MiniMaxProvider(api_key="test-key")
            assert provider.model_name == "MiniMax-M2.5"
        except ImportError:
            pass

    def test_highspeed_model(self) -> None:
        try:
            from openqueryagent.llm.minimax import MiniMaxProvider
            provider = MiniMaxProvider(model="MiniMax-M2.5-highspeed", api_key="test-key")
            assert provider.model_name == "MiniMax-M2.5-highspeed"
        except ImportError:
            pass


class TestBedrockProvider:
    def test_model_name(self) -> None:
        try:
            provider = BedrockProvider(model="anthropic.claude-3-sonnet", region="us-east-1")
            assert provider.model_name == "anthropic.claude-3-sonnet"
        except ImportError:
            pass  # aiobotocore not available

    def test_build_body_anthropic(self) -> None:
        try:
            provider = BedrockProvider(model="anthropic.claude-3-sonnet")
        except ImportError:
            return

        from openqueryagent.core.types import ChatMessage
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello"),
        ]
        body = provider._build_request_body(messages, 0.0, 100)
        assert "anthropic_version" in body
        assert body["system"] == "You are helpful"
        assert len(body["messages"]) == 1  # only user messages

    def test_build_body_titan(self) -> None:
        try:
            provider = BedrockProvider(model="amazon.titan-text-express-v1")
        except ImportError:
            return

        from openqueryagent.core.types import ChatMessage
        messages = [ChatMessage(role="user", content="Hello")]
        body = provider._build_request_body(messages, 0.0, 100)
        assert "inputText" in body

    def test_parse_response_anthropic(self) -> None:
        try:
            provider = BedrockProvider(model="anthropic.claude-3-sonnet")
        except ImportError:
            return

        resp = provider._parse_response({
            "content": [{"type": "text", "text": "Hello world"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        })
        assert resp.content == "Hello world"
        assert resp.usage.prompt_tokens == 10

    def test_parse_response_titan(self) -> None:
        try:
            provider = BedrockProvider(model="amazon.titan-text-express-v1")
        except ImportError:
            return

        resp = provider._parse_response({
            "results": [{"outputText": "Hello!", "tokenCount": 3, "completionReason": "FINISH"}],
            "inputTextTokenCount": 5,
        })
        assert resp.content == "Hello!"
        assert resp.usage.prompt_tokens == 5
