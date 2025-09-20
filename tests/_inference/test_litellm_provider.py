"""Tests for LiteLLM provider implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fenic._inference.litellm.litellm_provider import LiteLLMModelProvider, LiteLLMClientWrapper


def test_litellm_provider_properties():
    """Test basic LiteLLM provider properties."""
    provider = LiteLLMModelProvider()

    assert provider.name == "litellm"
    assert provider.api_base == "http://localhost:11434"  # Default


def test_litellm_provider_custom_api_base():
    """Test LiteLLM provider with custom API base."""
    custom_base = "http://custom-host:8080"
    provider = LiteLLMModelProvider(api_base=custom_base)

    assert provider.api_base == custom_base


def test_litellm_provider_creates_clients():
    """Test that LiteLLM provider creates client wrappers."""
    provider = LiteLLMModelProvider()

    sync_client = provider.create_client()
    async_client = provider.create_aio_client()

    assert isinstance(sync_client, LiteLLMClientWrapper)
    assert isinstance(async_client, LiteLLMClientWrapper)
    assert sync_client.api_base == provider.api_base
    assert async_client.api_base == provider.api_base
    assert not sync_client.async_mode
    assert async_client.async_mode


@pytest.mark.anyio
@patch('fenic._inference.litellm.litellm_provider.litellm')
async def test_litellm_provider_validation_success(mock_litellm):
    """Test successful API key validation."""
    # Mock successful completion
    mock_response = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    provider = LiteLLMModelProvider()

    # Should not raise an exception
    await provider.validate_api_key()

    # Verify the call was made with correct parameters
    mock_litellm.acompletion.assert_called_once()
    call_args = mock_litellm.acompletion.call_args[1]  # Get keyword arguments
    assert call_args["model"] == "ollama_chat/llama2"
    assert call_args["api_base"] == "http://localhost:11434"
    assert call_args["max_tokens"] == 5


@pytest.mark.anyio
@patch('fenic._inference.litellm.litellm_provider.litellm')
async def test_litellm_provider_validation_failure(mock_litellm):
    """Test API key validation failure."""
    # Mock failed completion
    mock_litellm.acompletion = AsyncMock(side_effect=Exception("Connection failed"))

    provider = LiteLLMModelProvider()

    # Should raise the exception
    with pytest.raises(Exception, match="Connection failed"):
        await provider.validate_api_key()


def test_litellm_client_wrapper_properties():
    """Test LiteLLM client wrapper properties."""
    api_base = "http://test:1234"

    sync_wrapper = LiteLLMClientWrapper(api_base, async_mode=False)
    async_wrapper = LiteLLMClientWrapper(api_base, async_mode=True)

    assert sync_wrapper.api_base == api_base
    assert async_wrapper.api_base == api_base
    assert not sync_wrapper.async_mode
    assert async_wrapper.async_mode


@pytest.mark.anyio
@patch('fenic._inference.litellm.litellm_provider.litellm')
async def test_litellm_client_wrapper_async_completion(mock_litellm):
    """Test async completion through client wrapper."""
    mock_response = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    wrapper = LiteLLMClientWrapper("http://test:1234", async_mode=True)

    result = await wrapper.completion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}]
    )

    assert result == mock_response
    mock_litellm.acompletion.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        api_base="http://test:1234"
    )


@patch('fenic._inference.litellm.litellm_provider.litellm')
def test_litellm_client_wrapper_sync_completion(mock_litellm):
    """Test sync completion through client wrapper."""
    mock_response = MagicMock()
    mock_litellm.completion = MagicMock(return_value=mock_response)

    wrapper = LiteLLMClientWrapper("http://test:1234", async_mode=False)

    result = wrapper.sync_completion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}]
    )

    assert result == mock_response
    mock_litellm.completion.assert_called_once_with(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
        api_base="http://test:1234"
    )