"""Tests for LiteLLM batch chat completions client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fenic._inference.litellm.litellm_batch_chat_completions_client import LiteLLMBatchChatCompletionsClient
from fenic._inference.model_client import FatalException, TransientException
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy
from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages, FewShotExample
from fenic.core._inference.model_catalog import ModelProvider


@pytest.fixture
def rate_limit_strategy():
    """Create a basic rate limit strategy for testing."""
    return UnifiedTokenRateLimitStrategy(rpm=100, tpm=10000)


@pytest.fixture
def litellm_client(rate_limit_strategy):
    """Create a LiteLLM batch client for testing."""
    return LiteLLMBatchChatCompletionsClient(
        rate_limit_strategy=rate_limit_strategy,
        model="qwen3:30b",
        queue_size=10,
        max_backoffs=3,
        api_base="http://localhost:11434"
    )


def test_litellm_client_initialization(litellm_client):
    """Test LiteLLM client initialization."""
    assert litellm_client.model == "qwen3:30b"
    assert litellm_client.model_provider == ModelProvider.LITELLM
    assert litellm_client._api_base == "http://localhost:11434"


@pytest.mark.anyio
@patch('fenic._inference.litellm.litellm_batch_chat_completions_client.litellm')
async def test_litellm_client_successful_request(mock_litellm, litellm_client):
    """Test successful LiteLLM API request."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 150

    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    # Create test request
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        user="Hello, how are you?",
        examples=[]
    )
    request = FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=100,
        top_logprobs=None,
        structured_output=None,
        temperature=0.7
    )

    # Make request
    result = await litellm_client.make_single_request(request)

    # Verify response
    assert result is not None
    assert hasattr(result, 'completion')
    assert result.completion == "Test response"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150

    # Verify API call
    mock_litellm.acompletion.assert_called_once()
    call_args = mock_litellm.acompletion.call_args[1]
    assert call_args["model"] == "qwen3:30b"
    assert call_args["api_base"] == "http://localhost:11434"
    assert call_args["max_tokens"] == 100
    assert call_args["temperature"] == 0.7


@pytest.mark.anyio
@patch('fenic._inference.litellm.litellm_batch_chat_completions_client.litellm')
async def test_litellm_client_structured_output(mock_litellm, litellm_client):
    """Test LiteLLM request with structured output."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 120
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 140

    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    # Create test request with structured output
    messages = LMRequestMessages(
        system="Extract information from text.",
        user="John is 30 years old.",
        examples=[]
    )

    # Mock structured output
    structured_output = MagicMock()
    structured_output.json_schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}

    request = FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=50,
        top_logprobs=None,
        structured_output=structured_output,
        temperature=0.0
    )

    # Make request
    result = await litellm_client.make_single_request(request)

    # Verify response
    assert result is not None
    assert result.completion == '{"name": "John", "age": 30}'

    # Verify API call includes JSON mode
    mock_litellm.acompletion.assert_called_once()
    call_args = mock_litellm.acompletion.call_args[1]
    assert call_args["response_format"] == {"type": "json_object"}


@pytest.mark.anyio
@patch('fenic._inference.litellm.litellm_batch_chat_completions_client.litellm')
async def test_litellm_client_rate_limit_error(mock_litellm, litellm_client):
    """Test handling of rate limit errors."""
    mock_litellm.acompletion = AsyncMock(side_effect=Exception("Rate limit exceeded - 429"))

    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        user="Hello",
        examples=[]
    )
    request = FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=50,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0
    )

    result = await litellm_client.make_single_request(request)

    assert isinstance(result, TransientException)


@pytest.mark.anyio
@patch('fenic._inference.litellm.litellm_batch_chat_completions_client.litellm')
async def test_litellm_client_fatal_error(mock_litellm, litellm_client):
    """Test handling of fatal errors."""
    mock_litellm.acompletion = AsyncMock(side_effect=ValueError("Invalid model"))

    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        user="Hello",
        examples=[]
    )
    request = FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=50,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0
    )

    result = await litellm_client.make_single_request(request)

    assert isinstance(result, FatalException)


def test_litellm_client_token_estimation(litellm_client):
    """Test token estimation for requests."""
    messages = LMRequestMessages(
        system="You are a helpful assistant.",
        user="Hello, how are you doing today?",
        examples=[FewShotExample(user="Hi", assistant="Hello!")]
    )
    request = FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=100,
        top_logprobs=None,
        structured_output=None,
        temperature=0.7
    )

    estimate = litellm_client.estimate_tokens_for_request(request)

    assert estimate.input_tokens > 0  # Should have some input tokens
    assert estimate.output_tokens == 100  # Should match max_completion_tokens


def test_litellm_client_request_key_generation(litellm_client):
    """Test request key generation for deduplication."""
    messages = LMRequestMessages(
        system="Test system",
        user="Test user",
        examples=[]
    )
    request = FenicCompletionsRequest(
        messages=messages,
        max_completion_tokens=50,
        top_logprobs=None,
        structured_output=None,
        temperature=0.0
    )

    key1 = litellm_client.get_request_key(request)
    key2 = litellm_client.get_request_key(request)

    assert key1 == key2  # Same request should generate same key
    assert isinstance(key1, str)
    assert len(key1) > 0


def test_litellm_client_metrics(litellm_client):
    """Test metrics functionality."""
    initial_metrics = litellm_client.get_metrics()
    assert initial_metrics.num_requests == 0
    assert initial_metrics.cost == 0.0

    # Reset metrics should work
    litellm_client.reset_metrics()
    reset_metrics = litellm_client.get_metrics()
    assert reset_metrics.num_requests == 0
    assert reset_metrics.cost == 0.0