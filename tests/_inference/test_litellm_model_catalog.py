"""Tests for LiteLLM model catalog integration."""

import pytest
from typing import get_args

from fenic.core._inference.model_catalog import (
    CompletionModelParameters,
    LiteLLMLanguageModelName,
    ModelCatalog,
    ModelProvider,
    model_catalog,
)


def test_litellm_models_have_valid_parameters():
    """Test that all LiteLLM models have valid parameters."""
    catalog = ModelCatalog()

    # Test LiteLLM Language models
    model_names = get_args(LiteLLMLanguageModelName)
    for model_name in model_names:
        params = catalog.get_completion_model_parameters(ModelProvider.LITELLM, model_name)
        assert params is not None and isinstance(params, CompletionModelParameters), (
            f"Could not fetch parameters for LiteLLM model: {model_name}"
        )
        assert hasattr(params, "input_token_cost"), f"Missing input_token_cost for LiteLLM model: {model_name}"
        assert hasattr(params, "output_token_cost"), f"Missing output_token_cost for LiteLLM model: {model_name}"
        assert hasattr(params, "context_window_length"), f"Missing context_window_length for LiteLLM model: {model_name}"
        assert hasattr(params, "max_output_tokens"), f"Missing max_output_tokens for LiteLLM model: {model_name}"


def test_litellm_model_parameters():
    """Test specific LiteLLM model parameters."""
    catalog = model_catalog

    # Test qwen3:30b
    qwen_params = catalog.get_completion_model_parameters(ModelProvider.LITELLM, "qwen3:30b")
    assert qwen_params is not None
    assert qwen_params.input_token_cost == 0.0  # Local models should be free
    assert qwen_params.output_token_cost == 0.0
    assert qwen_params.context_window_length == 16384  # 16k context as specified
    assert qwen_params.max_output_tokens == 4096
    assert qwen_params.supports_custom_temperature is True
    assert qwen_params.supports_profiles is False  # Local models don't have profiles
    assert qwen_params.supports_reasoning is False

    # Test gpt-oss
    gpt_oss_params = catalog.get_completion_model_parameters(ModelProvider.LITELLM, "gpt-oss")
    assert gpt_oss_params is not None
    assert gpt_oss_params.input_token_cost == 0.0
    assert gpt_oss_params.output_token_cost == 0.0
    assert gpt_oss_params.context_window_length == 16384
    assert gpt_oss_params.max_output_tokens == 4096
    assert gpt_oss_params.supports_custom_temperature is True
    assert gpt_oss_params.supports_profiles is False
    assert gpt_oss_params.supports_reasoning is False

    # Test deepseek-r1
    deepseek_params = catalog.get_completion_model_parameters(ModelProvider.LITELLM, "deepseek-r1")
    assert deepseek_params is not None
    assert deepseek_params.input_token_cost == 0.0
    assert deepseek_params.output_token_cost == 0.0
    assert deepseek_params.context_window_length == 16384
    assert deepseek_params.max_output_tokens == 4096
    assert deepseek_params.supports_custom_temperature is True
    assert deepseek_params.supports_profiles is False
    assert deepseek_params.supports_reasoning is False


def test_litellm_provider_in_catalog():
    """Test that LiteLLM provider is properly registered in the catalog."""
    catalog = model_catalog

    # Test that we can get LiteLLM models
    litellm_models = catalog._get_supported_completions_models_by_provider(ModelProvider.LITELLM)
    assert len(litellm_models) == 3  # qwen3:30b, gpt-oss, deepseek-r1

    expected_models = {"qwen3:30b", "gpt-oss", "deepseek-r1"}
    actual_models = set(litellm_models.keys())
    assert actual_models == expected_models


def test_litellm_cost_calculation():
    """Test that LiteLLM models have zero cost calculation."""
    catalog = model_catalog

    for model_name in ["qwen3:30b", "gpt-oss", "deepseek-r1"]:
        cost = catalog.calculate_completion_model_cost(
            model_provider=ModelProvider.LITELLM,
            model_name=model_name,
            uncached_input_tokens=1000,
            cached_input_tokens_read=0,
            output_tokens=500,
        )
        assert cost == 0.0, f"Expected zero cost for local model {model_name}, got {cost}"