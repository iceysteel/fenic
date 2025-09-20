"""LiteLLM model provider implementation."""

import logging
import os
from typing import Optional

import litellm

from fenic.core._inference.model_provider import ModelProviderClass

logger = logging.getLogger(__name__)


class LiteLLMModelProvider(ModelProviderClass):
    """LiteLLM implementation of ModelProvider for local models via Ollama."""

    def __init__(self, api_base: Optional[str] = None):
        """Initialize LiteLLM provider.

        Args:
            api_base: Base URL for the LiteLLM/Ollama server. Defaults to http://localhost:11434
        """
        self.api_base = api_base or os.getenv("LITELLM_API_BASE", "http://localhost:11434")
        # Configure litellm with the API base
        litellm.api_base = self.api_base
        # Disable litellm logging to reduce noise
        litellm.set_verbose = False

    @property
    def name(self) -> str:
        return "litellm"

    def create_client(self):
        """Create a LiteLLM sync client instance."""
        # LiteLLM doesn't have a dedicated client class - we use the module functions
        # Return a simple wrapper that maintains the API base configuration
        return LiteLLMClientWrapper(self.api_base, async_mode=False)

    def create_aio_client(self):
        """Create a LiteLLM async client instance."""
        # Return async wrapper
        return LiteLLMClientWrapper(self.api_base, async_mode=True)

    async def validate_api_key(self) -> None:
        """Validate LiteLLM connection by checking server accessibility."""
        try:
            # For local LiteLLM/Ollama, just check if the server is reachable
            # We'll skip model-specific validation since models may not be pulled yet
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.api_base}/api/tags", timeout=5.0)
                if response.status_code == 200:
                    logger.debug(f"LiteLLM server accessible at {self.api_base}")
                else:
                    logger.warning(f"LiteLLM server returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"LiteLLM connection check failed: {e}. Proceeding anyway for local models.")


class LiteLLMClientWrapper:
    """Wrapper class to provide a client-like interface for LiteLLM functions."""

    def __init__(self, api_base: str, async_mode: bool = False):
        self.api_base = api_base
        self.async_mode = async_mode

    async def completion(self, **kwargs):
        """Async completion wrapper."""
        kwargs.setdefault("api_base", self.api_base)
        if self.async_mode:
            return await litellm.acompletion(**kwargs)
        else:
            return litellm.completion(**kwargs)

    def sync_completion(self, **kwargs):
        """Sync completion wrapper."""
        kwargs.setdefault("api_base", self.api_base)
        return litellm.completion(**kwargs)