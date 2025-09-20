"""Client for making batch requests to LiteLLM's embeddings API."""

import hashlib
import logging
from typing import Union

import litellm

from fenic._inference.litellm.litellm_provider import LiteLLMModelProvider
from fenic._inference.model_client import (
    FatalException,
    ModelClient,
    TransientException,
)
from fenic._inference.rate_limit_strategy import (
    RateLimitStrategy,
    TokenEstimate,
    UnifiedTokenRateLimitStrategy,
)
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import FenicEmbeddingsRequest
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.metrics import RMMetrics

logger = logging.getLogger(__name__)


class LiteLLMBatchEmbeddingsClient(ModelClient[FenicEmbeddingsRequest, list[float]]):
    """Client for making batch requests to LiteLLM's embeddings API.

    This client enables communication with local embedding models via Ollama through LiteLLM's
    unified interface.
    """

    def __init__(
        self,
        rate_limit_strategy: RateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        api_base: str = "http://localhost:11434",
    ):
        """Initialize the LiteLLM batch embeddings client.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            model: The model to use (e.g., "nomic-embed-text")
            queue_size: Size of the request queue
            max_backoffs: Maximum number of backoff attempts
            api_base: Base URL for LiteLLM/Ollama server
        """
        # Use tiktoken for token counting as a reasonable approximation for local models
        token_counter = TiktokenTokenCounter(model_name="text-embedding-ada-002", fallback_encoding="cl100k_base")

        super().__init__(
            model=model,
            model_provider=ModelProvider.LITELLM,
            model_provider_class=LiteLLMModelProvider(api_base=api_base),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
        )

        self._model_parameters = model_catalog.get_embedding_model_parameters(
            ModelProvider.LITELLM, model
        )
        self._api_base = api_base
        self._metrics = RMMetrics()

        # Configure litellm settings
        litellm.api_base = self._api_base
        litellm.set_verbose = False

    async def make_single_request(
        self, request: FenicEmbeddingsRequest
    ) -> Union[None, list[float], TransientException, FatalException]:
        """Make a single request to the LiteLLM API.

        Args:
            request: The request to make

        Returns:
            The embedding vector or an exception
        """
        try:
            # Add ollama/ prefix for LiteLLM routing if not already present
            model_name = self.model if self.model.startswith("ollama/") else f"ollama/{self.model}"

            # Make the API call
            response = await litellm.aembedding(
                model=model_name,
                input=request.doc,
                api_base=self._api_base,
            )

            # Extract usage information if available
            usage = getattr(response, 'usage', None)
            if usage and hasattr(usage, 'total_tokens'):
                total_tokens = usage.total_tokens
            else:
                # Fallback to token estimation
                total_tokens = self.token_counter.count_tokens(request.doc)

            # Update metrics
            self._metrics.num_input_tokens += total_tokens
            self._metrics.num_requests += 1
            # Local models are typically free, so cost is 0
            self._metrics.cost += 0.0

            # Extract the embedding vector
            if hasattr(response, 'data') and response.data and len(response.data) > 0:
                embedding = response.data[0].embedding
                return embedding
            else:
                logger.error(f"No embedding data in LiteLLM response: {response}")
                return FatalException(Exception("No embedding data in response"))

        except Exception as e:
            # Handle common transient errors that should be retried
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit exceeded for LiteLLM embeddings: {e}")
                return TransientException(e)
            elif "connection" in error_str or "timeout" in error_str or "503" in error_str:
                logger.warning(f"Connection/timeout error with LiteLLM embeddings: {e}")
                return TransientException(e)
            else:
                logger.error(f"Fatal error with LiteLLM embeddings request: {e}")
                return FatalException(e)

    def get_request_key(self, request: FenicEmbeddingsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return hashlib.sha256(request.doc.encode()).hexdigest()[:10]

    def estimate_tokens_for_request(self, request: FenicEmbeddingsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate with input token count
        """
        input_tokens = self.token_counter.count_tokens(request.doc)
        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=0  # Embedding models don't generate output tokens
        )

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._metrics = RMMetrics()

    def get_metrics(self) -> RMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._metrics

    def _get_max_output_tokens(self, request: FenicEmbeddingsRequest) -> int:
        """Embedding models don't have output tokens."""
        return 0