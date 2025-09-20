"""Client for making batch requests to LiteLLM's chat completions API."""

import json
import logging
from typing import Optional, Union

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
from fenic._inference.request_utils import generate_completion_request_key
from fenic._inference.token_counter import TiktokenTokenCounter
from fenic._inference.types import (
    FenicCompletionsRequest,
    FenicCompletionsResponse,
    ResponseUsage,
)
from fenic.core._inference.model_catalog import ModelProvider, model_catalog
from fenic.core.metrics import LMMetrics

logger = logging.getLogger(__name__)


class LiteLLMBatchChatCompletionsClient(
    ModelClient[FenicCompletionsRequest, FenicCompletionsResponse]
):
    """Client for making batch requests to LiteLLM's chat completions API.

    This client enables communication with local models via Ollama through LiteLLM's
    unified interface. It supports standard chat completions and structured output
    where available.
    """

    def __init__(
        self,
        rate_limit_strategy: RateLimitStrategy,
        model: str,
        queue_size: int = 100,
        max_backoffs: int = 10,
        api_base: Optional[str] = None,
    ):
        """Initialize the LiteLLM batch chat completions client.

        Args:
            rate_limit_strategy: Strategy for handling rate limits
            model: The model to use (e.g., "ollama_chat/llama2")
            queue_size: Size of the request queue
            max_backoffs: Maximum number of backoff attempts
            api_base: Base URL for LiteLLM/Ollama server
        """
        # Use tiktoken for token counting as a reasonable approximation for local models
        token_counter = TiktokenTokenCounter(model_name="gpt-3.5-turbo", fallback_encoding="cl100k_base")

        super().__init__(
            model=model,
            model_provider=ModelProvider.LITELLM,
            model_provider_class=LiteLLMModelProvider(api_base=api_base),
            rate_limit_strategy=rate_limit_strategy,
            queue_size=queue_size,
            max_backoffs=max_backoffs,
            token_counter=token_counter,
        )

        self._model_parameters = model_catalog.get_completion_model_parameters(
            ModelProvider.LITELLM, model
        )
        self._api_base = api_base or "http://localhost:11434"
        self._metrics = LMMetrics()

        # Configure litellm settings
        litellm.api_base = self._api_base
        litellm.set_verbose = False

    async def make_single_request(
        self, request: FenicCompletionsRequest
    ) -> Union[None, FenicCompletionsResponse, TransientException, FatalException]:
        """Make a single request to the LiteLLM API.

        Args:
            request: The request to make

        Returns:
            The response from the API or an exception
        """
        try:
            # Build the request parameters
            # Add ollama/ prefix for LiteLLM routing if not already present
            model_name = self.model if self.model.startswith("ollama/") else f"ollama/{self.model}"
            common_params = {
                "model": model_name,
                "messages": request.messages.to_message_list(),
                "max_tokens": request.max_completion_tokens,
                "api_base": self._api_base,
            }

            if request.temperature is not None:
                common_params["temperature"] = request.temperature

            # Handle structured output using LiteLLM's JSON mode for local models
            if request.structured_output:
                # Use LiteLLM's format parameter for local models
                common_params["format"] = "json"

                # Add schema to the system message
                import json
                messages = common_params["messages"]
                system_msg_content = messages[0]["content"]
                schema_prompt = f"\n\nPlease respond with valid JSON that matches this schema:\n{json.dumps(request.structured_output.json_schema, indent=2)}"
                messages[0]["content"] = system_msg_content + schema_prompt

            # Also detect if the prompt is asking for JSON and enable JSON mode
            elif "json" in str(request.messages.system).lower() or "json" in str(request.messages.user).lower():
                # Enable JSON mode for prompts that explicitly ask for JSON
                common_params["format"] = "json"

            # Make the API call
            response = await litellm.acompletion(**common_params)

            # Extract the completion text safely
            completion_text = ""
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    completion_text = choice.message.content or ""

            # Handle usage information more robustly
            usage_info = getattr(response, 'usage', None)
            response_usage = None

            if usage_info:
                # Extract token counts safely, defaulting to 0 if not available
                prompt_tokens = getattr(usage_info, 'prompt_tokens', 0)
                completion_tokens = getattr(usage_info, 'completion_tokens', 0)
                total_tokens = getattr(usage_info, 'total_tokens', prompt_tokens + completion_tokens)

                response_usage = ResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=0,  # Local models don't typically have caching
                    thinking_tokens=0,  # Local models don't have separate thinking tokens
                )

                # Update metrics
                self._metrics.num_uncached_input_tokens += response_usage.prompt_tokens
                self._metrics.num_output_tokens += response_usage.completion_tokens
                self._metrics.num_requests += 1
                # Local models are typically free, so cost is 0
                self._metrics.cost += 0.0

            # Create response, ensuring we have valid completion content
            if not completion_text and request.structured_output:
                # For structured output, try to extract any JSON from the response
                completion_text = self._extract_json_from_response(response)

            # Debug logging for structured output
            if request.structured_output and completion_text:
                logger.debug(f"LiteLLM structured response: {completion_text[:200]}...")
                # Validate JSON format
                try:
                    import json
                    json.loads(completion_text)
                    logger.debug("JSON validation successful")
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from LiteLLM: {e}")
                    # Try to clean up common JSON issues
                    completion_text = self._clean_json_response(completion_text)

            return FenicCompletionsResponse(
                completion=completion_text,
                logprobs=None,  # LiteLLM with local models typically doesn't support logprobs
                usage=response_usage,
            )

        except Exception as e:
            # Handle common transient errors that should be retried
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                logger.warning(f"Rate limit exceeded for LiteLLM: {e}")
                return TransientException(e)
            elif "connection" in error_str or "timeout" in error_str or "503" in error_str:
                logger.warning(f"Connection/timeout error with LiteLLM: {e}")
                return TransientException(e)
            else:
                logger.error(f"Fatal error with LiteLLM request: {e}")
                return FatalException(e)

    def get_request_key(self, request: FenicCompletionsRequest) -> str:
        """Generate a unique key for request deduplication.

        Args:
            request: The request to generate a key for

        Returns:
            A unique key for the request
        """
        return generate_completion_request_key(request)

    def estimate_tokens_for_request(self, request: FenicCompletionsRequest) -> TokenEstimate:
        """Estimate the number of tokens for a request.

        Args:
            request: The request to estimate tokens for

        Returns:
            TokenEstimate: The estimated token usage
        """
        input_tokens = self.token_counter.count_tokens(request.messages)
        output_tokens = request.max_completion_tokens

        return TokenEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

    def reset_metrics(self):
        """Reset all metrics to their initial values."""
        self._metrics = LMMetrics()

    def get_metrics(self) -> LMMetrics:
        """Get the current metrics.

        Returns:
            The current metrics
        """
        return self._metrics

    def _get_max_output_tokens(self, request: FenicCompletionsRequest) -> int:
        """Get the maximum output tokens for a request."""
        return request.max_completion_tokens

    def _extract_json_from_response(self, response) -> str:
        """Extract JSON content from LiteLLM response when standard extraction fails."""
        try:
            # Try to get content from various possible response structures
            if hasattr(response, 'choices'):
                for choice in response.choices:
                    if hasattr(choice, 'message'):
                        content = getattr(choice.message, 'content', None)
                        if content:
                            return content
                    elif hasattr(choice, 'text'):
                        return choice.text

            # Try to get content directly from response
            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'text'):
                return response.text

            # Last resort: convert response to string and look for JSON
            response_str = str(response)
            if '{' in response_str and '}' in response_str:
                # Try to extract JSON-like content
                start = response_str.find('{')
                end = response_str.rfind('}') + 1
                if start < end:
                    return response_str[start:end]

            return ""
        except Exception as e:
            logger.warning(f"Failed to extract content from LiteLLM response: {e}")
            return ""

    def _clean_json_response(self, response_text: str) -> str:
        """Clean up common JSON formatting issues from local LLM responses."""
        import re

        # Remove markdown code blocks
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)

        # Remove any text before the first {
        if '{' in response_text:
            start = response_text.find('{')
            response_text = response_text[start:]

        # Remove any text after the last }
        if '}' in response_text:
            end = response_text.rfind('}') + 1
            response_text = response_text[:end]

        # Fix common JSON issues
        response_text = response_text.strip()

        # Try to validate and return
        try:
            import json
            json.loads(response_text)
            return response_text
        except json.JSONDecodeError:
            logger.warning(f"Could not clean JSON response: {response_text[:100]}...")
            return response_text