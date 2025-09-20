#!/usr/bin/env python3
"""
Fenic Hello World with Local LLMs (LiteLLM + Ollama)

This example demonstrates how to use Fenic with local LLMs via LiteLLM and Ollama
instead of cloud APIs. Perfect for privacy-sensitive workflows, cost control, or
offline development!

Prerequisites:
1. Install Ollama from https://ollama.ai
2. Run: ollama pull qwen3:30b (or gpt-oss, deepseek-r1)
3. Ensure Ollama is running on http://localhost:11434
4. Install: pip install fenic litellm

Benefits:
- 🔒 Complete privacy (no external API calls)
- 💰 Zero inference costs
- 🚀 No network dependency
- ⚙️ Full control over models
"""

import asyncio
import json
import sys
from typing import List, Dict, Any

# Check if required packages are available
try:
    import fenic as fc
    from pydantic import BaseModel, Field
    from fenic.core._inference.model_catalog import ModelProvider
    from fenic._inference.litellm.litellm_batch_chat_completions_client import LiteLLMBatchChatCompletionsClient
    from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy
    from fenic._inference.types import FenicCompletionsRequest, LMRequestMessages
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("📦 Install with: pip install fenic litellm")
    sys.exit(1)


# Pydantic models for structured extraction
class ErrorAnalysis(BaseModel):
    root_cause: str = Field(description="The root cause of this error")
    fix_recommendation: str = Field(description="How to fix this error")


class ErrorPattern(BaseModel):
    error_type: str = Field(description="Type of error (e.g., NullPointer, Timeout, ConnectionRefused)")
    component: str = Field(description="Affected component or system")


# Sample error logs data
ERROR_LOGS = [
    {
        "timestamp": "2024-01-20 14:23:45",
        "service": "api-gateway",
        "error_log": """
ERROR: NullPointerException in UserService.getProfile()
    at com.app.UserService.getProfile(UserService.java:45)
    at com.app.ApiController.handleRequest(ApiController.java:123)
    at java.base/java.lang.Thread.run(Thread.java:834)

User ID: 12345 was not found in cache, attempted DB lookup returned null
        """
    },
    {
        "timestamp": "2024-01-20 14:24:12",
        "service": "auth-service",
        "error_log": """
node:internal/process/promises:288
            triggerUncaughtException(err, true /* fromPromise */);
            ^

Error: connect ECONNREFUSED 127.0.0.1:6379
    at TCPConnectWrap.afterConnect [as oncomplete] (node:net:1494:16)
    at Protocol._enqueue (/app/node_modules/redis/lib/redis.js:458:48)
    at Protocol._write (/app/node_modules/redis/lib/redis.js:326:10)

Redis connection failed during session validation
        """
    },
    {
        "timestamp": "2024-01-20 14:25:33",
        "service": "payment-processor",
        "error_log": """
Traceback (most recent call last):
  File "/app/payment/processor.py", line 89, in process_payment
    response = stripe.Charge.create(
  File "/usr/local/lib/python3.9/site-packages/stripe/api_resources/charge.py", line 45, in create
    return cls._static_request("post", cls.class_url(), params=params)
  File "/usr/local/lib/python3.9/site-packages/stripe/api_requestor.py", line 234, in request
    raise error.APIConnectionError(msg)
stripe.error.APIConnectionError: Connection error: timeout after 30s

Payment processing failed for order_id: ORD-789456
        """
    },
    {
        "timestamp": "2024-01-20 14:26:01",
        "service": "data-pipeline",
        "error_log": """
django.db.utils.OperationalError: could not connect to server: Connection refused
    Is the server running on host "db.prod.internal" (10.0.1.50) and accepting
    TCP/IP connections on port 5432?

FATAL: Batch job 'daily_analytics' failed after 3 retries
Table 'user_metrics' has 2.3M pending records
        """
    },
    {
        "timestamp": "2024-01-20 14:27:15",
        "service": "frontend",
        "error_log": """
TypeError: Cannot read property 'map' of undefined
    at ProfileList (ProfileList.jsx:34:19)
    at renderWithHooks (react-dom.development.js:14985:18)
    at updateFunctionComponent (react-dom.development.js:17356:20)

API response was: {"error": "rate_limit_exceeded", "retry_after": 60}
Component tried to render before data loaded
        """
    }
]


class LocalLLMErrorAnalyzer:
    """Error log analyzer using local LLMs via LiteLLM."""

    def __init__(self, model_name: str = "ollama/qwen3:30b", api_base: str = "http://localhost:11434"):
        """Initialize the analyzer with local LLM configuration."""
        self.model_name = model_name
        self.api_base = api_base
        self.session = None
        self.llm_client = None

    def setup(self):
        """Set up Fenic session and LiteLLM client."""
        print("🔧 Setting up Fenic with local LLM...")

        # Configure Fenic session
        config = fc.SessionConfig(
            app_name="hello_debug_local",
            semantic=fc.SemanticConfig(
                default_language_model="local_model"
            )
        )

        self.session = fc.Session.get_or_create(config)

        # Set up local LLM client
        rate_limit_strategy = UnifiedTokenRateLimitStrategy(rpm=100, tpm=10000)

        self.llm_client = LiteLLMBatchChatCompletionsClient(
            rate_limit_strategy=rate_limit_strategy,
            model=self.model_name,
            queue_size=50,
            max_backoffs=3,
            api_base=self.api_base
        )

        print(f"✅ Session configured with local LLM: {self.model_name}")
        print(f"📍 Ollama endpoint: {self.api_base}")
        print(f"💰 Cost: $0.00 (local inference)")

    async def analyze_error_with_local_llm(self, error_log: str) -> Dict[str, Any]:
        """Analyze a single error log using the local LLM."""

        system_prompt = """
You are an expert software engineer analyzing error logs.
For the given error log, provide a JSON response with:
1. severity: one of "low", "medium", "high", "critical"
2. root_cause: brief description of what caused the error
3. fix_recommendation: specific steps to fix the issue

Respond only with valid JSON matching this format:
{
  "severity": "high",
  "root_cause": "...",
  "fix_recommendation": "..."
}
"""

        user_prompt = f"Analyze this error log:\n\n{error_log}"

        # Create request
        messages = LMRequestMessages(
            system=system_prompt,
            user=user_prompt,
            examples=[]
        )

        request = FenicCompletionsRequest(
            messages=messages,
            max_completion_tokens=200,
            top_logprobs=None,
            structured_output=None,
            temperature=0.1
        )

        # Make request to local LLM
        response = await self.llm_client.make_single_request(request)

        if hasattr(response, 'completion'):
            raw_response = response.completion
            print(f"    🔍 Raw LLM response: {raw_response[:200]}...")

            try:
                # Clean up the response - sometimes LLMs add extra text
                response_text = raw_response.strip()

                # Try to find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')

                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx + 1]
                    analysis = json.loads(json_text)

                    # Validate required fields
                    required_fields = ['severity', 'root_cause', 'fix_recommendation']
                    if all(field in analysis for field in required_fields):
                        return analysis
                    else:
                        print(f"    ⚠️  Missing fields in response: {[f for f in required_fields if f not in analysis]}")
                        # Fill in missing fields
                        for field in required_fields:
                            if field not in analysis:
                                analysis[field] = "Not provided"
                        return analysis
                else:
                    print(f"    ⚠️  No valid JSON found in response")
                    return {
                        "severity": "medium",
                        "root_cause": f"Invalid response format: {raw_response[:100]}",
                        "fix_recommendation": "Improve prompt or try different model"
                    }

            except json.JSONDecodeError as e:
                print(f"    ⚠️  JSON parsing failed: {e}")
                print(f"    📝 Attempted to parse: {json_text if 'json_text' in locals() else raw_response[:100]}")
                return {
                    "severity": "medium",
                    "root_cause": f"Failed to parse LLM response: {str(e)}",
                    "fix_recommendation": "Check LLM prompt and response format"
                }
        else:
            print(f"    ❌ No completion in response: {response}")
            return {
                "severity": "unknown",
                "root_cause": "LLM request failed - no completion returned",
                "fix_recommendation": "Check LLM connection and model availability"
            }

    async def analyze_logs(self, logs: List[Dict[str, Any]], max_logs: int = None) -> List[Dict[str, Any]]:
        """Analyze multiple error logs."""

        logs_to_process = logs[:max_logs] if max_logs else logs

        print(f"\n🤖 Starting local LLM analysis of {len(logs_to_process)} logs...")
        print("⏳ This may take a moment with local models...\n")

        results = []

        for i, log_entry in enumerate(logs_to_process):
            print(f"Analyzing log {i+1}/{len(logs_to_process)}: {log_entry['service']}...")

            # Run async analysis
            analysis = await self.analyze_error_with_local_llm(log_entry['error_log'])

            result = {
                'timestamp': log_entry['timestamp'],
                'service': log_entry['service'],
                'severity': analysis['severity'],
                'root_cause': analysis['root_cause'],
                'fix_recommendation': analysis['fix_recommendation']
            }
            results.append(result)

            print(f"  ✅ Severity: {analysis['severity']}")
            print(f"  📋 Root Cause: {analysis['root_cause'][:60]}...\n")

        return results

    def display_results(self, results: List[Dict[str, Any]]):
        """Display analysis results in a readable format."""

        print("\n" + "="*70)
        print("ERROR ANALYSIS RESULTS (Local LLM)")
        print("="*70)

        for result in results:
            print(f"\n🕐 {result['timestamp']} | 🔧 {result['service']}")
            print(f"🚨 Severity: {result['severity'].upper()}")
            print(f"🔍 Root Cause: {result['root_cause']}")
            print(f"💡 Fix: {result['fix_recommendation']}")
            print("-" * 70)

    def display_critical_errors(self, results: List[Dict[str, Any]]):
        """Display only critical and high severity errors."""

        critical_errors = [r for r in results if r['severity'] in ['critical', 'high']]

        if critical_errors:
            print("\n" + "="*70)
            print("CRITICAL ERRORS REQUIRING IMMEDIATE ATTENTION")
            print("="*70)

            for result in critical_errors:
                print(f"\n⚠️  URGENT: {result['service']} - {result['severity'].upper()}")
                print(f"🕐 Time: {result['timestamp']}")
                print(f"🔍 Cause: {result['root_cause']}")
                print(f"💡 Action: {result['fix_recommendation']}")
                print("-" * 70)
        else:
            print("\n✅ No critical errors found!")

    def display_metrics(self):
        """Display processing metrics."""

        metrics = self.llm_client.get_metrics()

        print("\n📊 LOCAL LLM PROCESSING METRICS")
        print("=" * 50)
        print(f"🔢 Total Requests: {metrics.num_requests}")
        print(f"📝 Input Tokens: {metrics.num_uncached_input_tokens:,}")
        print(f"📤 Output Tokens: {metrics.num_output_tokens:,}")
        print(f"💰 Total Cost: ${metrics.cost:.2f} (FREE!)")
        print(f"🏠 Model: {self.model_name} (Local)")
        print(f"🔗 Endpoint: {self.api_base}")
        print("\n✅ All processing completed locally - no data sent to external APIs!")

    def cleanup(self):
        """Clean up resources."""
        if self.session:
            self.session.stop()
        print("\n🧹 Session cleaned up")


def check_ollama_connection(api_base: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    try:
        import requests
        response = requests.get(f"{api_base}/api/version", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


async def main():
    """Main execution function."""

    print("🏠 Fenic Hello World - Local LLM Edition")
    print("=" * 70)
    print("Error Log Analyzer using LiteLLM + Ollama")
    print("No API keys required - 100% local processing!")
    print("=" * 70)

    # Check Ollama connection
    api_base = "http://localhost:11434"
    if not check_ollama_connection(api_base):
        print(f"❌ Cannot connect to Ollama at {api_base}")
        print("📋 Please ensure:")
        print("   1. Ollama is installed (https://ollama.ai)")
        print("   2. Ollama is running: ollama serve")
        print("   3. Model is available: ollama pull qwen3:30b")
        sys.exit(1)

    print(f"✅ Ollama connection verified at {api_base}")

    # Initialize analyzer
    analyzer = LocalLLMErrorAnalyzer(model_name="ollama/qwen3:30b", api_base=api_base)

    try:
        # Setup
        analyzer.setup()

        # Load data
        print(f"\n📊 Loaded {len(ERROR_LOGS)} error log entries for analysis")

        # Analyze logs (limit to first 3 for demo)
        results = await analyzer.analyze_logs(ERROR_LOGS, max_logs=3)

        # Display results
        analyzer.display_results(results)

        # Show critical errors
        analyzer.display_critical_errors(results)

        # Show metrics
        analyzer.display_metrics()

        # Success message
        print("\n🎉 Local LLM Analysis Complete!")
        print("\n🔧 **Next Steps with Local Models:**")
        print("   • Try different Ollama models (llama3, mistral, codellama)")
        print("   • Process larger datasets without API costs")
        print("   • Fine-tune models for your specific error patterns")
        print("   • Build always-available, offline analysis pipelines")
        print("\n💡 **Model Alternatives:**")
        print("   • ollama/qwen3:30b (current) - Great for analysis tasks")
        print("   • ollama/gpt-oss - Open source GPT-style model")
        print("   • ollama/deepseek-r1 - Reasoning-focused model")

    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("💡 Check your Ollama setup and model availability")
    finally:
        analyzer.cleanup()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())