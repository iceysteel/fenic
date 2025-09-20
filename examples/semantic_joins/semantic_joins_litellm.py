"""Semantic joins example using fenic with Local LLMs via LiteLLM.

This example demonstrates how to perform LLM-powered semantic joins that use
natural language reasoning to match data across different DataFrames using
local models through Ollama for complete privacy and zero API costs.

Prerequisites:
- Install Ollama from https://ollama.ai
- Run: ollama pull qwen3:30b
- Ensure Ollama is running on http://localhost:11434
"""

from typing import Optional
import fenic as fc
from fenic._inference.litellm.litellm_batch_chat_completions_client import LiteLLMBatchChatCompletionsClient
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy


def create_local_llm_config() -> fc.SessionConfig:
    """Create session config with local LLM via LiteLLM."""
    return fc.SessionConfig(
        app_name="semantic_joins_local",
        semantic=fc.SemanticConfig(
            language_models={
                "local_qwen": fc.LiteLLMLanguageModel(
                    model_name="qwen3:30b",
                    rpm=100,
                    tpm=10000,
                    api_base="http://localhost:11434"
                )
            },
            default_language_model="local_qwen"
        )
    )


def setup_local_llm_client() -> LiteLLMBatchChatCompletionsClient:
    """Set up the local LLM client for semantic operations."""
    rate_limit_strategy = UnifiedTokenRateLimitStrategy(rpm=100, tpm=10000)

    return LiteLLMBatchChatCompletionsClient(
        rate_limit_strategy=rate_limit_strategy,
        model="ollama/qwen3:30b",
        queue_size=50,
        max_backoffs=3,
        api_base="http://localhost:11434"
    )


def main():
    """Demonstrate semantic join capabilities using local LLM reasoning."""
    print("🏠 Semantic Joins Example - Local LLM Edition")
    print("=" * 50)
    print("Demonstrating LLM-powered reasoning joins with local models")
    print("💰 Cost: $0.00 | 🔒 Privacy: 100% Local | 🌐 Internet: Not Required")
    print()

    # Configure session with local LLM
    config = create_local_llm_config()
    session = fc.Session.get_or_create(config)

    # Set up local LLM client
    local_llm_client = setup_local_llm_client()
    print(f"✅ Local LLM configured: ollama/qwen3:30b")
    print(f"📍 Ollama endpoint: http://localhost:11434")
    print()

    # Sample user profiles data
    users_data = [
        {
            "user_id": "user_001",
            "name": "Sarah",
            "interests": "I love cooking Italian food and trying new pasta recipes"
        },
        {
            "user_id": "user_002",
            "name": "Mike",
            "interests": "I enjoy working on cars and fixing engines in my spare time"
        },
        {
            "user_id": "user_003",
            "name": "Emily",
            "interests": "Gardening is my passion, especially growing vegetables and flowers"
        },
        {
            "user_id": "user_004",
            "name": "David",
            "interests": "I'm interested in learning about car maintenance and automotive repair"
        }
    ]

    # Sample content/articles data
    articles_data = [
        {
            "article_id": "art_001",
            "title": "Cooking Pasta Recipes",
            "description": "Delicious pasta recipes including spaghetti carbonara and fettuccine alfredo"
        },
        {
            "article_id": "art_002",
            "title": "Car Engine Maintenance",
            "description": "Essential guide to automobile engine care and troubleshooting"
        },
        {
            "article_id": "art_003",
            "title": "Gardening for Beginners",
            "description": "Start your garden with basic techniques for growing vegetables and flowers"
        },
        {
            "article_id": "art_004",
            "title": "Advanced Automotive Repair",
            "description": "Comprehensive automotive repair instructions for experienced mechanics"
        }
    ]

    # Create DataFrames
    users_df = session.create_dataframe(users_data)
    articles_df = session.create_dataframe(articles_data)

    print("👥 User Profiles:")
    users_df.select("name", "interests").show()
    print()

    print("📚 Available Articles:")
    articles_df.select("title", "description").show()
    print()

    # Step 1: Semantic join to match users with relevant articles
    print("🔍 Step 1: Matching users to relevant articles using local LLM reasoning...")
    print("-" * 70)

    # Use semantic join to match users with articles based on their interests
    print("🤖 Running semantic join with local model (this may take a moment)...")

    user_article_matches = users_df.semantic.join(
        articles_df,
        predicate=(
            "A person with interests '{{left_on}}' would be interested in reading about '{{right_on}}'"
        ),
        left_on=fc.col("interests"),
        right_on=fc.col("description")
    )

    print("✅ User-Article Matches (Powered by Local LLM):")
    user_article_matches.select(
        "name",
        "interests",
        "title",
        "description"
    ).show()
    print()

    # Step 2: Product recommendation system using semantic joins
    print("🛍️ Step 2: Product recommendation system using local LLM...")
    print("-" * 50)

    # Sample customer purchase history
    purchases_data = [
        {
            "customer_id": "cust_001",
            "customer_name": "Alice",
            "purchased_product": "Professional DSLR Camera"
        },
        {
            "customer_id": "cust_002",
            "customer_name": "Bob",
            "purchased_product": "Gaming Laptop"
        },
        {
            "customer_id": "cust_003",
            "customer_name": "Carol",
            "purchased_product": "Yoga Mat"
        },
        {
            "customer_id": "cust_004",
            "customer_name": "Dan",
            "purchased_product": "Coffee Maker"
        }
    ]

    # Sample product catalog for recommendations
    products_data = [
        {
            "product_id": "prod_001",
            "product_name": "Camera Lens Kit",
            "category": "Photography"
        },
        {
            "product_id": "prod_002",
            "product_name": "Tripod Stand",
            "category": "Photography"
        },
        {
            "product_id": "prod_003",
            "product_name": "Gaming Mouse",
            "category": "Gaming"
        },
        {
            "product_id": "prod_004",
            "product_name": "Mechanical Keyboard",
            "category": "Gaming"
        },
        {
            "product_id": "prod_005",
            "product_name": "Yoga Blocks",
            "category": "Fitness"
        },
        {
            "product_id": "prod_006",
            "product_name": "Exercise Resistance Bands",
            "category": "Fitness"
        },
        {
            "product_id": "prod_007",
            "product_name": "Coffee Beans Premium Blend",
            "category": "Food & Beverage"
        },
        {
            "product_id": "prod_008",
            "product_name": "French Press",
            "category": "Food & Beverage"
        }
    ]

    # Create DataFrames
    purchases_df = session.create_dataframe(purchases_data)
    products_df = session.create_dataframe(products_data)

    print("🛒 Customer Purchase History:")
    purchases_df.select("customer_name", "purchased_product").show()
    print()

    print("📦 Available Products for Recommendation:")
    products_df.select("product_name", "category").show()
    print()

    # Use semantic join for product recommendations
    print("🤖 Generating recommendations with local LLM...")

    recommendations = purchases_df.semantic.join(
        products_df,
        predicate=(
            "A customer who bought '{{left_on}}' would also be interested in '{{right_on}}'"
        ),
        left_on=fc.col("purchased_product"),
        right_on=fc.col("product_name")
    )

    print("🎯 Product Recommendations (Generated Locally):")
    recommendations.select(
        "customer_name",
        "purchased_product",
        "product_name",
        "category"
    ).show()
    print()

    # Show metrics
    metrics = local_llm_client.get_metrics()
    print("📊 Local LLM Processing Metrics:")
    print(f"  🔢 Total Requests: {metrics.num_requests}")
    print(f"  📝 Input Tokens: {metrics.num_uncached_input_tokens:,}")
    print(f"  📤 Output Tokens: {metrics.num_output_tokens:,}")
    print(f"  💰 Total Cost: ${metrics.cost:.2f} (FREE!)")
    print(f"  🏠 Model: ollama/qwen3:30b (Local)")
    print()

    print("🎉 Benefits of Local LLM Semantic Joins:")
    print("  🔒 Complete privacy - no data sent to external APIs")
    print("  💰 Zero ongoing costs - unlimited usage")
    print("  🚀 No network dependency - works offline")
    print("  ⚡ Consistent performance - no rate limits")
    print("  🔧 Full control - customize models and parameters")

    # Clean up
    session.stop()
    print("\n✅ Local LLM semantic joins completed successfully!")


if __name__ == "__main__":
    main()