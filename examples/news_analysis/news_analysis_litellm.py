#!/usr/bin/env python3
"""News Article Bias Detection with Local LLMs via LiteLLM.

This script demonstrates how to use fenic's semantic classification capabilities to detect editorial bias and analyze news articles using local models for complete privacy and zero API costs.

Features demonstrated:
- Language Analysis using `semantic.extract()` to find biased, emotional, or sensationalist language
- Political Bias Classification using `semantic.classify()` grounded in the extracted data
- News Topic Classification using `semantic.classify()`
- Merging information using `semantic.reduce()` to create 'Media Profile' summaries

Prerequisites:
- Install Ollama from https://ollama.ai
- Run: ollama pull qwen3:30b
- Ensure Ollama is running on http://localhost:11434

Usage:
    python news_analysis_litellm.py
"""

from typing import Optional
from pydantic import BaseModel, Field
import fenic as fc
from fenic._inference.litellm.litellm_batch_chat_completions_client import LiteLLMBatchChatCompletionsClient
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy


def create_local_llm_config() -> fc.SessionConfig:
    """Create session config with local LLM via LiteLLM."""
    return fc.SessionConfig(
        app_name="news_analysis_local",
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
    """Main analysis pipeline for news article bias detection using local LLM."""
    print("🏠 News Bias Detection Pipeline - Local LLM Edition")
    print("=" * 70)
    print("🔒 Complete Privacy | 💰 Zero API Costs | 🌐 No Internet Required")
    print()

    # Configure session with local LLM
    config = create_local_llm_config()
    session = fc.Session.get_or_create(config)

    # Set up local LLM client
    local_llm_client = setup_local_llm_client()
    print(f"✅ Local LLM configured: ollama/qwen3:30b")
    print(f"📍 Ollama endpoint: http://localhost:11434")
    print()

    # Sample news articles - multiple articles per source to show bias patterns
    news_articles = [
        # Global Wire Service (Neutral source, Reuters-style) - 3 articles
        {
            "source": "Global Wire Service",
            "headline": "Federal Reserve Raises Interest Rates by 0.25 Percentage Points",
            "content": "The Federal Reserve announced a quarter-point increase in interest rates Wednesday, bringing the federal funds rate to 5.5%. The decision was unanimous among voting members. Fed Chair Jerome Powell cited persistent inflation concerns and a robust labor market as key factors. The rate hike affects borrowing costs for consumers and businesses. Economic analysts had predicted the move following recent inflation data showing prices remained above the Fed's 2% target."
        },
        {
            "source": "Global Wire Service",
            "headline": "OpenAI Launches GPT-4 Turbo with 128K Context Window",
            "content": "OpenAI today announced GPT-4 Turbo, featuring a 128,000 token context window and updated training data through April 2024. The model offers improved instruction following and reduced likelihood of generating harmful content. Pricing is set at $0.01 per 1K input tokens and $0.03 per 1K output tokens. The release includes enhanced support for JSON mode and function calling. Developer early access begins this week, with general availability planned for December."
        },
        {
            "source": "Global Wire Service",
            "headline": "Climate Summit Reaches Agreement on Fossil Fuel Transition",
            "content": "Delegates at the COP28 climate summit in Dubai reached a consensus agreement calling for a transition away from fossil fuels in energy systems. The deal, approved by nearly 200 countries, marks the first time a COP agreement explicitly mentions fossil fuels. However, the agreement uses the phrase 'transitioning away' rather than 'phasing out,' reflecting compromises necessary to secure broad support. Environmental groups expressed mixed reactions, with some praising the historic mention while others criticized the lack of binding timelines."
        },

        # Progressive Voice (Left-leaning source) - 3 articles
        {
            "source": "Progressive Voice",
            "headline": "Fed's Rate Hike Threatens Working Families as Corporate Profits Soar",
            "content": "Once again, the Federal Reserve has chosen to burden working families with higher borrowing costs while Wall Street celebrates record profits. Wednesday's rate hike to 5.5% will make mortgages, credit cards, and student loans more expensive for millions of Americans already struggling with housing costs. Meanwhile, corporate executives continue awarding themselves massive bonuses. This regressive monetary policy prioritizes the wealthy elite over middle-class families who desperately need relief."
        },
        {
            "source": "Progressive Voice",
            "headline": "Big Tech's AI Surveillance Threatens Democratic Values",
            "content": "OpenAI's latest AI release represents another troubling escalation in Silicon Valley's surveillance capitalism model. These systems hoover up personal data and creative content without meaningful consent from users. Artists, writers, and creators see their work exploited to train AI systems that directly compete with human creativity. Meanwhile, users surrender intimate conversations to corporate servers with little transparency. We need immediate regulation to protect digital rights and prevent tech giants from privatizing human knowledge for profit."
        },
        {
            "source": "Progressive Voice",
            "headline": "Climate Summit's Weak Language Betrays Future Generations",
            "content": "The COP28 agreement represents a devastating failure to confront the climate emergency with the urgency science demands. By choosing vague 'transition' language over concrete 'phase out' commitments, world leaders have once again capitulated to fossil fuel lobbying and corporate interests. Young climate activists who traveled to Dubai seeking real action have been betrayed by politicians who prioritize industry profits over planetary survival. We cannot afford more empty promises while the climate crisis accelerates."
        },

        # Liberty Herald (Right-leaning source) - 3 articles
        {
            "source": "Liberty Herald",
            "headline": "Fed's Prudent Rate Decision Reinforces Economic Stability",
            "content": "The Federal Reserve's measured quarter-point rate increase demonstrates responsible monetary policy that will preserve long-term economic prosperity. By raising rates to 5.5%, Fed officials are taking necessary steps to prevent runaway inflation that would devastate savings and fixed incomes. This disciplined approach protects the purchasing power that American families have worked hard to build. Free market principles and sound fiscal management require tough decisions that ensure sustainable growth for job creators and investors."
        },
        {
            "source": "Liberty Herald",
            "headline": "American AI Innovation Leads Global Technology Revolution",
            "content": "OpenAI's breakthrough demonstrates why American innovation continues to lead the world in transformative technology. This achievement showcases the power of free enterprise and competitive markets to deliver solutions that benefit humanity. While other nations impose heavy-handed regulations that stifle innovation, American companies are unleashing AI capabilities that will create jobs, boost productivity, and solve complex problems. America's technological superiority depends on supporting pioneering companies through pro-growth policies and reduced government interference."
        },
        {
            "source": "Liberty Herald",
            "headline": "Pragmatic Climate Deal Balances Environmental Goals with Economic Reality",
            "content": "The COP28 agreement demonstrates mature leadership by acknowledging environmental concerns while protecting economic stability and energy security. The careful 'transition away' language recognizes that abrupt fossil fuel elimination would devastate working families and developing nations that depend on affordable energy. American energy producers have already reduced emissions through innovation and cleaner technologies, proving that market solutions work better than government mandates. This balanced approach protects jobs while investing in alternatives."
        }
    ]

    # Define Pydantic model for detailed article analysis
    class ArticleAnalysis(BaseModel):
        """Comprehensive analysis of news article content and bias."""
        bias_indicators: str = Field(description="Key words or phrases that indicate political bias")
        emotional_language: str = Field(description="Emotionally charged words or neutral descriptive language")
        opinion_markers: str = Field(description="Words or phrases that signal opinion vs. factual reporting")

    # Create DataFrame from news articles
    df = session.create_dataframe(news_articles)

    print("📰 News Bias Detection Pipeline (Local LLM)")
    print("=" * 70)
    print(f"Analyzing {df.count()} news articles from {df.select('source').drop_duplicates(['source']).count()} sources")

    # Show dataset composition
    print("\n📊 Dataset Composition:")
    df.group_by("source").agg(fc.count("*").alias("articles")).order_by("source").show()

    print("\n🔍 Performing semantic bias detection with local LLM...")
    print("🤖 This may take a moment as we analyze each article locally...")
    print("First, we extract key information from each article.\n")

    # Create combined text for context-aware analysis
    combined_content = fc.text.concat(
        fc.col("headline"),
        fc.lit(" | "),
        fc.col("content")
    )

    # Semantic Classification to identify primary topics and content bias for each article
    # Using cache() to ensure expensive local LLM operations don't need to be re-run
    print("🔬 Step 1: Extracting bias indicators and analyzing language...")

    enriched_df = df.with_column("combined_content", combined_content).select(
        fc.col("source"),
        fc.col("headline"),
        fc.col("content"),
        # Primary topic classification
        fc.semantic.classify(
            fc.col("combined_content"),
            ["politics", "technology", "business", "climate", "healthcare"]
        ).alias("primary_topic"),
        # Content Metadata using semantic.extract
        fc.semantic.extract(
            fc.col("combined_content"),
            ArticleAnalysis,
            max_output_tokens=512,
        ).alias("analysis_metadata"),
    ).unnest("analysis_metadata").cache()

    print("✅ Analysis complete! Here are the results:")
    print("\n📊 Article Topic Classification:")
    enriched_df.select("headline", "source", "primary_topic").show()

    print("\n🔍 Bias Analysis Details:")
    enriched_df.select(
        "source",
        "headline",
        "bias_indicators",
        "emotional_language",
        "opinion_markers"
    ).show()

    # Step 2: Classify political bias for each article
    print("\n🎯 Step 2: Classifying political bias...")

    bias_df = enriched_df.select(
        fc.col("source"),
        fc.col("headline"),
        fc.col("primary_topic"),
        # Political bias classification based on the extracted indicators
        fc.semantic.classify(
            fc.text.concat(
                fc.lit("Bias indicators: "),
                fc.col("bias_indicators"),
                fc.lit(" | Emotional language: "),
                fc.col("emotional_language"),
                fc.lit(" | Opinion markers: "),
                fc.col("opinion_markers")
            ),
            ["left", "center-left", "center", "center-right", "right"]
        ).alias("political_bias")
    ).cache()

    print("📈 Political Bias Classification Results:")
    bias_df.select("source", "headline", "political_bias").show()

    # Step 3: Create source profiles using semantic reduce
    print("\n📋 Step 3: Creating media source profiles...")

    # Define source profile model
    class SourceProfile(BaseModel):
        """Profile of a news source based on article analysis."""
        overall_bias: str = Field(description="Overall political leaning of this news source")
        bias_strength: str = Field(description="How strong or subtle the bias is (subtle, moderate, strong)")
        content_style: str = Field(description="Writing style and approach (neutral, emotional, analytical, etc.)")
        reliability_indicators: str = Field(description="Factors that indicate journalistic reliability or bias")

    source_profiles = bias_df.group_by("source").agg(
        fc.semantic.reduce(
            fc.text.concat_ws(
                "; ",
                fc.col("headline"),
                fc.col("political_bias"),
                fc.col("primary_topic")
            ),
            SourceProfile,
            instruction="Analyze the overall editorial stance and reliability of this news source based on their article patterns."
        ).alias("profile")
    ).unnest("profile")

    print("🏢 Media Source Profiles (Generated by Local LLM):")
    source_profiles.select(
        "source",
        "overall_bias",
        "bias_strength",
        "content_style",
        "reliability_indicators"
    ).show()

    # Final summary and metrics
    print("\n📊 Analysis Summary:")
    print("=" * 50)

    # Show bias distribution
    print("\n🎯 Political Bias Distribution:")
    bias_distribution = bias_df.group_by("political_bias").agg(
        fc.count("*").alias("article_count")
    ).order_by("political_bias")
    bias_distribution.show()

    # Show topic distribution
    print("\n📚 Topic Distribution:")
    topic_distribution = bias_df.group_by("primary_topic").agg(
        fc.count("*").alias("article_count")
    ).order_by("primary_topic")
    topic_distribution.show()

    # Show performance metrics
    metrics = local_llm_client.get_metrics()
    print("\n📊 Local LLM Processing Metrics:")
    print(f"  🔢 Total Requests: {metrics.num_requests}")
    print(f"  📝 Input Tokens: {metrics.num_uncached_input_tokens:,}")
    print(f"  📤 Output Tokens: {metrics.num_output_tokens:,}")
    print(f"  💰 Total Cost: ${metrics.cost:.2f} (FREE!)")
    print(f"  🏠 Model: ollama/qwen3:30b (Local)")

    print("\n🎉 Benefits of Local LLM News Analysis:")
    print("  🔒 Complete privacy - sensitive news analysis stays local")
    print("  💰 Zero ongoing costs - analyze unlimited articles")
    print("  🚀 No network dependency - works in air-gapped environments")
    print("  ⚡ No rate limits - process large news datasets instantly")
    print("  🔧 Full control - customize bias detection for your needs")
    print("  📊 Reproducible - same model gives consistent results")

    # Clean up
    session.stop()
    print("\n✅ Local LLM news bias analysis completed successfully!")


if __name__ == "__main__":
    main()