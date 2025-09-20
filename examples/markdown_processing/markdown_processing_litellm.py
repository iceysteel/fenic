"""Markdown Processing with Fenic and Local LLMs.

This example demonstrates how to process academic papers and structured markdown documents
using fenic's specialized markdown functions combined with local LLM semantic analysis
for complete privacy and zero API costs.

Features demonstrated:
1. Markdown structure analysis with local LLM insights
2. Automatic section summarization using local models
3. Content extraction and semantic analysis
4. Citation pattern recognition
5. Document quality assessment

Prerequisites:
- Install Ollama from https://ollama.ai
- Run: ollama pull qwen3:30b
- Ensure Ollama is running on http://localhost:11434
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import fenic as fc
from fenic._inference.litellm.litellm_batch_chat_completions_client import LiteLLMBatchChatCompletionsClient
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy


def create_local_llm_config() -> fc.SessionConfig:
    """Create session config with local LLM via LiteLLM."""
    return fc.SessionConfig(
        app_name="markdown_processing_local",
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


# Sample academic paper content (simplified version)
SAMPLE_ACADEMIC_PAPER = """# Attention Is All You Need

## Abstract

We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

## 1. Introduction

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.

## 2. Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU, ByteNet and ConvS2S, all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.

## 3. Model Architecture

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

### 3.1 Encoder and Decoder Stacks

**Encoder:** The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.

**Decoder:** The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack.

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

## 4. Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations to another sequence of equal length.

## 5. Training

This section describes the training regime for our models.

### 5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens.

## 6. Results

### 6.1 Machine Translation

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big)) achieves a new state-of-the-art BLEU score of 28.4, improving over the existing best results, including ensembles, by over 2 BLEU.

## 7. Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

## References

1. Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

2. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

3. Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N Dauphin. Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122, 2017.
"""


def main():
    """Process academic paper markdown content using fenic with local LLM analysis."""
    print("🏠 Markdown Processing with Local LLM - Academic Paper Analysis")
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

    # Create DataFrame with the paper content
    df = session.create_dataframe({
        "paper_title": ["Attention Is All You Need"],
        "content": [SAMPLE_ACADEMIC_PAPER]
    })

    # Cast content to MarkdownType to enable markdown-specific functions
    df = df.select(
        fc.col("paper_title"),
        fc.col("content").cast(fc.MarkdownType).alias("markdown")
    )

    print("📄 Academic Paper Loaded:")
    df.select(fc.col('paper_title')).show()

    # Step 1: Generate Table of Contents using markdown structure
    print("\n📚 Step 1: Extracting Document Structure")
    print("=" * 50)

    # Extract sections using markdown structure
    toc_df = df.select(
        fc.col("paper_title"),
        fc.markdown.generate_toc(fc.col("markdown")).alias("table_of_contents")
    )

    print("📑 Table of Contents:")
    toc_df.show()

    # Step 2: Extract sections into structured data
    print("\n🔍 Step 2: Section-by-Section Analysis with Local LLM")
    print("=" * 55)

    # Extract sections as separate rows using header chunks
    sections_df = df.select(
        fc.col("paper_title"),
        fc.markdown.extract_header_chunks(fc.col("markdown"), header_level=1).alias("sections")
    ).explode("sections").select(
        fc.col("paper_title"),
        fc.col("sections")
    )

    print("📊 Document Sections Extracted:")
    sections_df.show()

    # Step 3: Semantic analysis of each section with local LLM
    print("\n🤖 Step 3: Semantic Section Analysis (Local LLM)")
    print("=" * 50)

    class SectionAnalysis(BaseModel):
        """Analysis of an academic paper section."""
        main_concepts: str = Field(description="Key concepts introduced in this section")
        technical_complexity: str = Field(description="Low, Medium, or High complexity level")
        contribution_type: str = Field(description="Type of contribution (methodology, results, background, etc.)")
        key_insights: str = Field(description="Most important insights from this section")

    print("🔬 Analyzing sections with local model...")

    analyzed_sections = sections_df.unnest("sections").filter(
        fc.col("level") <= 2  # Focus on main sections
    ).select(
        fc.col("heading"),
        fc.col("content"),
        fc.semantic.extract(
            fc.col("content"),
            SectionAnalysis,
            max_output_tokens=300
        ).alias("analysis")
    ).unnest("analysis")

    print("✅ Section Analysis Complete!")
    print("\n📊 Section Analysis Results:")
    analyzed_sections.select(
        "heading",
        "main_concepts",
        "technical_complexity",
        "contribution_type"
    ).show()

    # Step 4: Reference analysis
    print("\n📚 Step 4: Reference Analysis")
    print("=" * 35)

    # Extract references section
    references_df = sections_df.unnest("sections").filter(
        fc.col("heading").str.contains("References")
    ).select(
        fc.col("content"),
        # Count citations
        fc.text.count_pattern(fc.col("content"), r"\\d+\\.").alias("citation_count"),
        # Extract publication years using regex
        fc.text.extract_pattern(fc.col("content"), r"\\b(19|20)\\d{2}\\b", "all").alias("years")
    )

    print("📖 Reference Analysis:")
    references_df.show()

    # Step 5: Overall document assessment with local LLM
    print("\n📝 Step 5: Overall Document Assessment (Local LLM)")
    print("=" * 55)

    class DocumentAssessment(BaseModel):
        """Overall assessment of the academic paper."""
        research_field: str = Field(description="Primary research field or domain")
        innovation_level: str = Field(description="Level of innovation (incremental, significant, breakthrough)")
        clarity_score: str = Field(description="Writing clarity (poor, good, excellent)")
        technical_depth: str = Field(description="Technical depth (introductory, intermediate, advanced)")
        potential_impact: str = Field(description="Potential impact on the field")

    # Assess the full document
    document_assessment = df.select(
        fc.col("paper_title"),
        fc.semantic.extract(
            fc.col("markdown").cast(fc.StringType),
            DocumentAssessment,
            max_output_tokens=200
        ).alias("assessment")
    ).unnest("assessment")

    print("🎯 Document Assessment (Generated Locally):")
    document_assessment.select(
        "paper_title",
        "research_field",
        "innovation_level",
        "clarity_score",
        "technical_depth",
        "potential_impact"
    ).show()

    # Step 6: Generate summary insights
    print("\n📊 Step 6: Summary and Insights")
    print("=" * 35)

    # Count sections by complexity
    complexity_summary = analyzed_sections.group_by("technical_complexity").agg(
        fc.count("*").alias("section_count")
    ).order_by("technical_complexity")

    print("🧮 Technical Complexity Distribution:")
    complexity_summary.show()

    # Summary statistics
    total_sections = sections_df.count()
    analyzed_count = analyzed_sections.count()

    print(f"\n📈 Processing Summary:")
    print(f"  📄 Total Sections: {total_sections}")
    print(f"  🔬 Analyzed Sections: {analyzed_count}")
    print(f"  🧠 Analysis Coverage: {(analyzed_count/total_sections)*100:.1f}%")

    # Show performance metrics
    metrics = local_llm_client.get_metrics()
    print(f"\n📊 Local LLM Processing Metrics:")
    print(f"  🔢 Total Requests: {metrics.num_requests}")
    print(f"  📝 Input Tokens: {metrics.num_uncached_input_tokens:,}")
    print(f"  📤 Output Tokens: {metrics.num_output_tokens:,}")
    print(f"  💰 Total Cost: ${metrics.cost:.2f} (FREE!)")
    print(f"  🏠 Model: ollama/qwen3:30b (Local)")

    print(f"\n🎉 Benefits of Local LLM Document Processing:")
    print(f"  🔒 Complete privacy - sensitive documents stay local")
    print(f"  💰 Zero ongoing costs - process unlimited papers")
    print(f"  🚀 No network dependency - works in secure environments")
    print(f"  ⚡ No rate limits - batch process large document sets")
    print(f"  🔧 Full control - customize analysis for your domain")
    print(f"  📊 Consistent results - reproducible document insights")

    # Clean up
    session.stop()
    print(f"\n✅ Local LLM markdown processing completed successfully!")


if __name__ == "__main__":
    main()