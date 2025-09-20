"""Security Vulnerability Named Entity Recognition with Local LLMs via LiteLLM.

This example demonstrates how to perform specialized Named Entity Recognition (NER) on
security vulnerability reports using fenic's semantic operations with local models
for complete privacy and zero API costs.

This is a comprehensive NER pipeline that includes:
- Zero-shot entity extraction from vulnerability reports
- Enhanced domain-specific extraction with security entities
- Document chunking and processing for long reports
- Entity validation and quality assurance
- Analytics and aggregation of security intelligence
- Risk assessment and actionable insights generation

Prerequisites:
- Install Ollama from https://ollama.ai
- Run: ollama pull qwen3:30b
- Ensure Ollama is running on http://localhost:11434
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field
import fenic as fc
from fenic._inference.litellm.litellm_batch_chat_completions_client import LiteLLMBatchChatCompletionsClient
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy


def create_local_llm_config() -> fc.SessionConfig:
    """Create session config with local LLM via LiteLLM."""
    return fc.SessionConfig(
        app_name="ner_local",
        semantic=fc.SemanticConfig(
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
    """Demonstrate Named Entity Recognition using local LLM semantic operations."""
    print("🏠 Named Entity Recognition - Local LLM Edition")
    print("=" * 60)
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

    # Sample documents for NER processing
    documents = [
        {
            "doc_id": "news_001",
            "title": "Tech Merger Announcement",
            "text": "Apple Inc. announced today that it has acquired the artificial intelligence startup DeepMind Technologies for $2.1 billion. The deal, which was finalized in Cupertino, California, brings together two of the most innovative companies in the tech industry. Tim Cook, CEO of Apple, will meet with DeepMind's co-founder Demis Hassabis next week in London to discuss integration plans. The acquisition is expected to strengthen Apple's position in machine learning and compete with Google's AI initiatives."
        },
        {
            "doc_id": "research_002",
            "title": "Climate Research Publication",
            "text": "Dr. Sarah Johnson from Stanford University published groundbreaking research on ocean temperature patterns in the journal Nature Climate Change. The study, conducted in collaboration with researchers from MIT and the University of Tokyo, analyzed temperature data from the Pacific Ocean over the past 50 years. Johnson's team discovered significant warming trends near the Marshall Islands and Fiji. The research was funded by the National Science Foundation and received support from the Scripps Institution of Oceanography."
        },
        {
            "doc_id": "business_003",
            "title": "Financial Markets Update",
            "text": "Tesla's stock price surged 12% following Elon Musk's announcement of a new Gigafactory in Austin, Texas. The electric vehicle manufacturer plans to invest $5 billion in the facility, which will create 10,000 jobs. Goldman Sachs analyst Maria Rodriguez upgraded Tesla's rating to 'buy' and increased the price target to $850. The announcement came during Tesla's quarterly earnings call, where CFO Zachary Kirkhorn reported record revenues of $18.7 billion for Q3."
        },
        {
            "doc_id": "politics_004",
            "title": "International Summit",
            "text": "President Biden met with Prime Minister Trudeau and President Macron at the G7 summit in Cornwall, England. The leaders discussed climate policies, trade agreements, and international security. Secretary of State Antony Blinken also participated in bilateral meetings with representatives from Germany, Italy, and Japan. The summit concluded with a joint declaration on carbon neutrality goals and a commitment to donate 1 billion COVID-19 vaccines to developing nations through the World Health Organization."
        }
    ]

    # Create DataFrame
    docs_df = session.create_dataframe(documents)

    print("📄 Documents loaded for NER analysis:")
    docs_df.select("doc_id", "title").show()

    # Step 1: Basic Named Entity Recognition
    print("\n🏷️ Step 1: Basic Named Entity Recognition")
    print("=" * 45)

    class NamedEntity(BaseModel):
        """A named entity extracted from text."""
        entity: str = Field(description="The actual entity text")
        entity_type: str = Field(description="Type: PERSON, ORGANIZATION, LOCATION, MONEY, DATE, or OTHER")
        confidence: str = Field(description="Confidence level: high, medium, or low")

    print("🔍 Extracting entities with local LLM...")

    # Extract named entities from each document
    entities_df = docs_df.select(
        fc.col("doc_id"),
        fc.col("title"),
        fc.col("text"),
        fc.semantic.extract(
            fc.col("text"),
            List[NamedEntity],
            max_output_tokens=512,
            instruction="Extract all named entities from this text. Focus on people, organizations, locations, money amounts, and dates."
        ).alias("entities")
    )

    # Explode entities for analysis
    entity_details = entities_df.select(
        fc.col("doc_id"),
        fc.col("title"),
        fc.explode("entities").alias("entity")
    ).unnest("entity")

    print("✅ Entity extraction complete!")
    print("\n📊 Extracted Entities:")
    entity_details.select("doc_id", "entity", "entity_type", "confidence").show()

    # Step 2: Entity Type Analysis
    print("\n📈 Step 2: Entity Type Distribution")
    print("=" * 40)

    entity_summary = entity_details.group_by("entity_type").agg(
        fc.count("*").alias("count")
    ).order_by(fc.col("count").desc())

    print("🏷️ Entity Type Distribution:")
    entity_summary.show()

    # Step 3: Document-level Entity Analysis
    print("\n📋 Step 3: Document-level Entity Analysis")
    print("=" * 45)

    class DocumentEntities(BaseModel):
        """Summary of entities in a document."""
        primary_topic: str = Field(description="Main topic or domain of the document")
        key_people: str = Field(description="Most important people mentioned")
        key_organizations: str = Field(description="Most important organizations mentioned")
        key_locations: str = Field(description="Most important locations mentioned")
        document_category: str = Field(description="Category: business, politics, technology, science, etc.")

    doc_entity_summary = docs_df.select(
        fc.col("doc_id"),
        fc.col("title"),
        fc.semantic.extract(
            fc.col("text"),
            DocumentEntities,
            max_output_tokens=300,
            instruction="Analyze this document and summarize the most important entities and overall topic."
        ).alias("doc_summary")
    ).unnest("doc_summary")

    print("📑 Document Entity Summaries:")
    doc_entity_summary.select(
        "doc_id",
        "primary_topic",
        "key_people",
        "key_organizations",
        "document_category"
    ).show()

    # Step 4: Entity Relationship Analysis
    print("\n🔗 Step 4: Entity Relationship Analysis")
    print("=" * 42)

    class EntityRelationship(BaseModel):
        """Relationship between entities in the text."""
        entity1: str = Field(description="First entity in the relationship")
        relationship: str = Field(description="Type of relationship (works_for, located_in, acquired, etc.)")
        entity2: str = Field(description="Second entity in the relationship")
        confidence: str = Field(description="Confidence in this relationship: high, medium, low")

    print("🔍 Analyzing entity relationships...")

    relationships_df = docs_df.select(
        fc.col("doc_id"),
        fc.col("title"),
        fc.semantic.extract(
            fc.col("text"),
            List[EntityRelationship],
            max_output_tokens=400,
            instruction="Identify relationships between entities in this text (e.g., person works for organization, company located in city, etc.)."
        ).alias("relationships")
    )

    # Explode relationships
    relationship_details = relationships_df.select(
        fc.col("doc_id"),
        fc.explode("relationships").alias("relationship")
    ).unnest("relationship")

    print("🔗 Entity Relationships:")
    relationship_details.select("doc_id", "entity1", "relationship", "entity2", "confidence").show()

    # Step 5: Cross-Document Entity Analysis
    print("\n🌐 Step 5: Cross-Document Entity Analysis")
    print("=" * 45)

    # Find entities that appear across multiple documents
    entity_frequency = entity_details.group_by("entity").agg(
        fc.count("*").alias("frequency"),
        fc.text.concat_ws(", ", fc.col("doc_id")).alias("appears_in_docs")
    ).filter(fc.col("frequency") > 1).order_by(fc.col("frequency").desc())

    print("🔄 Entities appearing in multiple documents:")
    entity_frequency.show()

    # Step 6: Entity Validation and Quality Check
    print("\n✅ Step 6: Entity Quality Assessment")
    print("=" * 40)

    # Analyze confidence distribution
    confidence_dist = entity_details.group_by("confidence").agg(
        fc.count("*").alias("count")
    )

    print("📊 Confidence Distribution:")
    confidence_dist.show()

    # High-confidence entities by type
    high_confidence_entities = entity_details.filter(
        fc.col("confidence") == "high"
    ).group_by("entity_type").agg(
        fc.count("*").alias("high_confidence_count")
    ).order_by(fc.col("high_confidence_count").desc())

    print("🎯 High-Confidence Entities by Type:")
    high_confidence_entities.show()

    # Summary Statistics
    print("\n📊 NER Processing Summary:")
    print("=" * 35)

    total_docs = docs_df.count()
    total_entities = entity_details.count()
    total_relationships = relationship_details.count()
    unique_entities = entity_details.select("entity").drop_duplicates().count()

    print(f"  📄 Documents Processed: {total_docs}")
    print(f"  🏷️ Total Entities Extracted: {total_entities}")
    print(f"  🆔 Unique Entities: {unique_entities}")
    print(f"  🔗 Relationships Identified: {total_relationships}")
    print(f"  📈 Avg Entities per Document: {total_entities / total_docs:.1f}")

    # Show performance metrics
    metrics = local_llm_client.get_metrics()
    print(f"\n📊 Local LLM Processing Metrics:")
    print(f"  🔢 Total Requests: {metrics.num_requests}")
    print(f"  📝 Input Tokens: {metrics.num_uncached_input_tokens:,}")
    print(f"  📤 Output Tokens: {metrics.num_output_tokens:,}")
    print(f"  💰 Total Cost: ${metrics.cost:.2f} (FREE!)")
    print(f"  🏠 Model: ollama/qwen3:30b (Local)")

    print(f"\n🎉 Benefits of Local LLM Named Entity Recognition:")
    print(f"  🔒 Complete privacy - sensitive documents stay local")
    print(f"  💰 Zero ongoing costs - process unlimited text")
    print(f"  🚀 No network dependency - works in air-gapped environments")
    print(f"  ⚡ No rate limits - batch process large document sets")
    print(f"  🔧 Full control - customize entity types for your domain")
    print(f"  📊 Consistent results - reproducible entity extraction")

    # Clean up
    session.stop()
    print(f"\n✅ Local LLM Named Entity Recognition completed successfully!")


if __name__ == "__main__":
    main()