"""Meeting transcript processing example using fenic semantic operations with Local LLMs.

This example demonstrates how to work with transcripts in fenic using its native
transcript processing capabilities, including format detection, parsing, and semantic
extraction of structured information from unstructured meeting content using
local models for complete privacy and zero API costs.

Prerequisites:
- Install Ollama from https://ollama.ai
- Run: ollama pull qwen3:30b
- Ensure Ollama is running on http://localhost:11434
"""

from typing import List, Optional
from pydantic import BaseModel, Field
import fenic as fc
from fenic._inference.litellm.litellm_batch_chat_completions_client import LiteLLMBatchChatCompletionsClient
from fenic._inference.rate_limit_strategy import UnifiedTokenRateLimitStrategy


def create_local_llm_config() -> fc.SessionConfig:
    """Create session config with local LLM via LiteLLM."""
    return fc.SessionConfig(
        app_name="meeting_transcript_processing_local",
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


# Engineering Architecture Review Transcript
architecture_review_transcript = """Sarah (00:02:15)
  Alright, so we're here to discuss the user service redesign. Um, Mike can you walk us through the current bottlenecks?

Mike (00:02:28)
  Sure, so right now we're seeing about 2-second response times on the /users endpoint during peak hours. The main issue is we're hitting the PostgreSQL database directly for every request.

David (00:02:45)
  Right, and I think we discussed adding Redis caching last sprint but didn't get to it.

Sarah (00:03:01)
  OK so action item for Mike - investigate Redis implementation. What about the authentication service dependency?

Mike (00:03:18)
  Yeah that's another bottleneck. Every user request has to validate the JWT token with the auth service. We could cache those validations too.

David (00:03:35)
  Actually, we had that incident last week where auth service went down and took out the whole user flow. Incident #INC-2024-007.

Sarah (00:03:48)
  Good point. So we need resilience there. Mike, can you also look into circuit breaker patterns? I'm thinking we implement that by end of Q1.

Mike (00:04:05)
  Yep, I'll research both Redis caching and circuit breakers. Should have a design doc ready by next Friday.

Sarah (00:04:15)
  Perfect. David, anything else on the database side?

David (00:04:22)
  We should consider read replicas too. I've been seeing high CPU on the primary during reports generation.

Sarah (00:04:35)
  OK, let's add that to the backlog. Decision: we're moving forward with the caching and circuit breaker approach for user service optimization."""

# Incident Post-Mortem Transcript
incident_postmortem_transcript = """Alex (00:01:05)
  OK everyone, this is the post-mortem for yesterday's outage. Incident #INC-2024-012. We had approximately 45 minutes of downtime starting at 14:30 UTC.

Jordan (00:01:20)
  The root cause was the payment service running out of memory. We saw heap size spike to 8GB before the JVM crashed.

Sam (00:01:35)
  Yeah, I was monitoring the dashboards. CPU was normal but memory kept climbing. No garbage collection could keep up.

Alex (00:01:48)
  What triggered it? Any recent deployments?

Jordan (00:01:55)
  We deployed the new batch processing feature Tuesday morning. I think there's a memory leak in the transaction processing loop.

Sam (00:02:10)
  Action item for me - I'll review the batch processing code and look for leaked objects or unclosed resources.

Alex (00:02:20)
  Good. Jordan, can you increase the heap size as a temporary mitigation? Maybe 12GB instead of 8?

Jordan (00:02:30)
  Already done. I bumped it to 12GB and added memory alerts at 80% usage.

Alex (00:02:40)
  Perfect. Sam, when can you have the code review done?

Sam (00:02:45)
  I'll have findings by tomorrow EOD. If it's a simple fix, we can hotfix Friday.

Alex (00:02:55)
  Decision: temporary mitigation in place, code review by Thursday, hotfix target Friday if feasible."""

# Sprint Planning Transcript
sprint_planning_transcript = """Emma (00:00:30)
  Alright team, let's plan Sprint 23. We have 40 story points available this sprint. What's our priorities?

Ryan (00:00:45)
  The user authentication refactor is our biggest priority. That's been blocking the mobile team for two weeks.

Lisa (00:00:58)
  Right, and we estimated that at 13 story points. We also need to address the API rate limiting issues.

Emma (00:01:12)
  Good point. Ryan, can you take the auth refactor? And Lisa, would you handle the rate limiting?

Ryan (00:01:20)
  Yeah, I can do the auth work. I'll need to coordinate with the mobile team on the JWT token format changes.

Lisa (00:01:35)
  Sure, I'll take rate limiting. I'm thinking we implement token bucket algorithm with Redis backend.

Emma (00:01:50)
  Perfect. What about the database migration for the user profiles table?

Ryan (00:02:05)
  That's risky. We're adding three new columns and need to backfill data for 2 million users.

Lisa (00:02:18)
  Action item - let's create a migration plan with zero-downtime strategy. I can draft that by Wednesday.

Emma (00:02:35)
  Great. Decision: Sprint 23 priorities are auth refactor, rate limiting, and migration planning. Ryan and Lisa are the leads."""


def main():
    """Process meeting transcripts to extract insights using local LLM semantic operations."""
    print("🏠 Meeting Transcript Processing - Local LLM Edition")
    print("=" * 60)
    print("🔒 Complete Privacy | 💰 Zero API Costs | 🌐 No Internet Required")
    print()

    # Configure session with local LLM
    config = create_local_llm_config()
    fc.configure_logging()
    session = fc.Session.get_or_create(config)

    # Set up local LLM client
    local_llm_client = setup_local_llm_client()
    print(f"✅ Local LLM configured: ollama/qwen3:30b")
    print(f"📍 Ollama endpoint: http://localhost:11434")
    print()

    transcripts_data = [
        {
            "meeting_id": "ARCH-2024-001",
            "meeting_type": "Architecture Review",
            "transcript": architecture_review_transcript,
        },
        {
            "meeting_id": "INC-2024-012",
            "meeting_type": "Incident Post-Mortem",
            "transcript": incident_postmortem_transcript,
        },
        {
            "meeting_id": "SPRINT-23",
            "meeting_type": "Sprint Planning",
            "transcript": sprint_planning_transcript,
        },
    ]

    transcripts_df = session.create_dataframe(transcripts_data)

    print("📝 Meeting transcripts loaded:")
    transcripts_df.select(fc.col("meeting_id"), fc.col("meeting_type")).show()

    # Step 1: Use fenic's native transcript processing
    print("\n🔧 Step 1: Native Transcript Parsing")
    print("=" * 50)

    # Parse transcripts into structured format
    # generic is a commonly found format for transcripts that follows the format:
    # speaker (00:00:00)
    # content
    parsed_transcripts_df = transcripts_df.with_column(
        "structured_transcript",
        fc.text.parse_transcript(fc.col("transcript"), "generic"),
    )

    print("\n📊 Parsed transcript structure sample:")
    sample_parsed = parsed_transcripts_df.select(
        fc.col("meeting_id"), fc.col("structured_transcript")
    ).limit(1)
    sample_parsed.show()

    # Step 2: Extract key information with local LLM
    print("\n🤖 Step 2: Semantic Information Extraction (Local LLM)")
    print("=" * 60)
    print("🔬 Analyzing meeting content with local model...")

    # Define Pydantic models for extraction
    class ActionItem(BaseModel):
        """Action item from the meeting."""
        assignee: str = Field(description="Person assigned to this action")
        task: str = Field(description="Description of what needs to be done")
        deadline: str = Field(description="When this should be completed, if mentioned")

    class MeetingInsights(BaseModel):
        """Key insights extracted from meeting transcript."""
        main_topics: str = Field(description="Primary topics discussed")
        key_decisions: str = Field(description="Important decisions made")
        risks_identified: str = Field(description="Risks or concerns mentioned")
        technical_details: str = Field(description="Technical specifications or details")

    # Extract action items and insights
    enriched_df = parsed_transcripts_df.select(
        fc.col("meeting_id"),
        fc.col("meeting_type"),
        fc.col("transcript"),
        # Extract action items
        fc.semantic.extract(
            fc.col("transcript"),
            List[ActionItem],
            max_output_tokens=512,
            instruction="Extract all action items mentioned in this meeting transcript."
        ).alias("action_items"),
        # Extract meeting insights
        fc.semantic.extract(
            fc.col("transcript"),
            MeetingInsights,
            max_output_tokens=512,
            instruction="Analyze this meeting and extract key insights."
        ).alias("insights"),
        # Classify meeting priority/urgency
        fc.semantic.classify(
            fc.col("transcript"),
            ["low", "medium", "high", "critical"],
            instruction="Based on the content and decisions, classify the urgency level of this meeting."
        ).alias("urgency_level")
    ).unnest("insights")

    print("✅ Extraction complete!")

    # Display results
    print("\n📋 Meeting Insights:")
    enriched_df.select(
        "meeting_id",
        "meeting_type",
        "main_topics",
        "key_decisions",
        "urgency_level"
    ).show()

    print("\n⚠️ Risks and Technical Details:")
    enriched_df.select(
        "meeting_id",
        "risks_identified",
        "technical_details"
    ).show()

    # Step 3: Action Item Analysis
    print("\n✅ Step 3: Action Item Analysis")
    print("=" * 40)

    # Explode action items for detailed analysis
    action_items_df = enriched_df.select(
        fc.col("meeting_id"),
        fc.col("meeting_type"),
        fc.explode("action_items").alias("action_item")
    ).unnest("action_item")

    print("📝 Extracted Action Items:")
    action_items_df.select(
        "meeting_id",
        "assignee",
        "task",
        "deadline"
    ).show()

    # Step 4: Meeting Summary Generation
    print("\n📊 Step 4: Automated Meeting Summaries (Local LLM)")
    print("=" * 55)

    class MeetingSummary(BaseModel):
        """Executive summary of the meeting."""
        executive_summary: str = Field(description="Brief executive summary for leadership")
        next_steps: str = Field(description="Clear next steps and follow-ups")
        blockers: str = Field(description="Any blockers or dependencies identified")

    summaries_df = enriched_df.select(
        fc.col("meeting_id"),
        fc.col("meeting_type"),
        fc.semantic.extract(
            fc.col("transcript"),
            MeetingSummary,
            max_output_tokens=300,
            instruction="Create a concise executive summary of this meeting."
        ).alias("summary")
    ).unnest("summary")

    print("📑 Meeting Summaries (Generated Locally):")
    summaries_df.select(
        "meeting_id",
        "executive_summary",
        "next_steps",
        "blockers"
    ).show()

    # Step 5: Cross-Meeting Analysis
    print("\n🔍 Step 5: Cross-Meeting Pattern Analysis")
    print("=" * 45)

    # Analyze patterns across meetings
    all_action_items = action_items_df.group_by("assignee").agg(
        fc.count("*").alias("total_action_items"),
        fc.text.concat_ws("; ", fc.col("task")).alias("all_tasks")
    )

    print("👥 Action Item Distribution by Person:")
    all_action_items.show()

    # Final metrics and summary
    print("\n📊 Processing Summary:")
    print("=" * 30)

    total_meetings = transcripts_df.count()
    total_action_items = action_items_df.count()

    print(f"  📝 Meetings Processed: {total_meetings}")
    print(f"  ✅ Action Items Extracted: {total_action_items}")
    print(f"  🎯 Average Items per Meeting: {total_action_items / total_meetings:.1f}")

    # Show performance metrics
    metrics = local_llm_client.get_metrics()
    print(f"\n📊 Local LLM Processing Metrics:")
    print(f"  🔢 Total Requests: {metrics.num_requests}")
    print(f"  📝 Input Tokens: {metrics.num_uncached_input_tokens:,}")
    print(f"  📤 Output Tokens: {metrics.num_output_tokens:,}")
    print(f"  💰 Total Cost: ${metrics.cost:.2f} (FREE!)")
    print(f"  🏠 Model: ollama/qwen3:30b (Local)")

    print(f"\n🎉 Benefits of Local LLM Meeting Processing:")
    print(f"  🔒 Complete privacy - sensitive meeting content stays local")
    print(f"  💰 Zero ongoing costs - process unlimited transcripts")
    print(f"  🚀 No network dependency - works in secure environments")
    print(f"  ⚡ No rate limits - batch process large meetings instantly")
    print(f"  🔧 Full control - customize extraction for your workflows")
    print(f"  📊 Consistent results - same model, same outputs")

    # Clean up
    session.stop()
    print(f"\n✅ Local LLM meeting transcript processing completed successfully!")


if __name__ == "__main__":
    main()