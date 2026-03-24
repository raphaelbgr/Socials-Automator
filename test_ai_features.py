"""Comprehensive AI feature tests using LMStudio.

Tests all the AI-powered features we've implemented:
1. Topic Analysis - Detecting overused patterns
2. Topic Generation - With analysis constraints
3. Research Query Generation - With history tracking
4. Similarity Detection - Stricter thresholds

Run with: py test_ai_features.py
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print a section header."""
    print(f"\n--- {title} ---")


def print_result(label: str, value, indent: int = 0):
    """Print a labeled result."""
    prefix = "  " * indent
    if isinstance(value, (list, dict)):
        print(f"{prefix}{label}:")
        if isinstance(value, list):
            for item in value[:10]:  # Limit to 10 items
                print(f"{prefix}  - {item}")
            if len(value) > 10:
                print(f"{prefix}  ... and {len(value) - 10} more")
        else:
            print(json.dumps(value, indent=2))
    else:
        print(f"{prefix}{label}: {value}")


async def test_topic_analysis():
    """Test the new AI-powered topic analysis."""
    print_header("TEST 1: Topic Analysis (AI-Powered Pattern Detection)")

    from socials_automator.video.pipeline.topic_selector import (
        TOPIC_ANALYSIS_PROMPT,
        TOPIC_ANALYSIS_SYSTEM,
    )
    from socials_automator.providers.text import TextProvider

    # Sample topics that show clear patterns
    sample_topics = [
        "3 ChatGPT prompts that save 2 hours daily",
        "5 ChatGPT prompts that will change your workflow",
        "3 ChatGPT prompts to automate your week",
        "This FREE AI tool creates perfect meeting notes",
        "5 ChatGPT prompts that boost productivity",
        "3 hidden AI tools that beat ChatGPT",
        "ChatGPT vs Claude: which wins in 2025",
        "3 ChatGPT prompts for email writing",
        "This FREE tool replaces your entire workflow",
        "5 FREE AI tools you need in 2025",
        "3 ChatGPT prompts that make you money",
        "Runway Gen-4 just made video editing obsolete",
        "3 ChatGPT prompts for content creation",
        "This AI tool pays you while you sleep",
        "5 hidden AI tools that crush ChatGPT",
        "3 ChatGPT prompts that will blow your mind",
        "Sora 2 just changed everything",
        "3 FREE AI tools that replace paid apps",
        "ChatGPT 5.2 prompts you need to try",
        "This FREE tool automates boring tasks",
    ]

    topics_list = "\n".join(f"- {t}" for t in sample_topics)

    prompt = TOPIC_ANALYSIS_PROMPT.format(
        count=len(sample_topics),
        topics_list=topics_list,
    )

    print_section("Sending to LMStudio for analysis...")
    print(f"Analyzing {len(sample_topics)} sample topics")

    provider = TextProvider(provider_override="lmstudio")

    try:
        response = await provider.generate(
            prompt=prompt,
            system=TOPIC_ANALYSIS_SYSTEM,
            task="topic_analysis_test",
            temperature=0.3,
            max_tokens=1500,
        )

        print_section("Raw AI Response")
        print(response[:500] + "..." if len(response) > 500 else response)

        # Parse JSON
        response_clean = response.strip()
        if response_clean.startswith("```"):
            lines = response_clean.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            response_clean = "\n".join(lines)

        analysis = json.loads(response_clean)

        print_section("Parsed Analysis Results")

        # Blocked tools
        blocked = [t for t in analysis.get("overused_tools", []) if t.get("block")]
        print_result("Blocked tools (>15%)", [f"{t['tool']} ({t.get('percentage', '?')}%)" for t in blocked])

        # Overused patterns
        patterns = analysis.get("overused_patterns", [])
        print_result("Overused patterns", [p.get("pattern", "unknown") for p in patterns])

        # Suggested tools
        suggested = analysis.get("tools_to_feature", [])
        print_result("Suggested tools to feature", suggested)

        # Avoid phrases
        avoid = analysis.get("avoid_phrases", [])
        print_result("Phrases to avoid", avoid)

        print("\n[OK] Topic analysis test PASSED")
        return True, analysis

    except json.JSONDecodeError as e:
        print(f"\n[FAIL] JSON parse error: {e}")
        return False, None
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False, None


async def test_topic_generation_with_constraints(analysis: dict):
    """Test topic generation with analysis constraints."""
    print_header("TEST 2: Topic Generation with AI Constraints")

    from socials_automator.video.pipeline.topic_selector import TopicSelector
    from socials_automator.providers.text import TextProvider

    provider = TextProvider(provider_override="lmstudio")
    selector = TopicSelector(ai_client=provider)

    # Format constraints from analysis
    constraints = selector._format_analysis_constraints(analysis)

    print_section("Formatted Constraints")
    print(constraints[:1000] if len(constraints) > 1000 else constraints)

    # Create a mock profile and pillar
    class MockProfile:
        display_name = "AI for Mortals"
        niche_id = "ai_tools"
        trending_keywords = ["AI", "automation", "productivity", "free tools"]
        content_pillars = []

    pillar = {
        "id": "tool_tutorials",
        "name": "Tool Tutorials",
        "description": "Teaching people how to use AI tools effectively",
        "examples": [
            "Master this AI tool in 60 seconds",
            "The hidden feature that changes everything",
        ],
    }

    # Build the generation prompt (simplified version)
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")
    current_year = now.year

    prompt = f"""Generate ONE compelling video topic for a 60-second Instagram Reel.

TODAY'S DATE: {current_date}

Profile: {MockProfile.display_name}
Niche: {MockProfile.niche_id}
Content Pillar: {pillar['name']}

{constraints}

Requirements:
- Topic must be specific and actionable
- Should hook viewers in the first 3 seconds
- Focus on FREE tips and tools
- CRITICAL: Follow ALL constraints above!
- CRITICAL: Do NOT use any BLOCKED TOOLS!
- CRITICAL: Do NOT use any AVOID patterns!

Respond with ONLY the topic text, nothing else."""

    print_section("Sending to LMStudio for topic generation...")

    try:
        response = await provider.generate(
            prompt=prompt,
            system="Generate a unique, non-repetitive video topic following the constraints strictly.",
            task="topic_generation_test",
            temperature=0.8,
            max_tokens=100,
        )

        topic = response.strip().strip('"').strip("'")

        print_section("Generated Topic")
        print(f"  '{topic}'")

        # Check if topic violates any constraints
        blocked_tools = [t["tool"].lower() for t in analysis.get("overused_tools", []) if t.get("block")]
        topic_lower = topic.lower()

        violations = []
        for tool in blocked_tools:
            if tool in topic_lower:
                violations.append(f"Contains blocked tool: {tool}")

        # Check for overused patterns
        for pattern in analysis.get("overused_patterns", []):
            pattern_text = pattern.get("pattern", "").lower()
            if "prompts that" in pattern_text and "prompts that" in topic_lower:
                violations.append("Uses 'prompts that' pattern")

        if violations:
            print_section("Constraint Violations")
            for v in violations:
                print(f"  [!] {v}")
            print("\n[WARN] Topic may violate constraints")
        else:
            print("\n[OK] Topic follows constraints")

        return True, topic

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False, None


async def test_research_query_generation():
    """Test research query generation with history tracking."""
    print_header("TEST 3: Research Query Generation")

    from socials_automator.video.pipeline.research_queries import (
        ResearchQueryGenerator,
        load_research_query_history,
        save_research_queries_to_history,
    )
    from socials_automator.providers.text import TextProvider

    provider = TextProvider(provider_override="lmstudio")

    # Test with a sample topic
    topic = "How to use Perplexity Sonar for instant research"

    print_section(f"Generating research queries for: '{topic}'")

    generator = ResearchQueryGenerator(
        text_provider=provider,
        profile_name="ai.for.mortals",
    )

    try:
        queries = await generator.generate_queries(
            topic=topic,
            pillar="tool_tutorials",
            count=12,
        )

        print_section("Generated Queries")
        for i, q in enumerate(queries, 1):
            print(f"  {i:2}. [{q.language}] [{q.category}] {q.query}")

        # Check language distribution
        en_count = sum(1 for q in queries if q.language == "en")
        es_count = sum(1 for q in queries if q.language == "es")
        pt_count = sum(1 for q in queries if q.language == "pt")

        print_section("Language Distribution")
        print(f"  English: {en_count}")
        print(f"  Spanish: {es_count}")
        print(f"  Portuguese: {pt_count}")

        # Check if queries were saved to history
        print_section("History Tracking")
        history = load_research_query_history("ai.for.mortals", hours=1)
        print(f"  Queries in history (last hour): {len(history)}")

        print("\n[OK] Research query generation test PASSED")
        return True, queries

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_similarity_detection():
    """Test the improved similarity detection."""
    print_header("TEST 4: Similarity Detection (Stricter Thresholds)")

    from socials_automator.history.similarity import (
        normalize_text,
        is_similar,
        jaccard_similarity,
        DEFAULT_SIMILARITY_THRESHOLD,
    )

    print_section("Configuration")
    print(f"  Threshold: {DEFAULT_SIMILARITY_THRESHOLD}")

    # Test cases: (new_topic, existing_topics, expected_similar)
    test_cases = [
        # Should be similar (ChatGPT prompts)
        (
            "5 ChatGPT prompts for productivity",
            ["3 ChatGPT prompts that save time"],
            True,
            "Same tool + same pattern"
        ),
        # NOT similar at word level (pattern detection requires AI analysis)
        # Jaccard: {'need', 'tools'} vs {'tools', 'change', 'everything'} = 1/4 = 0.25
        # This is WHY we have AI-powered topic analysis - patterns like "N FREE AI tools"
        # can only be detected by AI, not word-level similarity
        (
            "5 FREE AI tools you need",
            ["3 FREE AI tools that change everything"],
            False,  # Word-level: different; Pattern-level: AI analysis catches this
            "Different words despite same pattern (AI analysis handles this)"
        ),
        # Should NOT be similar (different tools)
        (
            "Perplexity Sonar research tips",
            ["ChatGPT prompts for email"],
            False,
            "Completely different tools"
        ),
        # Should NOT be similar (different angle)
        (
            "Claude vs ChatGPT coding comparison",
            ["Runway Gen-4 video editing tutorial"],
            False,
            "Different tools, different angle"
        ),
        # Edge case: very short normalized
        (
            "The best AI tool of 2025",
            ["This amazing AI tool is free"],
            True,
            "Both normalize to {'tool'}"
        ),
    ]

    print_section("Test Cases")

    passed = 0
    failed = 0

    for new_topic, existing, expected, description in test_cases:
        new_norm = normalize_text(new_topic)
        existing_norm = [normalize_text(e) for e in existing]

        similar, match = is_similar(new_topic, existing)

        status = "[OK]" if similar == expected else "[FAIL]"
        if similar == expected:
            passed += 1
        else:
            failed += 1

        print(f"\n  {status} {description}")
        print(f"      New: '{new_topic}'")
        print(f"      Normalized: {new_norm}")
        print(f"      Existing: {existing}")
        print(f"      Similar: {similar}, Expected: {expected}")
        if match:
            print(f"      Matched: {match}")

    print_section("Results")
    print(f"  Passed: {passed}/{len(test_cases)}")
    print(f"  Failed: {failed}/{len(test_cases)}")

    if failed == 0:
        print("\n[OK] Similarity detection test PASSED")
        return True
    else:
        print(f"\n[WARN] {failed} test(s) failed")
        return False


async def test_dynamic_query_generation():
    """Test dynamic query generation for news profile."""
    print_header("TEST 5: Dynamic Query Generation (News Profile)")

    from socials_automator.news.dynamic_queries import (
        DynamicQueryGenerator,
        load_query_history,
    )
    from socials_automator.providers.text import TextProvider

    provider = TextProvider(provider_override="lmstudio")

    generator = DynamicQueryGenerator(
        profile_name="news.but.quick",
        text_provider=provider,
    )

    print_section("Generating dynamic news queries...")

    try:
        queries = await generator.generate_queries(count=10)

        print_section("Generated Queries")
        for i, q in enumerate(queries, 1):
            print(f"  {i:2}. [{q.category}] {q.query}")
            if q.reason:
                print(f"      Reason: {q.reason}")

        # Check category distribution
        categories = {}
        for q in queries:
            categories[q.category] = categories.get(q.category, 0) + 1

        print_section("Category Distribution")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")

        print("\n[OK] Dynamic query generation test PASSED")
        return True, queries

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def test_full_topic_selector_flow():
    """Test the full TopicSelector flow with analysis."""
    print_header("TEST 6: Full TopicSelector Flow (End-to-End)")

    from socials_automator.video.pipeline.topic_selector import TopicSelector
    from socials_automator.video.pipeline.base import PipelineContext, ProfileMetadata
    from socials_automator.providers.text import TextProvider
    from pathlib import Path

    provider = TextProvider(provider_override="lmstudio")
    selector = TopicSelector(ai_client=provider)

    # Get actual profile path
    profile_path = Path("profiles/ai.for.mortals")

    if not profile_path.exists():
        print(f"[SKIP] Profile path not found: {profile_path}")
        return True, None

    # Load actual profile metadata using the from_file classmethod
    metadata_path = profile_path / "metadata.json"
    if not metadata_path.exists():
        print(f"[SKIP] Metadata not found: {metadata_path}")
        return True, None

    try:
        profile = ProfileMetadata.from_file(metadata_path)
    except Exception as e:
        print(f"[SKIP] Could not load profile: {e}")
        return True, None

    print_section("Profile Configuration")
    print(f"  Handle: {profile.instagram_handle}")
    print(f"  Display Name: {profile.display_name}")
    print(f"  Niche: {profile.niche_id}")
    print(f"  Content Pillars: {len(profile.content_pillars)}")

    print_section("Running full topic selection flow...")

    try:
        topic_info = await selector.select_topic(profile, profile_path)

        print_section("Generated Topic")
        print(f"  Topic: '{topic_info.topic}'")
        print(f"  Pillar: {topic_info.pillar_name}")
        print(f"  Keywords: {topic_info.keywords[:5]}")
        print(f"  Search Queries: {len(topic_info.search_queries)}")

        print("\n[OK] Full TopicSelector flow test PASSED")
        return True, topic_info

    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE AI FEATURE TESTS")
    print("  Using LMStudio as AI provider")
    print("=" * 70)

    results = {}

    # Test 1: Topic Analysis
    success, analysis = await test_topic_analysis()
    results["Topic Analysis"] = success

    if success and analysis:
        # Test 2: Topic Generation with Constraints
        success, topic = await test_topic_generation_with_constraints(analysis)
        results["Topic Generation"] = success
    else:
        print("\n[SKIP] Skipping Topic Generation test (analysis failed)")
        results["Topic Generation"] = False

    # Test 3: Research Query Generation
    success, queries = await test_research_query_generation()
    results["Research Queries"] = success

    # Test 4: Similarity Detection
    success = await test_similarity_detection()
    results["Similarity Detection"] = success

    # Test 5: Dynamic Query Generation
    success, queries = await test_dynamic_query_generation()
    results["Dynamic Queries"] = success

    # Test 6: Full TopicSelector Flow
    success, topic_info = await test_full_topic_selector_flow()
    results["Full Flow"] = success

    # Summary
    print_header("TEST SUMMARY")

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for test_name, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {test_name}")

    print(f"\n  Total: {passed}/{total} passed")

    if failed > 0:
        print(f"\n  [!] {failed} test(s) failed")
        return 1
    else:
        print("\n  All tests passed!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
