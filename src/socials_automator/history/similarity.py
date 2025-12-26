"""Similarity functions for content deduplication.

Uses Jaccard similarity to detect near-duplicate content by comparing
normalized word sets. This catches variations like:
- "3 ChatGPT prompts that save time" vs "5 ChatGPT prompts to save time"
- "AI tools for productivity" vs "Best AI productivity tools"
"""

from __future__ import annotations

import re

# Common filler words to remove during normalization
FILLER_WORDS = frozenset({
    # Articles & prepositions
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "with", "and", "or",
    # Verbs
    "is", "are", "was", "were", "will", "can", "could", "should", "would",
    # Pronouns
    "this", "that", "these", "those", "your", "you", "my", "i", "me", "we",
    # Question words
    "how", "why", "what", "when", "where", "which", "who",
    # Common modifiers (often used interchangeably)
    "just", "really", "very", "most", "more", "some", "any", "all", "every",
    # Marketing words (overused, don't differentiate)
    "free", "new", "best", "top", "hidden", "secret", "ultimate", "amazing",
    "incredible", "powerful", "simple", "easy", "quick", "fast",
    # Years (content is often similar regardless of year)
    "2024", "2025", "2026",
    # Numbers written as words
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # Time units (don't differentiate "save 2 hours" vs "save 10 minutes")
    "hours", "hour", "minutes", "minute", "seconds", "second",
    "daily", "weekly", "monthly", "week", "day", "month", "year",
    # Common action verbs in AI content (don't differentiate topics)
    "save", "change", "replace", "automate", "make", "turn", "use", "create",
    "boost", "improve", "transform", "revolutionize", "unlock", "master",
    # Common result words
    "forever", "instantly", "faster", "better", "smarter", "obsolete",
    # Common patterns in AI topics
    "workflow", "productivity", "work", "tasks", "life",
})

# Default similarity threshold - lower = stricter (catches more similar topics)
DEFAULT_SIMILARITY_THRESHOLD = 0.35


def normalize_text(text: str) -> set[str]:
    """Normalize text for similarity comparison.

    Extracts meaningful words by:
    1. Converting to lowercase
    2. Removing punctuation
    3. Filtering out filler words
    4. Filtering out digit-only words (3, 5, 10, etc.)
    5. Keeping only words with 3+ characters

    Args:
        text: Text to normalize (topic, headline, etc.)

    Returns:
        Set of normalized key words.
    """
    if not text:
        return set()

    # Lowercase and remove punctuation
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", "", normalized)

    # Split and filter
    words = {
        word for word in normalized.split()
        if word not in FILLER_WORDS
        and len(word) > 2
        and not word.isdigit()  # Filter out "3", "5", "10", etc.
    }

    return words


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Calculate Jaccard similarity between two sets.

    Jaccard = |intersection| / |union|

    Args:
        set_a: First set of words.
        set_b: Second set of words.

    Returns:
        Similarity score from 0.0 (no overlap) to 1.0 (identical).
    """
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def is_similar(
    text: str,
    existing_texts: list[str],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> tuple[bool, str | None]:
    """Check if text is too similar to any existing texts.

    Args:
        text: New text to check.
        existing_texts: List of existing texts to compare against.
        threshold: Similarity threshold (default 0.35 = 35% word overlap).
                   Lower = stricter, catches more similar topics.

    Returns:
        Tuple of (is_similar, matching_text or None).
    """
    text_words = normalize_text(text)

    if not text_words:
        return False, None

    for existing in existing_texts:
        existing_words = normalize_text(existing)

        if not existing_words:
            continue

        similarity = jaccard_similarity(text_words, existing_words)

        if similarity >= threshold:
            return True, existing

    return False, None


def find_most_similar(
    text: str,
    existing_texts: list[str],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Find the most similar existing texts.

    Useful for debugging and understanding why content was rejected.

    Args:
        text: Text to compare.
        existing_texts: List of texts to compare against.
        top_n: Number of top matches to return.

    Returns:
        List of (text, similarity_score) tuples, sorted by similarity.
    """
    text_words = normalize_text(text)

    if not text_words:
        return []

    similarities = []

    for existing in existing_texts:
        existing_words = normalize_text(existing)

        if not existing_words:
            continue

        similarity = jaccard_similarity(text_words, existing_words)
        similarities.append((existing, similarity))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]
