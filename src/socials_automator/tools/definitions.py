"""Tool definitions for AI function calling.

Defines tools that AI can autonomously decide to use during content generation.
Uses OpenAI function calling format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> str:
        """Convert result to a message string for AI consumption."""
        if not self.success:
            return f"Tool '{self.tool_name}' failed: {self.error}"

        if isinstance(self.result, str):
            return self.result

        import json
        return json.dumps(self.result, indent=2, ensure_ascii=False)


# Tool schemas in OpenAI function calling format
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information on a topic. Use this to find current facts, "
                "statistics, examples, and expert opinions to enrich content. "
                "Returns relevant web results with titles, URLs, and snippets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of 3-5 search queries. Include variations: "
                            "main topic, 'best' prefix, 'how to' prefix, current year."
                        ),
                        "minItems": 1,
                        "maxItems": 10,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results per query (1-10). Default: 5",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": (
                "Search for recent news articles on a topic. Use this to find trending stories, "
                "recent developments, and timely information. Great for adding current relevance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of 2-3 news search queries focused on recent events.",
                        "minItems": 1,
                        "maxItems": 5,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum results per query (1-10). Default: 3",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate an image using AI (ComfyUI/DALL-E). Use this when you need a custom image "
                "for slide backgrounds or illustrations. Returns the image path. "
                "Best for: concept visualizations, abstract backgrounds, thematic illustrations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Detailed image description. Be specific about: "
                            "subject, style (realistic/lifestyle/minimal), colors, mood, lighting. "
                            "Prefer lifestyle/environmental scenes with human elements over techy imagery. "
                            "Example: 'Person working at sunny cafe with laptop, warm natural lighting, "
                            "plants in background, cozy lifestyle aesthetic'"
                        ),
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": (
                            "Things to avoid in the image. Default: 'blurry, low quality, distorted, text, watermark'"
                        ),
                    },
                    "style": {
                        "type": "string",
                        "enum": ["abstract", "realistic", "minimal", "illustration", "photography"],
                        "description": "Image style. Affects prompt enhancement.",
                    },
                    "size": {
                        "type": "string",
                        "enum": ["square", "portrait", "landscape"],
                        "description": "Image dimensions. Default: square (1080x1080)",
                        "default": "square",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
]


# Tool name to schema mapping for quick lookup
AVAILABLE_TOOLS = {
    schema["function"]["name"]: schema
    for schema in TOOL_SCHEMAS
}
