"""Schemas for the three target behaviors.

These are the single source of truth used both to *generate* synthetic training data and
to *validate* model output during evaluation:

  * DECK_PLAN_SCHEMA -- the structured deck plan a "schema" example must satisfy.
  * TOOL_SCHEMAS      -- per-tool argument schemas for the agent's tool surface.

The tool surface mirrors the orchestrator design: plan the deck, author animation HTML,
render it, inspect the render, then assemble the file.
"""
from __future__ import annotations

LAYOUTS = [
    "title", "title-bullets", "two-column", "section",
    "image-focus", "quote", "chart",
]
ANIMATION_KINDS = [
    "none", "build-in", "diagram", "chart-grow", "highlight", "transition",
]

# --------------------------------------------------------------------------------------
# Deck plan
# --------------------------------------------------------------------------------------
DECK_PLAN_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "slides"],
    "properties": {
        "title": {"type": "string", "minLength": 1},
        "theme": {"type": "string"},
        "slides": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "title", "layout", "bullets", "needs_animation"],
                "properties": {
                    "index": {"type": "integer", "minimum": 0},
                    "title": {"type": "string", "minLength": 1},
                    "layout": {"enum": LAYOUTS},
                    "bullets": {"type": "array", "items": {"type": "string"}},
                    "needs_animation": {"type": "boolean"},
                    "animation_kind": {"enum": ANIMATION_KINDS},
                },
            },
        },
    },
}

# --------------------------------------------------------------------------------------
# Agent tool surface
# --------------------------------------------------------------------------------------
TOOL_SCHEMAS: dict[str, dict] = {
    "plan_deck": {
        "type": "object",
        "additionalProperties": False,
        "required": ["topic"],
        "properties": {
            "topic": {"type": "string", "minLength": 1},
            "audience": {"type": "string"},
            "max_slides": {"type": "integer", "minimum": 1, "maximum": 40},
        },
    },
    "generate_animation_html": {
        "type": "object",
        "additionalProperties": False,
        "required": ["slide_index", "animation_kind"],
        "properties": {
            "slide_index": {"type": "integer", "minimum": 0},
            "animation_kind": {"enum": ANIMATION_KINDS},
            "notes": {"type": "string"},
        },
    },
    "render_animation": {
        "type": "object",
        "additionalProperties": False,
        "required": ["slide_index", "html_ref"],
        "properties": {
            "slide_index": {"type": "integer", "minimum": 0},
            "html_ref": {"type": "string", "minLength": 1},
        },
    },
    "inspect_render": {
        "type": "object",
        "additionalProperties": False,
        "required": ["slide_index", "render_ref"],
        "properties": {
            "slide_index": {"type": "integer", "minimum": 0},
            "render_ref": {"type": "string", "minLength": 1},
        },
    },
    "assemble_pptx": {
        "type": "object",
        "additionalProperties": False,
        "required": ["output_name"],
        "properties": {
            "output_name": {"type": "string", "minLength": 1},
            "include_slides": {"type": "array", "items": {"type": "integer"}},
        },
    },
}

ALLOWED_TOOLS = set(TOOL_SCHEMAS)

# Wrapper schema for a single tool call emitted by the model.
TOOL_CALL_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "arguments"],
    "properties": {
        "name": {"enum": sorted(ALLOWED_TOOLS)},
        "arguments": {"type": "object"},
    },
}


def tools_description() -> str:
    """A compact human-readable tool list for system prompts."""
    lines = ["Available tools:"]
    for name, schema in TOOL_SCHEMAS.items():
        req = ", ".join(schema.get("required", []))
        lines.append(f"- {name}(required: {req or 'none'})")
    return "\n".join(lines)
