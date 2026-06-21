"""Optional LLM prompt enhancer backed by Nebius Token Factory (Cosmos3 reasoner).

This is the Layer-A "understand the domain" upgrade seam. When a Token Factory
API key is available it refines a base sweep prompt into a richer, physically
grounded image-generation prompt using nvidia/Cosmos3-Super-Reasoner. On any
error (no key, network, bad response) it returns the base prompt unchanged so
the pipeline never breaks and stays deterministic offline.

Stdlib-only (urllib); no new third-party dependencies.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any

TOKEN_FACTORY_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
REASONER_MODEL = "nvidia/Cosmos3-Super-Reasoner"

_SYSTEM = (
    "You rewrite robot dataset scene descriptions into a single vivid, physically "
    "grounded image-generation prompt for a robot manipulation dataset. Keep the "
    "robot task and all listed scene attributes (lighting, surfaces, object "
    "positions, clutter, camera). Output ONE prompt paragraph, no preamble, no lists."
)


def is_available() -> bool:
    return bool(os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip())


def enhance_prompt(base_prompt: str, *, temperature: float = 0.2, timeout: int = 40) -> str:
    """Refine base_prompt via the Cosmos reasoner; fall back to base on any error."""
    key = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if not key:
        return base_prompt
    payload = {
        "model": REASONER_MODEL,
        "max_tokens": 320,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": "Rewrite into one image prompt:\n" + base_prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(TOKEN_FACTORY_URL, data=data, method="POST")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        return text or base_prompt
    except Exception:
        return base_prompt

# ---------------------------------------------------------------------------
# LLM-driven scene schema (replaces keyword/word-matching domain packs).
# The reasoner reads the plain-English goal and decides WHICH environmental
# axes are worth varying for robustness, and proposes concrete values per axis.
# ---------------------------------------------------------------------------

_SCHEMA_SYSTEM = (
    "You are a robotics data-augmentation planner. Given a plain-English robot "
    "goal, decide which ENVIRONMENTAL axes are most useful to vary so a learned "
    "policy generalizes (sim2real robustness). Return STRICT JSON only, no prose, "
    "with this shape: {\"scene\": str, \"axes\": {\"<axis_name>\": [\"value\", ...]}}. "
    "Pick 4-6 axes that genuinely matter FOR THIS TASK (e.g. lighting, surface/"
    "texture, background, object_layout, clutter/distractors, camera_angle, and "
    "task-specific ones like fridge_fill_level or shelf_arrangement). Give 3-5 "
    "concrete values per axis. Be specific to the goal's real environment."
)


def build_scene_schema_llm(goal: str, *, timeout: int = 50):
    """Ask the Cosmos reasoner for a goal-specific scene schema (JSON).

    Returns a dict {scene, axes:{axis:[values]}} or None on any failure so the
    caller can fall back to the deterministic keyword schema.
    """
    key = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if not key:
        return None
    payload = {
        "model": REASONER_MODEL,
        "max_tokens": 600,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": _SCHEMA_SYSTEM},
            {"role": "user", "content": "Robot goal: " + goal},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(TOKEN_FACTORY_URL, data=data, method="POST")
    req.add_header("Authorization", "Bearer " + key)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"].strip()
        # strip code fences if present
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.lstrip().lower().startswith("json"):
                text = text.lstrip()[4:]
        start, end = text.find("{"), text.rfind("}")
        schema = json.loads(text[start:end + 1])
        if isinstance(schema, dict) and isinstance(schema.get("axes"), dict) and schema["axes"]:
            return schema
        return None
    except Exception:
        return None
