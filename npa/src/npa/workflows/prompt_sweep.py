"""Structured prompt-sweep generator for Sim2Real dataset augmentation.

Expands a plain-English robotics goal (e.g. "train a robot to be a barista in my
coffee shop") into deterministic, Cosmos-ready dataset variations that sweep the
common robustness axes: lighting, weather/atmosphere (outdoor goals), background,
object configuration/layout, texture/material, camera/contrast style.

Architecture (two layers):
    A) build_scene_schema(goal): the once-per-goal "understand the domain" step.
       Produces a generic, domain-agnostic scene vocabulary, refined by keyword
       domain packs (cafe / warehouse / kitchen / outdoor / ...). This is the
       single seam where a real LLM enhancer can later be dropped in.
    B) generate_variation(goal, index, seed): the cheap deterministic fan-out,
       called once per frame (commonly hundreds, up to ~1024 per run). Samples
       concrete values from the scene schema -> structured axes + enhanced prompt.

Design constraints (intentionally conservative):
    * Pure-Python, no new third-party dependencies, no network / LLM calls.
    * Fully deterministic given a seed -> reproducible datasets and tests.
    * Backward compatible: every variation still exposes a "perturbation" label
      from ["lighting", "texture", "background", "contrast"].
    * generate_variation() signature is stable so callers never change.
"""

from __future__ import annotations

import os
import random
from typing import Any

LEGACY_PERTURBATION_LABELS: list[str] = ["lighting", "texture", "background", "contrast"]

# ---------------------------------------------------------------------------
# Generic, domain-agnostic vocabulary (axes whose values transfer everywhere).
# ---------------------------------------------------------------------------
_LIGHTING_TYPES = [
    "warm indoor lighting",
    "cool fluorescent overhead lighting",
    "soft natural window daylight",
    "dim ambient evening lighting",
    "bright midday light",
    "mixed warm and cool task lighting",
]
_LIGHTING_INTENSITY = ["low", "medium", "high"]
_LIGHTING_SHADOW = ["soft", "moderate", "hard"]

_WEATHER = ["clear", "overcast", "light rain", "bright sun glare", "foggy morning light"]

_BACKGROUND_CLUTTER = ["minimal", "moderate", "heavy"]
_CAMERA_VIEWS = [
    "front-left counter-level view",
    "overhead top-down view",
    "right-side angled view",
    "wrist-mounted gripper view",
    "front eye-level view",
]
_CAMERA_STYLE = ["neutral contrast", "high-contrast crisp detail", "soft low-contrast film look"]

# ---------------------------------------------------------------------------
# Scene schema: the domain vocabulary produced once per goal (Layer A).
# Generic defaults below; keyword packs override the domain-specific slots.
# ---------------------------------------------------------------------------
_GENERIC_SCHEMA: dict[str, Any] = {
    "workspace": "a tabletop workspace",
    "scenes": ["a clean tabletop", "a cluttered workbench", "an indoor work area"],
    "primary_object": "target object",
    "primary_object_material": ["white plastic", "matte metal", "painted wood"],
    "secondary_object": "tool",
    "surfaces": ["light laminate", "dark matte surface", "stainless steel"],
    "distractor_pool": ["clutter item", "small box", "loose cable", "cloth", "container"],
    "primary_positions": ["left of the workspace", "center of the workspace", "right of the robot gripper"],
    "secondary_positions": ["near the robot gripper", "at the back of the workspace", "beside the target object"],
}

# Keyword -> partial schema override. First matching pack wins.
_DOMAIN_PACKS: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        ("barista", "coffee", "cafe", "espresso", "latte"),
        {
            "workspace": "an espresso counter in a coffee shop",
            "scenes": ["busy coffee shop counter", "quiet empty cafe at opening time",
                       "minimalist modern espresso bar"],
            "primary_object": "coffee cup",
            "primary_object_material": ["white ceramic", "clear glass", "matte black stoneware", "disposable paper"],
            "secondary_object": "milk pitcher",
            "surfaces": ["dark matte stone", "light marble", "stainless steel", "warm wooden butcher block"],
            "distractor_pool": ["napkins", "spoon", "receipt", "sugar packets", "stir stick", "to-go lid"],
            "primary_positions": ["left of the espresso machine", "directly under the spout", "center of the counter"],
            "secondary_positions": ["near the robot gripper", "on the steaming station", "left of the cup"],
        },
    ),
    (
        ("warehouse", "package", "parcel", "logistics", "fulfillment", "sorting", "pick and place"),
        {
            "workspace": "a warehouse conveyor and sorting station",
            "scenes": ["busy fulfillment warehouse", "loading dock", "rows of storage shelving"],
            "primary_object": "cardboard box",
            "primary_object_material": ["plain corrugated", "shrink-wrapped", "branded retail"],
            "secondary_object": "shipping label scanner",
            "surfaces": ["galvanized steel conveyor", "concrete floor", "black rubber belt"],
            "distractor_pool": ["packing tape", "bubble wrap", "pallet", "packing peanuts", "barcode label"],
            "primary_positions": ["on the conveyor belt", "at the edge of the table", "center of the sorting bin"],
            "secondary_positions": ["mounted above the belt", "at the side of the station", "near the robot gripper"],
        },
    ),
    (
        ("kitchen", "cook", "dish", "plate", "chef", "food"),
        {
            "workspace": "a kitchen prep counter",
            "scenes": ["home kitchen counter", "industrial commercial kitchen", "tidy prep station"],
            "primary_object": "ceramic plate",
            "primary_object_material": ["white ceramic", "stainless steel", "glass"],
            "secondary_object": "utensil",
            "surfaces": ["wooden cutting board", "stainless steel counter", "granite countertop"],
            "distractor_pool": ["dish towel", "cutlery", "bowl", "spice jar", "sponge"],
            "primary_positions": ["left of the sink", "center of the counter", "on the drying rack"],
            "secondary_positions": ["in the utensil holder", "near the robot gripper", "beside the plate"],
        },
    ),
    (
        ("laundry", "fold", "clothes", "fabric", "textile", "garment"),
        {
            "workspace": "a laundry folding table",
            "scenes": ["home laundry room", "tidy folding station", "cluttered utility room"],
            "primary_object": "folded shirt",
            "primary_object_material": ["cotton fabric", "denim", "knit wool"],
            "secondary_object": "laundry basket",
            "surfaces": ["light wooden table", "white melamine", "padded folding mat"],
            "distractor_pool": ["sock", "hanger", "detergent bottle", "lint roller", "towel"],
            "primary_positions": ["center of the table", "left of the basket", "on top of the pile"],
            "secondary_positions": ["beside the table", "near the robot gripper", "on the floor"],
        },
    ),
]

_OUTDOOR_HINTS = (
    "outdoor", "patio", "street", "garden", "yard", "warehouse",
    "field", "drive", "delivery", "sidewalk", "loading dock",
)


def is_outdoor_relevant(goal: str) -> bool:
    """Return True when the goal suggests an outdoor / weather-relevant scene."""
    text = (goal or "").lower()
    return any(hint in text for hint in _OUTDOOR_HINTS)


def build_scene_schema(goal: str) -> dict[str, Any]:
    """Layer A: turn a plain-English goal into a domain scene vocabulary.

    Starts from a generic schema and applies the first matching keyword domain
    pack. This is the single seam to later replace with an LLM enhancer.
    """
    schema: dict[str, Any] = {k: (list(v) if isinstance(v, list) else v) for k, v in _GENERIC_SCHEMA.items()}
    text = (goal or "").lower()
    for keywords, override in _DOMAIN_PACKS:
        if any(kw in text for kw in keywords):
            schema.update({k: (list(v) if isinstance(v, list) else v) for k, v in override.items()})
            break
    schema["outdoor"] = is_outdoor_relevant(goal)
    return schema


def _pick(rng: random.Random, options: list[str]) -> str:
    return options[rng.randrange(len(options))]


def _sample_distractors(rng: random.Random, pool: list[str]) -> list[str]:
    count = min(rng.randint(1, 3), len(pool))
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return sorted(shuffled[:count])


def build_axes(schema: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Sample a structured, multi-axis variation spec from the scene schema."""
    axes: dict[str, Any] = {
        "lighting": {
            "type": _pick(rng, _LIGHTING_TYPES),
            "intensity": _pick(rng, _LIGHTING_INTENSITY),
            "shadow": _pick(rng, _LIGHTING_SHADOW),
        },
        "background": {
            "scene": _pick(rng, schema["scenes"]),
            "clutter": _pick(rng, _BACKGROUND_CLUTTER),
        },
        "object_configuration": {
            "primary_object": schema["primary_object"],
            "primary_position": _pick(rng, schema["primary_positions"]),
            "secondary_object": schema["secondary_object"],
            "secondary_position": _pick(rng, schema["secondary_positions"]),
            "distractors": _sample_distractors(rng, schema["distractor_pool"]),
        },
        "texture": {
            "surface": _pick(rng, schema["surfaces"]),
            "primary_object_material": _pick(rng, schema["primary_object_material"]),
        },
        "camera": {
            "view": _pick(rng, _CAMERA_VIEWS),
            "style": _pick(rng, _CAMERA_STYLE),
        },
    }
    if schema.get("outdoor"):
        axes["weather"] = {"condition": _pick(rng, _WEATHER)}
    return axes


def _join(items: list[str]) -> str:
    if not items:
        return "no extra clutter"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def build_enhanced_prompt(goal: str, schema: dict[str, Any], axes: dict[str, Any]) -> str:
    """Compose a natural-language Cosmos prompt from goal + schema + axes."""
    light = axes["lighting"]
    bg = axes["background"]
    obj = axes["object_configuration"]
    tex = axes["texture"]
    cam = axes["camera"]
    clean_goal = (goal or "").strip().rstrip(".")
    parts = [
        "An autonomous ROBOT ARM with a mechanical metal gripper (an industrial robot manipulator, NOT a human, no person, no human hands) performing the task: "
        + clean_goal + ".",
        "The articulated robot arm, its joints, and the mechanical end-effector gripper are clearly visible and central in the frame. A robot operates in "
        + schema["workspace"] + ". Setting: " + bg["scene"] + ".",
        light["type"].capitalize() + " at " + light["intensity"] + " intensity with "
        + light["shadow"] + " shadows.",
        "On a " + tex["surface"] + ", a " + tex["primary_object_material"] + " "
        + obj["primary_object"] + " placed " + obj["primary_position"] + ", "
        + obj["secondary_object"] + " " + obj["secondary_position"] + ".",
        bg["clutter"].capitalize() + " background clutter with " + _join(obj["distractors"]) + ".",
    ]
    if "weather" in axes:
        parts.append("Outdoor atmosphere: " + axes["weather"]["condition"] + ".")
    parts.append(
        cam["view"].capitalize() + ", " + cam["style"] + ". "
        + "Photorealistic. The mechanical robotic arm and gripper MUST be visible performing the action; no human present. "
        + "The scene should emphasize clear robot-object interaction and realistic physical layout."
    )
    return " ".join(parts)


def dominant_perturbation(rng: random.Random) -> str:
    """Pick the legacy single-label perturbation (backward compatibility)."""
    return _pick(rng, LEGACY_PERTURBATION_LABELS)


def generate_variation(goal: str, index: int, *, seed: int = 0) -> dict[str, Any]:
    """Generate one structured, Cosmos-ready dataset variation for any goal.

    Returns keys: perturbation (legacy label), axes (structured spec),
    enhanced_prompt (natural-language Cosmos prompt). Deterministic for a given
    (seed, index, goal).
    """
    rng = random.Random((seed, index, goal))
    schema = build_scene_schema(goal)
    axes = build_axes(schema, rng)
    return {
        "perturbation": dominant_perturbation(rng),
        "axes": axes,
        "enhanced_prompt": build_enhanced_prompt(goal, schema, axes),
    }


def generate_sweep(goal: str, count: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """Generate a deterministic sweep of count structured variations."""
    return [generate_variation(goal, i, seed=seed) for i in range(count)]


def resolve_goal(config: Any) -> str:
    """Resolve the plain-English augmentation goal.

    Priority: NPA_SIM2REAL_AUGMENT_GOAL env override, then config.isaac_task,
    then a generic manipulation default.
    """
    env_goal = os.environ.get("NPA_SIM2REAL_AUGMENT_GOAL", "").strip()
    if env_goal:
        return env_goal
    task = (getattr(config, "isaac_task", "") or "").strip()
    if task:
        return "train a robot to perform: " + task
    return "train a robot arm to perform a tabletop manipulation task"