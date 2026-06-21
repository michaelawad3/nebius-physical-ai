"""Unit tests for the structured prompt-sweep dataset-augmentation generator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src" / "npa" / "workflows" / "prompt_sweep.py"
)
_spec = importlib.util.spec_from_file_location("prompt_sweep", _MODULE_PATH)
prompt_sweep = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(prompt_sweep)

GOAL = "train a robot to be a barista in my coffee shop"


def test_variation_has_required_keys():
    v = prompt_sweep.generate_variation(GOAL, 0, seed=0)
    assert set(v) >= {"perturbation", "axes", "enhanced_prompt"}


def test_legacy_perturbation_label_is_backward_compatible():
    v = prompt_sweep.generate_variation(GOAL, 3, seed=7)
    assert v["perturbation"] in prompt_sweep.LEGACY_PERTURBATION_LABELS


def test_axes_are_structured_not_bare_words():
    axes = prompt_sweep.generate_variation(GOAL, 0, seed=0)["axes"]
    for key in ("lighting", "background", "object_configuration", "texture", "camera"):
        assert isinstance(axes[key], dict)
    assert isinstance(axes["object_configuration"]["distractors"], list)


def test_deterministic_for_same_seed_and_index():
    a = prompt_sweep.generate_variation(GOAL, 5, seed=42)
    b = prompt_sweep.generate_variation(GOAL, 5, seed=42)
    assert a == b


def test_enhanced_prompt_preserves_goal_text():
    v = prompt_sweep.generate_variation(GOAL, 1, seed=1)
    assert "barista" in v["enhanced_prompt"]
    assert len(v["enhanced_prompt"]) > 80


def test_weather_axis_only_for_outdoor_goals():
    indoor = prompt_sweep.generate_variation(GOAL, 0, seed=0)["axes"]
    outdoor = prompt_sweep.generate_variation("robot for street delivery", 0, seed=0)["axes"]
    assert "weather" not in indoor
    assert "weather" in outdoor


def test_generate_sweep_count():
    sweep = prompt_sweep.generate_sweep(GOAL, 8, seed=0)
    assert len(sweep) == 8
    assert len({s["enhanced_prompt"] for s in sweep}) > 1


def test_resolve_goal_env_override(monkeypatch):
    monkeypatch.setenv("NPA_SIM2REAL_AUGMENT_GOAL", "custom goal")
    assert prompt_sweep.resolve_goal(object()) == "custom goal"
