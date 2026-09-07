"""Path helpers for the TraceShift research tree."""

from __future__ import annotations

from pathlib import Path


def research_root() -> Path:
    """Return the ``research/`` directory (parent of the ``traceshift`` package)."""
    return Path(__file__).resolve().parent.parent


def default_experiment_config() -> Path:
    return research_root() / "configs" / "experiment.yaml"


def default_cloze_probes() -> Path:
    return research_root() / "data" / "frozen" / "cloze_probes.yaml"
