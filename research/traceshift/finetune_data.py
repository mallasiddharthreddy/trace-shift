"""Frozen fine-tuning dataset loading for TraceShift (primary + control).

Same preprocessing for both runs. Text is not modified or augmented.
Example order follows the frozen YAML list order (deterministic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from .paths import research_root
from .yaml_io import load_yaml

DatasetKind = Literal["primary", "control"]

EXPECTED_DATASET_IDS: dict[DatasetKind, str] = {
    "primary": "tennis_biography_ft",
    "control": "control_general_ft",
}
EXPECTED_N_EXAMPLES = 40


class FinetuneDataError(ValueError):
    """Frozen FT dataset resolution / validation failure."""


@dataclass(frozen=True)
class FinetuneExample:
    """One supervised plain-text training example from a frozen FT YAML."""

    id: str
    subject: str
    text: str
    fact_id: str | None = None
    index: int = 0  # 0-based position in frozen YAML order


@dataclass
class FinetuneDataset:
    """Loaded frozen fine-tuning corpus (primary or control)."""

    kind: DatasetKind
    dataset_id: str
    path: Path
    examples: list[FinetuneExample] = field(default_factory=list)
    domain: str = ""
    format: str = "plain_text_examples"
    raw_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.examples)

    def texts(self) -> list[str]:
        return [ex.text for ex in self.examples]

    def example_ids(self) -> list[str]:
        return [ex.id for ex in self.examples]


def resolve_path(rel_or_abs: str | Path, *, root: Path | None = None) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    base = root or research_root()
    return (base / p).resolve()


def resolve_dataset_path(
    kind: DatasetKind,
    *,
    config: dict[str, Any] | None = None,
    config_path: Path | str | None = None,
) -> Path:
    """Resolve the frozen dataset YAML for ``primary`` or ``control``.

    Selection is explicit: the requested kind's path is taken from
    ``fine_tuning.datasets`` / ``fine_tuning.runs``. No silent cross-fallback.
    """
    if kind not in ("primary", "control"):
        raise FinetuneDataError(
            f"Dataset kind must be 'primary' or 'control', got {kind!r}"
        )
    if config is None:
        from .engram import load_experiment_config

        config = load_experiment_config(config_path)

    ft = config.get("fine_tuning") or {}
    datasets = ft.get("datasets") or {}
    runs = ft.get("runs") or {}

    path_str = datasets.get(kind)
    if not path_str:
        run = runs.get(kind) or {}
        path_str = run.get("dataset")
    if not path_str:
        raise FinetuneDataError(
            f"No dataset path configured for kind={kind!r} under "
            "fine_tuning.datasets / fine_tuning.runs. Refusing silent fallback."
        )

    # Isolation: primary path must not equal control path
    other = "control" if kind == "primary" else "primary"
    other_path = datasets.get(other) or (runs.get(other) or {}).get("dataset")
    if other_path and Path(str(path_str)) == Path(str(other_path)):
        raise FinetuneDataError(
            f"Primary and control dataset paths are identical ({path_str!r}); "
            "refusing to train with ambiguous corpus identity."
        )

    resolved = resolve_path(str(path_str))
    if not resolved.is_file():
        raise FinetuneDataError(f"Frozen FT dataset file not found: {resolved}")
    return resolved


def _validate_example_fields(
    raw: dict[str, Any],
    *,
    kind: DatasetKind,
    index: int,
) -> FinetuneExample:
    required = ("id", "subject", "text")
    for key in required:
        if key not in raw or raw[key] is None or str(raw[key]).strip() == "":
            raise FinetuneDataError(
                f"{kind} example index={index}: missing required field {key!r}"
            )
    text = raw["text"]
    if not isinstance(text, str):
        raise FinetuneDataError(
            f"{kind} example index={index}: text must be str, got {type(text)!r}"
        )
    fact_id = raw.get("fact_id")
    if kind == "primary":
        if not fact_id:
            raise FinetuneDataError(
                f"primary example index={index} id={raw['id']!r}: "
                "fact_id is required"
            )
        if fact_id not in ("F01", "F02"):
            raise FinetuneDataError(
                f"primary example {raw['id']!r}: fact_id must be F01 or F02, "
                f"got {fact_id!r}"
            )
    return FinetuneExample(
        id=str(raw["id"]),
        subject=str(raw["subject"]),
        text=text,  # exact frozen text; no strip/augment
        fact_id=str(fact_id) if fact_id is not None else None,
        index=index,
    )


def load_finetune_dataset(
    kind: DatasetKind,
    *,
    config: dict[str, Any] | None = None,
    config_path: Path | str | None = None,
    path: Path | str | None = None,
) -> FinetuneDataset:
    """Load and validate a frozen primary or control FT dataset.

    Ordering: examples are kept in YAML list order (deterministic).
    Text content is not modified.
    """
    if kind not in ("primary", "control"):
        raise FinetuneDataError(
            f"Dataset kind must be 'primary' or 'control', got {kind!r}"
        )
    if config is None and path is None:
        from .engram import load_experiment_config

        config = load_experiment_config(config_path)

    resolved = (
        resolve_path(path)
        if path is not None
        else resolve_dataset_path(kind, config=config, config_path=config_path)
    )
    data = load_yaml(resolved)
    block = (data or {}).get("dataset") or {}
    dataset_id = str(block.get("id") or "")
    expected_id = EXPECTED_DATASET_IDS[kind]
    if dataset_id != expected_id:
        raise FinetuneDataError(
            f"Dataset identity mismatch for kind={kind!r}: "
            f"expected id={expected_id!r}, got {dataset_id!r} at {resolved}"
        )

    raw_examples: Sequence[Any] = block.get("examples") or []
    if not isinstance(raw_examples, list):
        raise FinetuneDataError(f"{resolved}: dataset.examples must be a list")

    examples: list[FinetuneExample] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_examples):
        if not isinstance(raw, dict):
            raise FinetuneDataError(f"{kind} example index={i}: expected mapping")
        ex = _validate_example_fields(raw, kind=kind, index=i)
        if ex.id in seen_ids:
            raise FinetuneDataError(f"Duplicate example id {ex.id!r} in {resolved}")
        seen_ids.add(ex.id)
        examples.append(ex)

    declared_n = block.get("n_examples")
    if declared_n is not None and int(declared_n) != len(examples):
        raise FinetuneDataError(
            f"{resolved}: n_examples={declared_n} but found {len(examples)} examples"
        )
    if len(examples) != EXPECTED_N_EXAMPLES:
        raise FinetuneDataError(
            f"{resolved}: expected {EXPECTED_N_EXAMPLES} examples, got {len(examples)}"
        )

    return FinetuneDataset(
        kind=kind,
        dataset_id=dataset_id,
        path=resolved,
        examples=examples,
        domain=str(block.get("domain") or ""),
        format=str(block.get("format") or "plain_text_examples"),
        raw_meta={
            k: v
            for k, v in block.items()
            if k != "examples"
        },
    )
