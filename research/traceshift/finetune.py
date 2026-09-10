"""Reusable LoRA supervised fine-tuning pipeline for TraceShift.

Produces (when executed later with local weights):

* primary: Qwen3-1.7B + LoRA on frozen Federer/Nadal corpus
* control: identical procedure on frozen Curie/Armstrong corpus

All hyperparameters are read from ``experiment.yaml`` → ``fine_tuning``.
This module does not invent hparams, download weights, or fabricate results.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import yaml
from torch.utils.data import Dataset

from .engram import load_experiment_config
from .finetune_data import (
    EXPECTED_DATASET_IDS,
    DatasetKind,
    FinetuneDataError,
    FinetuneDataset,
    load_finetune_dataset,
    resolve_dataset_path,
)
from .model_loader import LoadedModel, load_model
from .paths import research_root

# Frozen design expectations (must match experiment.yaml; fail on drift).
FROZEN_BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
FROZEN_METHOD = "lora"
FROZEN_SEED = 42
FROZEN_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
FROZEN_EXPECTED_HPARAMS = {
    "num_epochs": 3,
    "learning_rate": 2.0e-4,
    "optimizer": "adamw",
    "lr_scheduler_type": "cosine",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 256,
    "weight_decay": 0.0,
    "warmup_steps": 5,
}
FROZEN_EXPECTED_LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


class FinetuneConfigError(ValueError):
    """Frozen fine-tuning configuration validation failure."""


class FinetuneModuleError(RuntimeError):
    """Model / LoRA module mismatch (do not auto-adapt targets)."""


@dataclass
class FinetuneRunSpec:
    """Resolved primary or control run (config-driven)."""

    kind: DatasetKind
    base_model_id: str
    dataset_path: Path
    dataset_id: str
    output_dir: Path
    final_adapter_dirname: str
    seed: int
    hyperparameters: dict[str, Any]
    lora: dict[str, Any]
    condition: str
    tokenizer_id: str
    method: str
    peft_library: str
    config_path: str
    identical_hparams_for_primary_and_control: bool
    only_difference_between_runs: str


@dataclass
class TokenizedFinetuneDataset(Dataset):
    """Tokenized causal-LM examples; labels == input_ids (standard SFT)."""

    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    labels: list[list[int]]
    example_ids: list[str]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


@dataclass
class FinetuneMetadata:
    """Machine-readable training run record (written only after a real run)."""

    base_model_id: str
    dataset_kind: str
    dataset_id: str
    dataset_path: str
    seed: int
    hyperparameters: dict[str, Any]
    lora: dict[str, Any]
    tokenizer_id: str
    software: dict[str, str]
    output_checkpoint_path: str
    final_adapter_path: str
    training_start_utc: str
    training_end_utc: str
    training_duration_seconds: float
    n_examples: int
    condition: str
    notes: list[str] = field(default_factory=list)
    # Explicit role marker for the unrelated-FT control experiment (alt hypothesis A).
    unrelated_finetuning_control: bool = False
    experiment_role: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Ensure control runs always carry an explicit marker in serialized metadata.
        if self.dataset_kind == "control" or self.unrelated_finetuning_control:
            d["unrelated_finetuning_control"] = True
            d["experiment_role"] = (
                self.experiment_role
                or "unrelated_fine_tuning_control_alternative_hypothesis_A"
            )
            d["dataset_id_expected"] = "control_general_ft"
        return d


def _software_versions() -> dict[str, str]:
    out: dict[str, str] = {
        "python": f"{__import__('sys').version_info.major}."
        f"{__import__('sys').version_info.minor}."
        f"{__import__('sys').version_info.micro}",
        # torch.__version__ may be TorchVersion (not a plain str); YAML needs str.
        "torch": str(torch.__version__),
    }
    for pkg in ("transformers", "peft", "accelerate", "ai-engram"):
        try:
            out[pkg] = str(importlib.metadata.version(pkg))
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def _resolve_under_research(rel: str | Path) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (research_root() / p).resolve()


def load_finetune_run_spec(
    kind: DatasetKind,
    *,
    config_path: Path | str | None = None,
) -> FinetuneRunSpec:
    """Load and validate frozen FT config for an explicit primary/control run."""
    if kind not in ("primary", "control"):
        raise FinetuneConfigError(
            f"Run kind must be explicitly 'primary' or 'control', got {kind!r}"
        )

    cfg_path = Path(config_path) if config_path else None
    config = load_experiment_config(cfg_path)
    ft = config.get("fine_tuning") or {}
    model_cfg = config.get("model") or {}

    base_model_id = str(
        ft.get("base_model_id")
        or model_cfg.get("base_model_id")
        or model_cfg.get("model_id")
        or ""
    )
    if base_model_id != FROZEN_BASE_MODEL_ID:
        raise FinetuneConfigError(
            f"Frozen model ID mismatch: expected {FROZEN_BASE_MODEL_ID!r}, "
            f"got {base_model_id!r}"
        )
    model_section_id = str(
        model_cfg.get("model_id") or model_cfg.get("base_model_id") or ""
    )
    if model_section_id and model_section_id != base_model_id:
        raise FinetuneConfigError(
            f"model.model_id ({model_section_id!r}) != fine_tuning.base_model_id "
            f"({base_model_id!r})"
        )

    method = str(ft.get("method") or "")
    if method != FROZEN_METHOD:
        raise FinetuneConfigError(
            f"Expected method={FROZEN_METHOD!r}, got {method!r}"
        )

    seed = int(ft.get("seed") if ft.get("seed") is not None else -1)
    if seed != FROZEN_SEED:
        raise FinetuneConfigError(
            f"Frozen seed mismatch: expected {FROZEN_SEED}, got {seed}"
        )

    hparams = dict(ft.get("hyperparameters") or {})
    for key, expected in FROZEN_EXPECTED_HPARAMS.items():
        if key not in hparams:
            raise FinetuneConfigError(f"Missing frozen hyperparameter: {key}")
        actual = hparams[key]
        if isinstance(expected, float):
            if abs(float(actual) - expected) > 1e-12:
                raise FinetuneConfigError(
                    f"Frozen hyperparameter {key}: expected {expected}, got {actual}"
                )
        elif isinstance(expected, int):
            try:
                if int(actual) != expected:
                    raise FinetuneConfigError(
                        f"Frozen hyperparameter {key}: expected {expected}, got {actual}"
                    )
            except (TypeError, ValueError) as exc:
                raise FinetuneConfigError(
                    f"Frozen hyperparameter {key}: expected {expected}, got {actual}"
                ) from exc
        elif actual != expected:
            raise FinetuneConfigError(
                f"Frozen hyperparameter {key}: expected {expected}, got {actual}"
            )

    lora = dict(ft.get("lora") or {})
    for key, expected in FROZEN_EXPECTED_LORA.items():
        if key not in lora:
            raise FinetuneConfigError(f"Missing frozen LoRA field: {key}")
        actual = lora[key]
        if isinstance(expected, float):
            if abs(float(actual) - float(expected)) > 1e-12:
                raise FinetuneConfigError(
                    f"Frozen LoRA {key}: expected {expected}, got {actual}"
                )
        elif actual != expected:
            raise FinetuneConfigError(
                f"Frozen LoRA {key}: expected {expected}, got {actual}"
            )

    targets = list(lora.get("target_modules") or [])
    if targets != list(FROZEN_LORA_TARGETS):
        raise FinetuneConfigError(
            f"Frozen LoRA target_modules must be exactly {list(FROZEN_LORA_TARGETS)}, "
            f"got {targets}. Refusing to auto-adapt."
        )

    runs = ft.get("runs") or {}
    run = runs.get(kind)
    if not isinstance(run, dict):
        raise FinetuneConfigError(f"fine_tuning.runs.{kind} missing")

    outputs = ft.get("outputs") or {}
    out_key = f"{kind}_checkpoint_dir"
    output_rel = run.get("output_dir") or outputs.get(out_key)
    if not output_rel:
        raise FinetuneConfigError(f"No output_dir configured for {kind}")
    output_dir = _resolve_under_research(str(output_rel))

    # Isolation: primary and control output dirs must differ
    other = "control" if kind == "primary" else "primary"
    other_run = runs.get(other) or {}
    other_out = other_run.get("output_dir") or outputs.get(f"{other}_checkpoint_dir")
    if other_out and Path(str(output_rel)).as_posix().rstrip("/") == Path(
        str(other_out)
    ).as_posix().rstrip("/"):
        raise FinetuneConfigError(
            "Primary and control output directories are identical; refusing."
        )

    dataset_path = resolve_dataset_path(kind, config=config)
    # Load lightly for identity (full load also in dry-validate / train)
    ds = load_finetune_dataset(kind, config=config, path=dataset_path)

    tokenizer_id = str(
        model_cfg.get("tokenizer_identifier")
        or model_cfg.get("model_id")
        or base_model_id
    )
    if tokenizer_id in ("null", "None"):
        tokenizer_id = base_model_id

    final_adapter = str(outputs.get("final_adapter_dirname") or "final_adapter")

    return FinetuneRunSpec(
        kind=kind,
        base_model_id=base_model_id,
        dataset_path=dataset_path,
        dataset_id=ds.dataset_id,
        output_dir=output_dir,
        final_adapter_dirname=final_adapter,
        seed=seed,
        hyperparameters=hparams,
        lora=lora,
        condition=str(run.get("condition") or ""),
        tokenizer_id=tokenizer_id,
        method=method,
        peft_library=str(ft.get("peft_library") or "peft"),
        config_path=str(cfg_path or research_root() / "configs" / "experiment.yaml"),
        identical_hparams_for_primary_and_control=bool(
            ft.get("identical_hparams_for_primary_and_control", True)
        ),
        only_difference_between_runs=str(
            ft.get("only_difference_between_runs") or "training_corpus"
        ),
    )


def set_reproducibility_seeds(seed: int) -> None:
    """Set Python / NumPy / Torch seeds. Document CUDA nondeterminism separately."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Prefer deterministic algorithms when available; may error on some ops.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001
            pass
        # cudnn nondeterminism is often unavoidable for speed; document it.
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def verify_lora_target_modules(model: Any, target_modules: Sequence[str]) -> None:
    """Fail if configured LoRA targets are missing; never auto-rewrite the list."""
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    missing = [t for t in target_modules if t not in names]
    if missing:
        raise FinetuneModuleError(
            f"LoRA target modules not found in model: {missing}. "
            f"Configured targets={list(target_modules)}. "
            "Refusing to modify the target-module list."
        )


def tokenize_finetune_dataset(
    dataset: FinetuneDataset,
    tokenizer: Any,
    *,
    max_seq_length: int,
) -> TokenizedFinetuneDataset:
    """Tokenize frozen texts with deterministic truncation; explicit padding.

    - Truncation: ``truncation=True``, ``max_length=max_seq_length``.
    - Padding: pad to ``max_seq_length`` with ``padding='max_length'``.
    - Text is tokenized as-is (no chat template, no rewrite).
    - Labels equal input_ids; pad positions (attention_mask==0) set to -100.
    """
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise FinetuneConfigError(
                "Tokenizer has neither pad_token_id nor eos_token_id"
            )
        tokenizer.pad_token = tokenizer.eos_token

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    example_ids: list[str] = []

    for ex in dataset.examples:
        enc = tokenizer(
            ex.text,
            truncation=True,
            max_length=int(max_seq_length),
            padding="max_length",
            return_tensors=None,
            add_special_tokens=True,
        )
        ids = list(enc["input_ids"])
        mask = list(enc["attention_mask"])
        if len(ids) != int(max_seq_length):
            raise FinetuneConfigError(
                f"Tokenization length mismatch for {ex.id}: "
                f"got {len(ids)}, expected {max_seq_length}"
            )
        lab = [tok if m == 1 else -100 for tok, m in zip(ids, mask, strict=True)]
        input_ids.append(ids)
        attention_mask.append(mask)
        labels.append(lab)
        example_ids.append(ex.id)

    return TokenizedFinetuneDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        example_ids=example_ids,
    )


def _build_lora_config(lora: dict[str, Any]) -> Any:
    from peft import LoraConfig, TaskType

    task = str(lora["task_type"])
    task_type = TaskType[task] if task in TaskType.__members__ else TaskType.CAUSAL_LM
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        bias=str(lora["bias"]),
        task_type=task_type,
        target_modules=list(lora["target_modules"]),
    )


def _build_training_arguments(spec: FinetuneRunSpec) -> Any:
    from transformers import TrainingArguments

    hp = spec.hyperparameters
    optim = str(hp["optimizer"]).lower()
    # HF expects adamw_torch / adamw_hf etc.
    if optim in ("adamw", "adamw_torch"):
        optim_name = "adamw_torch"
    else:
        optim_name = optim

    use_bf16 = bool(hp.get("bf16", True))
    use_fp16 = bool(hp.get("fp16", False))
    # On non-CUDA devices bf16/fp16 mixed precision may be unsupported; Trainer
    # will still receive the config flags — callers should set device appropriately.
    # We do not silently rewrite frozen flags; note in metadata if disabled at runtime.

    return TrainingArguments(
        output_dir=str(spec.output_dir),
        num_train_epochs=int(hp["num_epochs"]),
        learning_rate=float(hp["learning_rate"]),
        optim=optim_name,
        lr_scheduler_type=str(hp["lr_scheduler_type"]),
        per_device_train_batch_size=int(hp["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(hp["gradient_accumulation_steps"]),
        weight_decay=float(hp["weight_decay"]),
        warmup_steps=int(hp["warmup_steps"]),
        max_grad_norm=float(hp.get("max_grad_norm", 1.0)),
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=int(hp.get("logging_steps", 1)),
        save_strategy=str(hp.get("save_strategy", "epoch")),
        eval_strategy=str(hp.get("eval_strategy", "no")),
        seed=int(spec.seed),
        data_seed=int(spec.seed),
        report_to=[],
        remove_unused_columns=False,
        dataloader_drop_last=False,
    )


def write_training_metadata(meta: FinetuneMetadata, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_metadata.yaml"
    path.write_text(
        yaml.safe_dump(meta.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    json_path = output_dir / "training_metadata.json"
    json_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return path


def run_finetune(
    kind: DatasetKind,
    *,
    config_path: Path | str | None = None,
    device: Optional[str] = None,
    loaded: LoadedModel | None = None,
) -> dict[str, Any]:
    """Execute LoRA SFT for ``primary`` or ``control`` (requires local weights).

    Does not download weights. Saves final adapter under the frozen output dir.
    """
    from peft import get_peft_model
    from transformers import Trainer

    # Fail before any training if primary/control symmetry is broken.
    assert_primary_control_symmetry(config_path=config_path)

    spec = load_finetune_run_spec(kind, config_path=config_path)
    dataset = load_finetune_dataset(kind, config_path=config_path, path=spec.dataset_path)
    if kind == "control":
        validate_control_corpus_contents(dataset)
    assert_disjoint_from_held_out(dataset, config_path=config_path)

    if loaded is None:
        loaded = load_model(config_path, device=device, local_files_only=True)
    if loaded.model_id != spec.base_model_id:
        raise FinetuneConfigError(
            f"Loaded model {loaded.model_id!r} != frozen {spec.base_model_id!r}"
        )

    set_reproducibility_seeds(spec.seed)
    verify_lora_target_modules(loaded.model, spec.lora["target_modules"])

    max_len = int(spec.hyperparameters["max_seq_length"])
    tok_ds = tokenize_finetune_dataset(
        dataset, loaded.tokenizer, max_seq_length=max_len
    )

    peft_config = _build_lora_config(spec.lora)
    model = get_peft_model(loaded.model, peft_config)
    model.train()

    if loaded.tokenizer.pad_token_id is None and loaded.tokenizer.eos_token is not None:
        loaded.tokenizer.pad_token = loaded.tokenizer.eos_token

    training_args = _build_training_arguments(spec)
    runtime_notes: list[str] = [
        "Primary and control share identical hparams; only the dataset differs.",
        "CUDA/cuDNN kernel nondeterminism may prevent bitwise-identical reruns "
        "even with fixed seeds.",
    ]
    is_control = kind == "control"
    if is_control:
        runtime_notes.append(
            "EXPERIMENT ROLE: unrelated fine-tuning control "
            "(alternative hypothesis A — generic FT effects). "
            "Dataset is control_general_ft (Marie Curie / Neil Armstrong)."
        )

    if str(loaded.device.type) == "cpu":
        training_args.bf16 = False
        training_args.fp16 = False
        runtime_notes.append(
            "Runtime: disabled bf16/fp16 mixed precision on CPU "
            "(frozen config still records bf16=true)."
        )

    class _HFDataset(Dataset):
        def __len__(self) -> int:
            return len(tok_ds)

        def __getitem__(self, i: int) -> dict[str, Any]:
            item = tok_ds[i]
            return {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
                "labels": item["labels"],
            }

    spec.output_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": _HFDataset(),
    }
    # transformers 5 prefers processing_class; older API used tokenizer=
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=loaded.tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=loaded.tokenizer)

    trainer.train()
    end = datetime.now(timezone.utc)

    final_dir = spec.output_dir / spec.final_adapter_dirname
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    loaded.tokenizer.save_pretrained(str(final_dir))

    meta = FinetuneMetadata(
        base_model_id=spec.base_model_id,
        dataset_kind=kind,
        dataset_id=spec.dataset_id,
        dataset_path=str(spec.dataset_path),
        seed=spec.seed,
        hyperparameters=dict(spec.hyperparameters),
        lora=dict(spec.lora),
        tokenizer_id=spec.tokenizer_id,
        software=_software_versions(),
        output_checkpoint_path=str(spec.output_dir),
        final_adapter_path=str(final_dir),
        training_start_utc=start.isoformat(),
        training_end_utc=end.isoformat(),
        training_duration_seconds=(end - start).total_seconds(),
        n_examples=dataset.n_examples,
        condition=spec.condition,
        notes=runtime_notes,
        unrelated_finetuning_control=is_control,
        experiment_role=(
            "unrelated_fine_tuning_control_alternative_hypothesis_A"
            if is_control
            else "primary_tennis_biography_finetuning"
        ),
    )
    meta_path = write_training_metadata(meta, spec.output_dir)
    return {
        "kind": kind,
        "output_dir": str(spec.output_dir),
        "final_adapter": str(final_dir),
        "metadata": str(meta_path),
        "n_examples": dataset.n_examples,
        "dataset_id": spec.dataset_id,
    }


def _held_out_evaluation_texts(
    config_path: Path | str | None = None,
) -> set[str]:
    """Collect frozen extraction + reference + cloze probe strings (exact text)."""
    from .cloze import load_cloze_spec
    from .engram import FACT_IDS as ENGRAM_FACT_IDS
    from .engram import resolve_extraction_bundle

    cfg = load_experiment_config(config_path)
    texts: set[str] = set()
    for variant in ("explicit", "lexical_control"):
        for fid in ENGRAM_FACT_IDS:
            bundle = resolve_extraction_bundle(fid, variant, cfg=cfg)
            texts.update(bundle.forget_texts)
            texts.update(bundle.total_texts)

    probes = load_cloze_spec()
    block = probes.get("probes") or {}
    for _fid, entry in block.items():
        for item in (entry or {}).get("items") or []:
            prefix = (item or {}).get("prefix") or (item or {}).get("text")
            if isinstance(prefix, str) and prefix.strip():
                texts.add(prefix)
    return texts


def assert_primary_control_symmetry(
    *,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fail before training if primary/control hparams or paths are asymmetric."""
    primary = load_finetune_run_spec("primary", config_path=config_path)
    control = load_finetune_run_spec("control", config_path=config_path)

    if primary.base_model_id != control.base_model_id:
        raise FinetuneConfigError("Primary/control base_model_id mismatch")
    if primary.seed != control.seed:
        raise FinetuneConfigError("Primary/control seed mismatch")
    if primary.hyperparameters != control.hyperparameters:
        raise FinetuneConfigError(
            "Primary/control hyperparameters differ; refusing to train"
        )
    if primary.lora != control.lora:
        raise FinetuneConfigError("Primary/control LoRA configs differ; refusing")
    if primary.method != control.method:
        raise FinetuneConfigError("Primary/control method mismatch")
    if primary.tokenizer_id != control.tokenizer_id:
        raise FinetuneConfigError("Primary/control tokenizer_id mismatch")
    if primary.dataset_path.resolve() == control.dataset_path.resolve():
        raise FinetuneConfigError(
            "Primary and control dataset paths are identical; refusing"
        )
    if primary.output_dir.resolve() == control.output_dir.resolve():
        raise FinetuneConfigError(
            "Primary and control output paths are identical; refusing"
        )
    if primary.dataset_id == control.dataset_id:
        raise FinetuneConfigError("Primary and control dataset IDs are identical")

    return {
        "model_id": {"primary": primary.base_model_id, "control": control.base_model_id, "same": True},
        "seed": {"primary": primary.seed, "control": control.seed, "same": True},
        "epochs": {
            "primary": primary.hyperparameters.get("num_epochs"),
            "control": control.hyperparameters.get("num_epochs"),
            "same": True,
        },
        "learning_rate": {
            "primary": primary.hyperparameters.get("learning_rate"),
            "control": control.hyperparameters.get("learning_rate"),
            "same": True,
        },
        "optimizer": {
            "primary": primary.hyperparameters.get("optimizer"),
            "control": control.hyperparameters.get("optimizer"),
            "same": True,
        },
        "lr_scheduler_type": {
            "primary": primary.hyperparameters.get("lr_scheduler_type"),
            "control": control.hyperparameters.get("lr_scheduler_type"),
            "same": True,
        },
        "per_device_train_batch_size": {
            "primary": primary.hyperparameters.get("per_device_train_batch_size"),
            "control": control.hyperparameters.get("per_device_train_batch_size"),
            "same": True,
        },
        "gradient_accumulation_steps": {
            "primary": primary.hyperparameters.get("gradient_accumulation_steps"),
            "control": control.hyperparameters.get("gradient_accumulation_steps"),
            "same": True,
        },
        "max_seq_length": {
            "primary": primary.hyperparameters.get("max_seq_length"),
            "control": control.hyperparameters.get("max_seq_length"),
            "same": True,
        },
        "lora": {"primary": primary.lora, "control": control.lora, "same": True},
        "dataset": {
            "primary": {
                "id": primary.dataset_id,
                "path": str(primary.dataset_path),
            },
            "control": {
                "id": control.dataset_id,
                "path": str(control.dataset_path),
            },
            "same": False,
        },
        "output_path": {
            "primary": str(primary.output_dir),
            "control": str(control.output_dir),
            "same": False,
        },
    }


def format_comparability_report(comp: dict[str, Any]) -> str:
    """Compact human-readable primary vs control comparison."""
    lines = [
        "=== Primary vs control comparability ===",
        f"model ID:        same ({comp['model_id']['primary']})",
        f"seed:            same ({comp['seed']['primary']})",
        f"epochs:          same ({comp['epochs']['primary']})",
        f"LR:              same ({comp['learning_rate']['primary']})",
        f"optimizer:       same ({comp['optimizer']['primary']})",
        f"scheduler:       same ({comp['lr_scheduler_type']['primary']})",
        f"batch:           same ({comp['per_device_train_batch_size']['primary']})",
        f"grad accum:      same ({comp['gradient_accumulation_steps']['primary']})",
        f"sequence length: same ({comp['max_seq_length']['primary']})",
        "LoRA:            same "
        f"(r={comp['lora']['primary']['r']}, "
        f"alpha={comp['lora']['primary']['alpha']}, "
        f"dropout={comp['lora']['primary']['dropout']}, "
        f"targets={comp['lora']['primary']['target_modules']})",
        f"dataset:         DIFFERENT "
        f"(primary={comp['dataset']['primary']['id']}, "
        f"control={comp['dataset']['control']['id']})",
        f"output path:     DIFFERENT",
        f"  primary → {comp['output_path']['primary']}",
        f"  control → {comp['output_path']['control']}",
    ]
    return "\n".join(lines)


def validate_control_corpus_contents(dataset: FinetuneDataset) -> dict[str, Any]:
    """Control-only content checks (counts, subjects, no tennis entities)."""
    if dataset.kind != "control":
        raise FinetuneDataError("validate_control_corpus_contents requires kind=control")
    if dataset.dataset_id != "control_general_ft":
        raise FinetuneDataError(
            f"Control dataset_id must be control_general_ft, got {dataset.dataset_id!r}"
        )
    if dataset.n_examples != 40:
        raise FinetuneDataError(
            f"Control corpus must have 40 examples, got {dataset.n_examples}"
        )

    curie = [ex for ex in dataset.examples if ex.subject == "Marie Curie"]
    armstrong = [ex for ex in dataset.examples if ex.subject == "Neil Armstrong"]
    if len(curie) != 20 or len(armstrong) != 20:
        raise FinetuneDataError(
            f"Expected 20 Marie Curie + 20 Neil Armstrong, got "
            f"{len(curie)} Curie + {len(armstrong)} Armstrong"
        )

    banned = ("federer", "nadal", "roger federer", "rafael nadal")
    for ex in dataset.examples:
        blob = f"{ex.subject} {ex.text}".lower()
        for term in banned:
            if term in blob:
                raise FinetuneDataError(
                    f"Control example {ex.id!r} contains banned tennis entity "
                    f"term {term!r}"
                )
        if ex.fact_id in ("F01", "F02"):
            raise FinetuneDataError(
                f"Control example {ex.id!r} has tennis fact_id {ex.fact_id!r}"
            )

    return {
        "n_examples": 40,
        "n_marie_curie": len(curie),
        "n_neil_armstrong": len(armstrong),
        "dataset_id": dataset.dataset_id,
        "no_federer_nadal": True,
    }


def assert_disjoint_from_held_out(
    dataset: FinetuneDataset,
    *,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fail if any FT example text exactly matches extraction/reference/cloze text."""
    held = _held_out_evaluation_texts(config_path)
    overlaps = [ex.id for ex in dataset.examples if ex.text in held]
    if overlaps:
        raise FinetuneDataError(
            f"FT examples overlap held-out extraction/probe text: {overlaps[:5]}"
        )
    return {"held_out_text_count": len(held), "exact_text_overlaps": 0}


def validate_finetune_pipeline_without_model(
    kind: DatasetKind,
    *,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Validate config + dataset + preprocessing wiring without weights / training."""
    if kind not in ("primary", "control"):
        raise FinetuneConfigError(
            f"Run kind must be explicitly 'primary' or 'control', got {kind!r}"
        )

    # Explicit mapping check: control → control_finetuning_dataset_spec.yaml only
    resolved = resolve_dataset_path(kind, config_path=config_path)
    if kind == "control":
        expected_name = "control_finetuning_dataset_spec.yaml"
        if resolved.name != expected_name:
            raise FinetuneConfigError(
                f"--dataset control must map to {expected_name}, got {resolved.name}"
            )

    comparability = assert_primary_control_symmetry(config_path=config_path)

    spec = load_finetune_run_spec(kind, config_path=config_path)
    dataset = load_finetune_dataset(kind, config_path=config_path, path=spec.dataset_path)

    other: DatasetKind = "control" if kind == "primary" else "primary"
    other_spec = load_finetune_run_spec(other, config_path=config_path)
    other_ds = load_finetune_dataset(
        other, config_path=config_path, path=other_spec.dataset_path
    )

    # Isolation checks
    if dataset.dataset_id == other_ds.dataset_id:
        raise FinetuneDataError("Primary and control share the same dataset id")
    if dataset.path.resolve() == other_ds.path.resolve():
        raise FinetuneDataError("Primary and control resolve to the same file")
    if set(dataset.example_ids()) & set(other_ds.example_ids()):
        raise FinetuneDataError("Primary and control share example ids")
    if dataset.texts() == other_ds.texts():
        raise FinetuneDataError("Primary and control have identical text sequences")

    held_out_check = assert_disjoint_from_held_out(dataset, config_path=config_path)
    control_contents: dict[str, Any] | None = None
    if kind == "control":
        control_contents = validate_control_corpus_contents(dataset)
        # Also ensure primary tennis subjects are absent from control subject list
        subjects = {ex.subject for ex in dataset.examples}
        if subjects & {"Roger Federer", "Rafael Nadal"}:
            raise FinetuneDataError(
                "Control corpus accidentally includes Federer/Nadal subjects"
            )

    from .model_loader import find_local_snapshot

    weights_local = find_local_snapshot(spec.base_model_id) is not None

    return {
        "status": "pending_model_weights" if not weights_local else "weights_present_not_trained",
        "selected_dataset": kind,
        "dataset_id": dataset.dataset_id,
        "dataset_path": str(dataset.path),
        "example_count": dataset.n_examples,
        "example_id_order_head": dataset.example_ids()[:3],
        "example_id_order_tail": dataset.example_ids()[-3:],
        "expected_dataset_id": EXPECTED_DATASET_IDS[kind],
        "paired_other_dataset": {
            "kind": other,
            "dataset_id": other_ds.dataset_id,
            "example_count": other_ds.n_examples,
            "path": str(other_ds.path),
        },
        "model_id": spec.base_model_id,
        "tokenizer_id": spec.tokenizer_id,
        "method": spec.method,
        "seed": spec.seed,
        "hyperparameters": spec.hyperparameters,
        "lora": spec.lora,
        "output_path": str(spec.output_dir),
        "final_adapter_dirname": spec.final_adapter_dirname,
        "preprocessing": {
            "format": dataset.format,
            "max_seq_length": spec.hyperparameters["max_seq_length"],
            "truncation": "deterministic max_length truncation",
            "padding": "max_length to max_seq_length; labels -100 on pads",
            "text_augmentation": "none",
            "chat_template": "none — plain text causal LM",
            "shared_with_primary": True,
        },
        "identical_hparams_for_primary_and_control": True,
        "only_difference_between_runs": "training_corpus",
        "condition": spec.condition,
        "unrelated_finetuning_control": kind == "control",
        "control_corpus_checks": control_contents,
        "held_out_disjointness": held_out_check,
        "primary_control_comparability": comparability,
        "comparability_report_text": format_comparability_report(comparability),
        "weights_available_locally": weights_local,
        "training_executed": False,
        "software": _software_versions(),
        "nondeterminism_note": (
            "Seeds are set for Python/NumPy/Torch. CUDA/cuDNN kernels may still "
            "introduce nondeterminism; bitwise-identical training is not claimed."
        ),
    }


def write_pending_status(
    kind: DatasetKind | None = None,
    *,
    report: dict[str, Any] | None = None,
) -> Path:
    """Write shared pending STATUS.yaml (does not invent training results).

    For control dry-runs prefer :func:`write_control_status` so primary status
    is not overwritten.
    """
    root = research_root() / "results" / "raw" / "finetuning"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "STATUS.yaml"
    payload: dict[str, Any] = {
        "status": "pending_model_weights",
        "message": (
            "Fine-tuning pipeline is implemented. Execution awaits local "
            f"{FROZEN_BASE_MODEL_ID} weights. No training results have been fabricated."
        ),
        "model_id": FROZEN_BASE_MODEL_ID,
        "method": FROZEN_METHOD,
        "runs": {
            "primary": {
                "dataset_id": EXPECTED_DATASET_IDS["primary"],
                "output": "results/raw/finetuning/qwen3-1.7b-lora-tennis-primary/",
            },
            "control": {
                "dataset_id": EXPECTED_DATASET_IDS["control"],
                "output": "results/raw/finetuning/qwen3-1.7b-lora-control/",
            },
        },
        "dry_validate_kind": kind,
        "training_executed": False,
    }
    if report is not None and kind != "control":
        payload["last_dry_validate"] = {
            k: report[k]
            for k in (
                "selected_dataset",
                "dataset_id",
                "example_count",
                "output_path",
                "weights_available_locally",
                "training_executed",
            )
            if k in report
        }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def write_control_status(
    *,
    report: dict[str, Any] | None = None,
) -> Path:
    """Write control-only pending status (does not overwrite primary STATUS.yaml)."""
    root = research_root() / "results" / "raw" / "finetuning"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "control_status.yaml"
    payload: dict[str, Any] = {
        "status": "pending_not_executed",
        "experiment_role": "unrelated_fine_tuning_control_alternative_hypothesis_A",
        "unrelated_finetuning_control": True,
        "message": (
            "Unrelated fine-tuning control path is implemented and dry-validated. "
            "Control LoRA training has NOT been executed. No checkpoint or "
            "training metrics have been fabricated."
        ),
        "model_id": FROZEN_BASE_MODEL_ID,
        "dataset_id": EXPECTED_DATASET_IDS["control"],
        "dataset_path": "data/frozen/control_finetuning_dataset_spec.yaml",
        "expected_checkpoint_dir": (
            "results/raw/finetuning/qwen3-1.7b-lora-control/"
        ),
        "final_adapter_dirname": "final_adapter",
        "training_executed": False,
        "checkpoint_present": False,
        "same_procedure_as_primary": True,
        "only_difference": "training_corpus",
    }
    if report is not None:
        payload["last_dry_validate"] = {
            k: report[k]
            for k in (
                "selected_dataset",
                "dataset_id",
                "example_count",
                "output_path",
                "control_corpus_checks",
                "held_out_disjointness",
                "weights_available_locally",
                "training_executed",
                "unrelated_finetuning_control",
            )
            if k in report
        }
    # Detect accidental fabricated checkpoint presence without claiming success
    adapter = (
        research_root()
        / "results"
        / "raw"
        / "finetuning"
        / "qwen3-1.7b-lora-control"
        / "final_adapter"
        / "adapter_config.json"
    )
    payload["checkpoint_present"] = adapter.is_file()
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
