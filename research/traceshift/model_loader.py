"""Reusable Qwen loader driven by frozen ``experiment.yaml`` (no hard-coded model id)."""

from __future__ import annotations

import importlib.metadata
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .paths import default_experiment_config, research_root
from .yaml_io import load_yaml


class ModelWeightsUnavailableError(FileNotFoundError):
    """Raised when frozen model weights are not present locally (no download attempted)."""


def discover_hf_hub_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_HOME"):
        val = os.environ.get(key)
        if not val:
            continue
        p = Path(val)
        if key == "HF_HOME":
            p = p / "hub"
        roots.append(p)
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen: set[Path] = set()
    out: list[Path] = []
    for r in roots:
        key = r.resolve() if r.exists() else r
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def find_local_snapshot(model_id: str) -> Optional[Path]:
    """Return a local HF snapshot path if config + weights exist; never downloads."""
    dirname = "models--" + model_id.replace("/", "--")
    for hub in discover_hf_hub_roots():
        model_dir = hub / dirname
        snaps = model_dir / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in sorted(snaps.iterdir()):
            if not snap.is_dir():
                continue
            has_config = (snap / "config.json").is_file()
            has_weights = any(
                True
                for pat in (
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model.safetensors.index.json",
                    "pytorch_model.bin.index.json",
                )
                if list(snap.glob(pat))
            )
            if has_config and has_weights:
                return snap
    return None


def resolve_device(prefer: Optional[str] = None) -> torch.device:
    """Pick CUDA → MPS → CPU unless ``prefer`` forces a device.

    Warning: if ``prefer`` is omitted and CUDA is unavailable, this falls back
    to MPS/CPU silently. For paid GPU runs, pass ``device='cuda'`` explicitly
    so a missing CUDA stack fails rather than training/scoring on CPU.
    """
    if prefer:
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but torch.cuda.is_available() is False. "
                "Refusing silent CPU fallback for an explicit CUDA request."
            )
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Map frozen dtype name to torch.dtype; fall back if unsupported on device."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(str(name).lower())
    if dtype is None:
        raise ValueError(f"Unsupported dtype in config: {name!r}")
    # Some CPU builds lack efficient bf16; keep requested dtype when possible.
    if device.type == "cpu" and dtype == torch.bfloat16:
        # torch CPU supports bf16 tensors on recent versions; keep as configured.
        return dtype
    return dtype


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    model_id: str
    tokenizer_id: str
    device: torch.device
    dtype: torch.dtype
    source_path: str
    config_path: str
    software: dict[str, str] = field(default_factory=dict)

    def eval_mode(self) -> None:
        self.model.eval()


def load_experiment_model_section(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else default_experiment_config()
    cfg = load_yaml(path)
    model_cfg = cfg.get("model") or {}
    ae = cfg.get("ai_engram") or {}
    precision = ae.get("precision") or {}
    return {
        "config_path": str(path),
        "model_id": model_cfg.get("model_id") or model_cfg.get("base_model_id"),
        "tokenizer_id": model_cfg.get("tokenizer_identifier") or model_cfg.get("model_id"),
        "revision": model_cfg.get("revision"),
        "model_weights_dtype": precision.get("model_weights", "bfloat16"),
        "raw": cfg,
    }


def load_model(
    config_path: Path | str | None = None,
    *,
    device: Optional[str] = None,
    local_files_only: bool = True,
    allow_download: bool = False,
) -> LoadedModel:
    """Load the frozen TraceShift base model + tokenizer.

    Parameters
    ----------
    config_path:
        Path to ``experiment.yaml``. Model id is read from config (not hard-coded).
    device:
        Optional device string (``cuda``, ``mps``, ``cpu``). Auto-selected if omitted.
    local_files_only:
        If True (default), never hit the network for weights.
    allow_download:
        Must stay False for local engineering validation. Raising this requires
        an explicit caller choice; default refuses downloads.
    """
    if allow_download:
        raise ValueError(
            "allow_download=True is disabled in this engineering layer. "
            "Download weights only in an explicit later GPU-prep task."
        )

    section = load_experiment_model_section(config_path)
    model_id = section["model_id"]
    if not model_id:
        raise ValueError(
            f"No model_id in {section['config_path']}. "
            "Set model.model_id in experiment.yaml."
        )
    tokenizer_id = section["tokenizer_id"] or model_id
    if tokenizer_id is None or str(tokenizer_id).lower() == "null":
        tokenizer_id = model_id

    snap = find_local_snapshot(str(model_id))
    if snap is None:
        hub_hint = ", ".join(str(r) for r in discover_hf_hub_roots())
        raise ModelWeightsUnavailableError(
            f"Local weights for {model_id!r} were not found under Hugging Face cache "
            f"({hub_hint}). Expected directory name "
            f"'models--{str(model_id).replace('/', '--')}'. "
            "No download was attempted. Place or download the weights in a later "
            "GPU-prep step, then re-run."
        )

    torch_device = resolve_device(device)
    dtype = resolve_dtype(section["model_weights_dtype"], torch_device)

    tok_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    model_kwargs: dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    revision = section.get("revision")
    if revision not in (None, "null"):
        tok_kwargs["revision"] = revision
        model_kwargs["revision"] = revision

    # Load from local snapshot path so id is still config-driven but files are local.
    # On CUDA, place weights directly on GPU (device_map) to avoid a full CPU-resident
    # materialization that OOMs ~15 GiB host RAM on g2-standard-4. Scientific
    # config/dtype/model_id unchanged.
    tokenizer = AutoTokenizer.from_pretrained(str(snap), **tok_kwargs)
    if torch_device.type == "cuda":
        model_kwargs["device_map"] = "cuda"
        model = AutoModelForCausalLM.from_pretrained(str(snap), **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(str(snap), **model_kwargs)
        model.to(torch_device)
    model.eval()

    software = {
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
    }
    try:
        software["ai-engram"] = importlib.metadata.version("ai-engram")
    except importlib.metadata.PackageNotFoundError:
        software["ai-engram"] = "not-installed"

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        model_id=str(model_id),
        tokenizer_id=str(tokenizer_id),
        device=torch_device,
        dtype=dtype,
        source_path=str(snap),
        config_path=section["config_path"],
        software=software,
    )


def reproducibility_record(loaded: LoadedModel) -> dict[str, Any]:
    return {
        "model_id": loaded.model_id,
        "tokenizer_id": loaded.tokenizer_id,
        "device": str(loaded.device),
        "dtype": str(loaded.dtype).replace("torch.", ""),
        "source_path": loaded.source_path,
        "config_path": loaded.config_path,
        "research_root": str(research_root()),
        "software": loaded.software,
    }
