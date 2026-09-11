"""AI-Engram extraction / intervention layer for TraceShift (v0.9.0).

Collect-once Engrams via ``get_engram``; reuse with ``apply_engram`` for alpha
sweeps. Builds AI-Engram-derived traces (fact-conditioned parameter projections)
— not literal memory locations, and not raw base−FT weight differences.

This module does not implement alpha calibration or response matrices.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional, Sequence

import torch

from .paths import default_experiment_config, research_root
from .yaml_io import load_yaml

ExtractionVariant = Literal["explicit", "lexical_control"]
FACT_IDS = ("F01", "F02", "F03", "F04", "F05", "F06")
EXPECTED_ENGRAM_VERSION = "0.9.0"
EXPECTED_MODEL_ID = "Qwen/Qwen3-1.7B"


class EngramConfigError(ValueError):
    """Frozen config / data resolution failure."""


class EngramVariantMismatchError(EngramConfigError):
    """Target set and reference corpus variants do not match."""


class EngramVersionError(EngramConfigError):
    """Installed or configured AI-Engram version is not the frozen 0.9.0."""


@dataclass(frozen=True)
class ExtractionBundle:
    """Resolved forget/total text for one fact and one extraction variant."""

    fact_id: str
    variant: ExtractionVariant
    forget_texts: list[str]
    forget_ids: list[str]
    total_texts: list[str]
    total_doc_ids: list[str]
    target_set_path: str
    reference_corpus_path: str
    reference_corpus_id: str


@dataclass
class ExtractedEngram:
    """Structured result of a single AI-Engram collection (alpha-free)."""

    fact_id: str
    variant: ExtractionVariant
    model_id: str
    engram: Any  # engram.scaling.EngramResult — not serialized here
    target_modules: list[str]
    layers_to_transform: list[int]
    peft_target_modules: list[str]
    forget_ids: list[str]
    total_doc_ids: list[str]
    reference_corpus_id: str
    target_set_path: str
    reference_corpus_path: str
    ai_engram_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceVector:
    """AI-Engram-derived trace: concatenated float32 projections over target modules.

    This is a fact-conditioned parameter projection estimate, not a literal
    isolated memory location and not a raw base-vs-FT weight difference.
    """

    fact_id: str
    variant: ExtractionVariant
    model_id: str
    module_names: list[str]
    vector: torch.Tensor  # 1-D float32 on CPU
    per_module_numel: dict[str, int]
    dtype: str = "float32"
    representation: str = (
        "concatenated_vectorized_engram_projections_across_target_modules"
    )


def _research_path(rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return research_root() / p


def load_experiment_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else default_experiment_config()
    return load_yaml(path)


def installed_ai_engram_version() -> str:
    return importlib.metadata.version("ai-engram")


def require_ai_engram_version(expected: str = EXPECTED_ENGRAM_VERSION) -> str:
    ver = installed_ai_engram_version()
    if ver != expected:
        raise EngramVersionError(
            f"Installed ai-engram=={ver!r} but TraceShift freezes {expected!r}."
        )
    return ver


def get_frozen_target_module_spec(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_experiment_config()
    ae = cfg.get("ai_engram") or {}
    tm = ae.get("target_modules") or {}
    modules = list(tm.get("modules") or [])
    layers = list(tm.get("layers_to_transform") or [])
    peft = list(tm.get("peft_style_suffix") or ["down_proj"])
    if not modules:
        raise EngramConfigError("ai_engram.target_modules.modules missing in experiment.yaml")
    return {
        "modules": modules,
        "layers_to_transform": layers,
        "peft_style_suffix": peft,
        "model_id": (cfg.get("model") or {}).get("model_id")
        or (ae.get("model_id")),
        "version": str(ae.get("version")),
        "precision": dict(ae.get("precision") or {}),
    }


def _variant_paths(cfg: dict[str, Any], variant: ExtractionVariant) -> tuple[Path, Path, str]:
    ae = cfg.get("ai_engram") or {}
    paths = (ae.get("extraction") or {}).get("paths") or {}
    datasets = cfg.get("datasets") or {}

    if variant == "explicit":
        target_rel = paths.get("targets_explicit") or datasets.get(
            "extraction_sets_explicit"
        )
        ref_key = "explicit"
    elif variant == "lexical_control":
        target_rel = paths.get("targets_lexical") or datasets.get(
            "extraction_sets_lexical"
        )
        ref_key = "lexical_control"
    else:
        raise EngramConfigError(
            f"Unknown extraction variant {variant!r}; "
            "use 'explicit' or 'lexical_control'."
        )

    ref_rel = paths.get("reference_corpus") or datasets.get("reference_corpus")
    if not target_rel or not ref_rel:
        raise EngramConfigError("Frozen extraction/reference paths missing in config")
    return _research_path(str(target_rel)), _research_path(str(ref_rel)), ref_key


def resolve_extraction_bundle(
    fact_id: str,
    variant: ExtractionVariant,
    *,
    config_path: Path | str | None = None,
    cfg: dict[str, Any] | None = None,
) -> ExtractionBundle:
    """Resolve forget (5 sentences) + total (shared reference) for one fact/variant.

    Raises EngramVariantMismatchError if the reference corpus variant does not
    match the requested extraction variant.
    """
    if fact_id not in FACT_IDS:
        raise EngramConfigError(f"Unknown fact_id {fact_id!r}; expected one of {FACT_IDS}")
    if variant not in ("explicit", "lexical_control"):
        raise EngramConfigError(
            f"variant must be 'explicit' or 'lexical_control', got {variant!r}"
        )

    cfg = cfg or load_experiment_config(config_path)
    target_path, ref_path, ref_key = _variant_paths(cfg, variant)
    targets = load_yaml(target_path)
    reference = load_yaml(ref_path)

    sets = (targets.get("sets") or {}).get(fact_id)
    if not isinstance(sets, dict):
        raise EngramConfigError(f"{target_path}: missing sets.{fact_id}")
    kind = sets.get("kind")
    if kind is not None and kind != variant:
        raise EngramVariantMismatchError(
            f"Target set kind {kind!r} does not match requested variant {variant!r}"
        )

    items = sets.get("items") or []
    if len(items) != 5:
        raise EngramConfigError(
            f"{fact_id}/{variant}: expected 5 target sentences, got {len(items)}"
        )
    forget_texts: list[str] = []
    forget_ids: list[str] = []
    for it in items:
        text = (it or {}).get("text")
        iid = (it or {}).get("id")
        if not text or not str(text).strip():
            raise EngramConfigError(f"{fact_id}/{variant}: empty target sentence {iid!r}")
        forget_texts.append(str(text))
        forget_ids.append(str(iid))

    corpora = reference.get("corpora") or {}
    # Enforce variant isolation: only the matching corpus key is allowed.
    if ref_key not in corpora:
        raise EngramConfigError(f"{ref_path}: missing corpora.{ref_key}")
    wrong = "lexical_control" if variant == "explicit" else "explicit"
    # Caller must not pass mismatched pairs — we only read the matching key.
    corpus = corpora[ref_key]
    corpus_variant = corpus.get("variant")
    if corpus_variant and corpus_variant != variant:
        raise EngramVariantMismatchError(
            f"Reference corpus variant {corpus_variant!r} != requested {variant!r}"
        )
    if wrong in corpora and corpora[wrong].get("variant") == variant:
        raise EngramVariantMismatchError(
            f"Internal inconsistency: corpora.{wrong} claims variant={variant!r}"
        )

    docs = corpus.get("documents") or []
    if not docs:
        raise EngramConfigError(f"{ref_path} corpora.{ref_key}: empty documents")
    total_texts: list[str] = []
    total_ids: list[str] = []
    for d in docs:
        text = (d or {}).get("text")
        did = (d or {}).get("doc_id")
        if not text or not str(text).strip():
            raise EngramConfigError(f"Empty reference document {did!r}")
        total_texts.append(str(text))
        total_ids.append(str(did))

    return ExtractionBundle(
        fact_id=fact_id,
        variant=variant,
        forget_texts=forget_texts,
        forget_ids=forget_ids,
        total_texts=total_texts,
        total_doc_ids=total_ids,
        target_set_path=str(target_path),
        reference_corpus_path=str(ref_path),
        reference_corpus_id=str(corpus.get("id") or ref_key),
    )


def assert_variant_isolation(
    forget_variant: ExtractionVariant,
    reference_variant: ExtractionVariant,
) -> None:
    if forget_variant != reference_variant:
        raise EngramVariantMismatchError(
            f"Mismatched variants: target={forget_variant!r} "
            f"reference={reference_variant!r}. "
            "Explicit and lexical_control must not be mixed."
        )


def _module_exists(model: Any, dotted: str) -> bool:
    """Resolve ``model.layers.N.mlp.down_proj`` on a CausalLM (with/without inner .model)."""
    bases = []
    if hasattr(model, "model"):
        bases.append(model.model)
    bases.append(model)
    for base in bases:
        obj = base
        parts = dotted.split(".")
        if hasattr(base, "layers") and parts and parts[0] == "model":
            parts = parts[1:]
        try:
            for part in parts:
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            return obj is not None
        except Exception:
            continue
    return False


def validate_pre_extraction(
    *,
    model: Any | None,
    model_id: str | None,
    fact_id: str,
    variant: ExtractionVariant,
    config_path: Path | str | None = None,
) -> tuple[dict[str, Any], ExtractionBundle, dict[str, Any]]:
    """Validate config, package version, texts, and (if model given) module paths."""
    require_ai_engram_version(EXPECTED_ENGRAM_VERSION)
    cfg = load_experiment_config(config_path)
    spec = get_frozen_target_module_spec(cfg)
    frozen_model = spec["model_id"]
    if frozen_model != EXPECTED_MODEL_ID:
        raise EngramConfigError(
            f"Frozen model_id is {frozen_model!r}, expected {EXPECTED_MODEL_ID!r}"
        )
    if model_id is not None and model_id != frozen_model:
        raise EngramConfigError(
            f"Caller model_id {model_id!r} does not match frozen {frozen_model!r}"
        )
    if str(spec["version"]) != EXPECTED_ENGRAM_VERSION:
        raise EngramConfigError(
            f"Config ai_engram.version={spec['version']!r}, "
            f"expected {EXPECTED_ENGRAM_VERSION!r}"
        )

    bundle = resolve_extraction_bundle(fact_id, variant, cfg=cfg)
    assert_variant_isolation(bundle.variant, variant)
    if not bundle.forget_texts or not bundle.total_texts:
        raise EngramConfigError("Resolved forget/total text is empty")

    if model is not None:
        missing = [m for m in spec["modules"] if not _module_exists(model, m)]
        if missing:
            raise EngramConfigError(
                f"{len(missing)} configured target modules missing on model; "
                f"examples: {missing[:3]}"
            )

    return cfg, bundle, spec


def extract_engram(
    model: Any,
    tokenizer: Any,
    fact_id: str,
    variant: ExtractionVariant,
    *,
    config_path: Path | str | None = None,
    model_id: str | None = None,
    max_length: int = 512,
    batch_size: int = 8,
) -> ExtractedEngram:
    """Collect an AI-Engram once for ``fact_id`` / ``variant`` (no alpha).

    Uses ``engram.get_engram(model, tokenizer, forget=..., total=..., ...)``.
    Caller supplies an already-loaded model; this function does not load weights.
    """
    from engram import get_engram  # installed ai-engram==0.9.0
    from engram.config import EditorConfig

    if model_id is None:
        model_id = EXPECTED_MODEL_ID
    cfg, bundle, spec = validate_pre_extraction(
        model=model,
        model_id=model_id,
        fact_id=fact_id,
        variant=variant,
        config_path=config_path,
    )

    peft_mods = list(spec["peft_style_suffix"])
    layers = list(spec["layers_to_transform"]) or list(range(28))
    modules = list(spec["modules"])

    result = get_engram(
        model,
        tokenizer,
        forget=bundle.forget_texts,
        total=bundle.total_texts,
        target_modules=peft_mods,
        layers_to_transform=layers,
        max_length=max_length,
        batch_size=batch_size,
        # Engineering memory placement only: accumulate D×D covariances on CPU so
        # GPU can hold the model (+ temporary intervention copy) on ~24 GiB L4.
        # Same AI-Engram 0.9.0 math / frozen forget/total / modules; not a science change.
        config=EditorConfig(storage_device="cpu"),
    )

    # Keep EngramResult tensors on CPU to avoid retaining large GPU allocations
    # across sequential matrix rows.
    layers_dict = getattr(result, "layers", None)
    if isinstance(layers_dict, dict):
        for _name, info in layers_dict.items():
            for attr in ("projection", "weight", "u", "inv_lam"):
                t = getattr(info, attr, None)
                if torch.is_tensor(t) and t.device.type == "cuda":
                    setattr(info, attr, t.detach().to("cpu"))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ExtractedEngram(
        fact_id=bundle.fact_id,
        variant=bundle.variant,
        model_id=model_id,
        engram=result,
        target_modules=modules,
        layers_to_transform=layers,
        peft_target_modules=peft_mods,
        forget_ids=bundle.forget_ids,
        total_doc_ids=bundle.total_doc_ids,
        reference_corpus_id=bundle.reference_corpus_id,
        target_set_path=bundle.target_set_path,
        reference_corpus_path=bundle.reference_corpus_path,
        ai_engram_version=EXPECTED_ENGRAM_VERSION,
        metadata={
            "n_forget": len(bundle.forget_texts),
            "n_total": len(bundle.total_texts),
            "precision_projection": (spec["precision"] or {}).get(
                "engram_projection_and_intervention_arithmetic", "float32"
            ),
            "layer_names_in_result": list(getattr(result, "layers", {}).keys()),
        },
    )


def engram_to_trace(
    extracted: ExtractedEngram,
    *,
    module_order: Sequence[str] | None = None,
) -> TraceVector:
    """Convert an extracted Engram into the TraceShift concatenated projection vector.

    For each configured target module present in ``EngramResult.layers``, flatten
    ``projection`` to float32 on CPU and concatenate in stable module order
    (frozen config order by default). Consistent with the prior experiment's
    vectorized projection representation.
    """
    order = list(module_order) if module_order is not None else list(extracted.target_modules)
    layers = getattr(extracted.engram, "layers", None)
    if not isinstance(layers, dict) or not layers:
        raise EngramConfigError("EngramResult has no layers to vectorize")

    # Map possible key styles: full name vs suffix-only.
    def _lookup(name: str) -> Any:
        if name in layers:
            return layers[name]
        # try without leading 'model.'
        alt = name[len("model.") :] if name.startswith("model.") else f"model.{name}"
        if alt in layers:
            return layers[alt]
        # match by suffix
        for k, v in layers.items():
            if k.endswith(name) or name.endswith(k):
                return v
        return None

    pieces: list[torch.Tensor] = []
    used: list[str] = []
    per_numel: dict[str, int] = {}
    missing: list[str] = []
    for name in order:
        info = _lookup(name)
        if info is None:
            missing.append(name)
            continue
        proj = info.projection.detach().to("cpu", dtype=torch.float32).reshape(-1)
        pieces.append(proj)
        used.append(name)
        per_numel[name] = int(proj.numel())

    if missing and not pieces:
        raise EngramConfigError(
            f"None of the configured modules found in EngramResult.layers; "
            f"result keys sample={list(layers)[:3]}; missing e.g. {missing[:3]}"
        )
    if missing:
        # Partial match is an error for TraceShift (need full configured stack).
        raise EngramConfigError(
            f"EngramResult missing {len(missing)} configured modules "
            f"(e.g. {missing[:3]}). Refusing partial trace."
        )

    vector = torch.cat(pieces, dim=0)
    return TraceVector(
        fact_id=extracted.fact_id,
        variant=extracted.variant,
        model_id=extracted.model_id,
        module_names=used,
        vector=vector,
        per_module_numel=per_numel,
        dtype="float32",
    )


def apply_intervention(
    model: Any,
    engram: Any,
    alpha: float,
    *,
    inplace: bool = False,
    scale: Any = None,
) -> Any:
    """Apply an already-collected Engram at ``alpha`` (no recollect).

    Uses ``engram.apply_engram``. Default ``inplace=False`` returns an edited
    copy so the caller's canonical model is not permanently mutated. Pass a
    fresh model/copy explicitly when composing multiple interventions.
    Alpha is supplied by the caller — not hard-coded here.
    """
    from engram import apply_engram  # installed ai-engram==0.9.0

    require_ai_engram_version(EXPECTED_ENGRAM_VERSION)
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"alpha must be a number, got {type(alpha)!r}")
    kwargs: dict[str, Any] = {"alpha": float(alpha), "inplace": bool(inplace)}
    if scale is not None:
        kwargs["scale"] = scale
    return apply_engram(model, engram, **kwargs)


def apply_extracted_intervention(
    model: Any,
    extracted: ExtractedEngram,
    alpha: float,
    *,
    inplace: bool = False,
) -> Any:
    """Convenience wrapper around :func:`apply_intervention` for ExtractedEngram."""
    return apply_intervention(
        model, extracted.engram, alpha, inplace=inplace
    )


def reextract_engram(
    model: Any,
    tokenizer: Any,
    fact_id: str,
    variant: ExtractionVariant,
    **kwargs: Any,
) -> ExtractedEngram:
    """Re-extract after intervention using the same frozen texts/API as extract_engram.

    Base and fine-tuned models must call this with identical ``fact_id`` / ``variant``
    so extraction behavior stays matched across model stages.
    """
    return extract_engram(model, tokenizer, fact_id, variant, **kwargs)


def validate_without_model(config_path: Path | str | None = None) -> dict[str, Any]:
    """Pure config/API validation (no weights, no get_engram execution)."""
    import inspect

    from engram import apply_engram, get_engram

    ver = require_ai_engram_version()
    cfg = load_experiment_config(config_path)
    spec = get_frozen_target_module_spec(cfg)

    bundles = {}
    for variant in ("explicit", "lexical_control"):
        for fact_id in FACT_IDS:
            b = resolve_extraction_bundle(fact_id, variant, cfg=cfg)  # type: ignore[arg-type]
            assert_variant_isolation(b.variant, variant)  # type: ignore[arg-type]
            bundles[f"{fact_id}:{variant}"] = {
                "n_forget": len(b.forget_texts),
                "n_total": len(b.total_texts),
                "reference_corpus_id": b.reference_corpus_id,
                "forget_ids": b.forget_ids,
            }

    # Mismatch must error
    mismatch_ok = False
    try:
        assert_variant_isolation("explicit", "lexical_control")  # type: ignore[arg-type]
    except EngramVariantMismatchError:
        mismatch_ok = True

    return {
        "ai_engram_version": ver,
        "get_engram_signature": str(inspect.signature(get_engram)),
        "apply_engram_signature": str(inspect.signature(apply_engram)),
        "frozen_model_id": spec["model_id"],
        "n_target_modules": len(spec["modules"]),
        "layers_to_transform": spec["layers_to_transform"],
        "peft_style_suffix": spec["peft_style_suffix"],
        "precision": spec["precision"],
        "bundles_resolved": len(bundles),
        "bundle_example_F01_explicit": bundles["F01:explicit"],
        "variant_mismatch_raises": mismatch_ok,
        "model_extraction_executed": False,
    }


def discover_api_signatures() -> dict[str, str]:
    import inspect

    from engram import apply_engram, get_engram

    return {
        "get_engram": str(inspect.signature(get_engram)),
        "apply_engram": str(inspect.signature(apply_engram)),
        "version": installed_ai_engram_version(),
    }
