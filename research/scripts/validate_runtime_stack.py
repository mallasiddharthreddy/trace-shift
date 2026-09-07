#!/usr/bin/env python3
"""TraceShift pre-GPU runtime validation (Qwen3-1.7B + AI-Engram 0.9.0).

Engineering check only — not a research result. Does not fine-tune, sweep alpha,
build matrices, install packages, or download model weights.

Usage:
  python scripts/validate_runtime_stack.py
  python scripts/validate_runtime_stack.py --research-root /path/to/research
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


EXPECTED_MODEL_ID = "Qwen/Qwen3-1.7B"
EXPECTED_ENGRAM_VERSION = "0.9.0"
EXPECTED_ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
EXPECTED_MODULES = [f"model.layers.{i}.mlp.down_proj" for i in range(28)]
HF_HUB_DIRNAME = "models--Qwen--Qwen3-1.7B"


@dataclass
class CheckResult:
    name: str
    status: str  # verified | pending | failed
    detail: str = ""


@dataclass
class Report:
    checks: list[CheckResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.checks.append(CheckResult(name, status, detail))
        if status == "failed":
            self.blockers.append(f"{name}: {detail}" if detail else name)

    def pending(self, name: str, detail: str) -> None:
        self.add(name, "pending", detail)

    def verified(self, name: str, detail: str = "") -> None:
        self.add(name, "verified", detail)

    def failed(self, name: str, detail: str) -> None:
        self.add(name, "failed", detail)

    @property
    def has_failure(self) -> bool:
        return any(c.status == "failed" for c in self.checks)


def _pkg_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _try_import(module_name: str) -> tuple[Optional[Any], Optional[str]]:
    try:
        return importlib.import_module(module_name), None
    except Exception as exc:  # noqa: BLE001 — report any import failure
        return None, f"{type(exc).__name__}: {exc}"


def discover_hf_cache_roots() -> list[Path]:
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
    # de-dupe preserving order
    seen = set()
    out = []
    for r in roots:
        rp = r.resolve() if r.exists() else r
        if rp not in seen:
            seen.add(rp)
            out.append(r)
    return out


def local_model_snapshot(model_id: str = EXPECTED_MODEL_ID) -> Optional[Path]:
    """Return a local snapshot path if weights appear fully present; else None.

    Never triggers a download.
    """
    dirname = "models--" + model_id.replace("/", "--")
    for hub in discover_hf_cache_roots():
        model_dir = hub / dirname
        if not model_dir.is_dir():
            continue
        snaps = model_dir / "snapshots"
        if not snaps.is_dir():
            continue
        for snap in sorted(snaps.iterdir()):
            if not snap.is_dir():
                continue
            # Heuristic: config + at least one weight shard/index present
            has_config = (snap / "config.json").is_file()
            has_weights = any(
                snap.glob(pat)
                for pat in (
                    "model.safetensors",
                    "pytorch_model.bin",
                    "model.safetensors.index.json",
                    "pytorch_model.bin.index.json",
                )
            )
            if has_config and has_weights:
                return snap
    return None


def load_yaml_via_frozen_validator(path: Path) -> Any:
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from validate_frozen_experiment import load_yaml  # type: ignore

    return load_yaml(path)


def check_environment(report: Report) -> dict[str, Any]:
    report.info["python_version"] = sys.version.split()[0]
    report.info["python_executable"] = sys.executable
    print("=== 1. Environment discovery ===")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")

    versions: dict[str, Optional[str]] = {}
    for dist, mod in (
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("ai-engram", "engram"),
    ):
        ver = _pkg_version(dist)
        versions[dist] = ver
        mod_obj, err = _try_import(mod)
        if ver is None and mod_obj is None:
            print(f"  {dist}: NOT INSTALLED ({err})")
            report.pending(f"dep.{dist}", f"not installed ({err})")
        elif ver is None and mod_obj is not None:
            mv = getattr(mod_obj, "__version__", "unknown")
            print(f"  {dist}: importable as {mod} (version metadata missing; module says {mv})")
            report.pending(f"dep.{dist}", f"importable but distribution metadata missing ({mv})")
        else:
            print(f"  {dist}: {ver}")
            report.verified(f"dep.{dist}", ver or "")

    report.info["package_versions"] = versions

    torch_mod, torch_err = _try_import("torch")
    cuda_avail = False
    cuda_detail = "torch not installed"
    if torch_mod is not None:
        try:
            cuda_avail = bool(torch_mod.cuda.is_available())
            n = int(torch_mod.cuda.device_count()) if cuda_avail else 0
            cuda_detail = f"available={cuda_avail}, device_count={n}"
            if cuda_avail:
                try:
                    name = torch_mod.cuda.get_device_name(0)
                    cuda_detail += f", device0={name}"
                except Exception:  # noqa: BLE001
                    pass
        except Exception as exc:  # noqa: BLE001
            cuda_detail = f"cuda probe failed: {exc}"
            report.failed("cuda.probe", str(exc))
    else:
        report.pending("cuda", f"cannot probe ({torch_err})")

    report.info["cuda_available"] = cuda_avail
    report.info["cuda_detail"] = cuda_detail
    print(f"  CUDA: {cuda_detail}")
    if torch_mod is not None and not report.has_failure:
        report.verified("cuda.probe", cuda_detail)

    return {
        "versions": versions,
        "torch": torch_mod,
        "cuda_available": cuda_avail,
    }


def check_engram_api(report: Report) -> dict[str, Any]:
    print("\n=== 3. AI-Engram import/API path ===")
    out: dict[str, Any] = {
        "module": None,
        "get_engram": None,
        "apply_engram": None,
        "version_ok": False,
    }
    dist_ver = _pkg_version("ai-engram")
    engram, err = _try_import("engram")
    if engram is None:
        print(f"  import engram: FAILED ({err})")
        report.pending("engram.import", f"not available ({err})")
        report.pending("engram.version", "cannot verify 0.9.0 — package missing")
        report.pending("engram.get_engram", "cannot verify — package missing")
        report.pending("engram.apply_engram", "cannot verify — package missing")
        return out

    print("  import engram: OK")
    report.verified("engram.import")
    out["module"] = engram

    if dist_ver != EXPECTED_ENGRAM_VERSION:
        # Installed but wrong version is an incompatibility
        detail = f"expected ai-engram=={EXPECTED_ENGRAM_VERSION}, found {dist_ver!r}"
        print(f"  version: FAIL ({detail})")
        report.failed("engram.version", detail)
    else:
        print(f"  version: {dist_ver} (matches frozen)")
        report.verified("engram.version", dist_ver)
        out["version_ok"] = True

    get_fn = getattr(engram, "get_engram", None)
    apply_fn = getattr(engram, "apply_engram", None)
    if not callable(get_fn):
        report.failed("engram.get_engram", "attribute missing or not callable")
        print("  get_engram: FAIL")
    else:
        sig = str(inspect.signature(get_fn))
        params = list(inspect.signature(get_fn).parameters)
        # Frozen config uses forget= and total=
        ok_kwargs = "forget" in params and "total" in params
        if ok_kwargs:
            print(f"  get_engram: OK signature {sig}")
            report.verified("engram.get_engram", sig)
            out["get_engram"] = get_fn
        else:
            # Still may use *args/**kwargs — inspect carefully
            if any(
                p.kind
                in (
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.VAR_POSITIONAL,
                )
                for p in inspect.signature(get_fn).parameters.values()
            ):
                print(f"  get_engram: OK (variadic) signature {sig}")
                report.verified("engram.get_engram", sig)
                out["get_engram"] = get_fn
            else:
                detail = f"signature incompatible with forget=/total=: {sig}"
                print(f"  get_engram: FAIL ({detail})")
                report.failed("engram.get_engram", detail)

    if not callable(apply_fn):
        report.failed("engram.apply_engram", "attribute missing or not callable")
        print("  apply_engram: FAIL")
    else:
        sig = str(inspect.signature(apply_fn))
        params = list(inspect.signature(apply_fn).parameters)
        if "alpha" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(apply_fn).parameters.values()
        ):
            print(f"  apply_engram: OK signature {sig}")
            report.verified("engram.apply_engram", sig)
            out["apply_engram"] = apply_fn
        else:
            detail = f"signature incompatible with alpha=: {sig}"
            print(f"  apply_engram: FAIL ({detail})")
            report.failed("engram.apply_engram", detail)

    return out


def check_frozen_data_connection(research_root: Path, report: Report) -> dict[str, Any]:
    print("\n=== 5. Frozen data / config connection ===")
    frozen = research_root / "data" / "frozen"
    cfg_path = research_root / "configs" / "experiment.yaml"
    paths = {
        "config": cfg_path,
        "explicit": frozen / "extraction_sets_explicit.yaml",
        "lexical": frozen / "extraction_sets_lexical.yaml",
        "reference": frozen / "reference_corpus.yaml",
        "cloze": frozen / "cloze_probes.yaml",
    }
    for label, p in paths.items():
        if not p.is_file():
            report.failed(f"frozen.file.{label}", f"missing {p}")
            print(f"  missing: {p}")
            return {}

    try:
        cfg = load_yaml_via_frozen_validator(cfg_path)
        explicit = load_yaml_via_frozen_validator(paths["explicit"])
        lexical = load_yaml_via_frozen_validator(paths["lexical"])
        reference = load_yaml_via_frozen_validator(paths["reference"])
        cloze = load_yaml_via_frozen_validator(paths["cloze"])
    except Exception as exc:  # noqa: BLE001
        report.failed("frozen.load", str(exc))
        print(f"  YAML load FAIL: {exc}")
        return {}

    # Fact-ID resolution for extraction targets
    for kind, data in (("explicit", explicit), ("lexical", lexical)):
        sets = data.get("sets") or {}
        ok = all(
            fid in sets
            and isinstance(sets[fid], dict)
            and isinstance(sets[fid].get("items"), list)
            and len(sets[fid]["items"]) == 5
            for fid in ("F01", "F02", "F03", "F04")
        )
        if ok:
            report.verified(f"frozen.extract.{kind}", "F01–F04 × 5 sentences")
            print(f"  extraction {kind}: F01–F04 resolvable (5 each)")
        else:
            report.failed(f"frozen.extract.{kind}", "cannot resolve F01–F04 × 5")
            print(f"  extraction {kind}: FAIL")

    corpora = reference.get("corpora") or {}
    for variant in ("explicit", "lexical_control"):
        docs = (corpora.get(variant) or {}).get("documents")
        if isinstance(docs, list) and len(docs) == 20:
            report.verified(f"frozen.reference.{variant}", "20 documents")
            print(f"  reference {variant}: 20 docs resolvable")
        else:
            n = len(docs) if isinstance(docs, list) else None
            report.failed(f"frozen.reference.{variant}", f"expected 20 docs, got {n}")
            print(f"  reference {variant}: FAIL (n={n})")

    probes = cloze.get("probes") or {}
    cloze_ok = all(
        fid in probes
        and isinstance((probes[fid] or {}).get("items"), list)
        and len(probes[fid]["items"]) == 3
        for fid in ("F01", "F02", "F03", "F04")
    )
    if cloze_ok:
        report.verified("frozen.cloze", "F01–F04 × 3 probes")
        print("  cloze probes: F01–F04 resolvable (3 each)")
    else:
        report.failed("frozen.cloze", "cannot resolve F01–F04 × 3 probes")
        print("  cloze probes: FAIL")

    grid = (cfg.get("intervention") or {}).get("alpha_grid")
    grid_ok = (
        isinstance(grid, list)
        and len(grid) == len(EXPECTED_ALPHA_GRID)
        and all(abs(float(a) - b) < 1e-12 for a, b in zip(grid, EXPECTED_ALPHA_GRID))
    )
    if grid_ok:
        report.verified("frozen.alpha_grid", str(EXPECTED_ALPHA_GRID))
        print(f"  alpha_grid: {grid}")
    else:
        report.failed("frozen.alpha_grid", f"expected {EXPECTED_ALPHA_GRID}, got {grid!r}")
        print(f"  alpha_grid: FAIL ({grid!r})")

    model_id = (cfg.get("model") or {}).get("model_id")
    ae_ver = str((cfg.get("ai_engram") or {}).get("version"))
    modules = ((cfg.get("ai_engram") or {}).get("target_modules") or {}).get("modules")
    agree = (
        model_id == EXPECTED_MODEL_ID
        and ae_ver == EXPECTED_ENGRAM_VERSION
        and modules == EXPECTED_MODULES
    )
    if agree:
        report.verified(
            "frozen.config_agreement",
            f"model={model_id}, ai_engram={ae_ver}, modules=28×down_proj",
        )
        print(
            f"  config agreement: model={model_id}, ai-engram={ae_ver}, "
            f"target_modules={len(modules or [])}"
        )
    else:
        report.failed(
            "frozen.config_agreement",
            f"model={model_id!r}, ai_engram={ae_ver!r}, "
            f"n_modules={len(modules) if isinstance(modules, list) else None}",
        )
        print("  config agreement: FAIL")

    # Build forget/total for F01 explicit (for optional integration)
    f01_items = ((explicit.get("sets") or {}).get("F01") or {}).get("items") or []
    forget = [it["text"] for it in f01_items if isinstance(it, dict) and it.get("text")]
    total_docs = ((corpora.get("explicit") or {}).get("documents") or [])
    total = [d["text"] for d in total_docs if isinstance(d, dict) and d.get("text")]

    return {
        "cfg": cfg,
        "forget_f01_explicit": forget,
        "total_explicit": total,
    }


def resolve_module(root: Any, dotted: str) -> Any:
    """Resolve 'model.layers.0.mlp.down_proj' relative to a CausalLM module."""
    obj = root
    # HF CausalLM often nests the transformer as .model
    parts = dotted.split(".")
    # Try as-is from root; if first part is 'model' and root already is the inner model, skip
    for part in parts:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def check_model_path(
    report: Report,
    env: dict[str, Any],
) -> dict[str, Any]:
    print("\n=== 2. Model loading path ===")
    print(f"  Frozen model_id: {EXPECTED_MODEL_ID}")
    snap = local_model_snapshot(EXPECTED_MODEL_ID)
    report.info["local_snapshot"] = str(snap) if snap else None

    if snap is None:
        print(
            f"  Local weights: NOT FOUND under HF cache "
            f"(looked for {HF_HUB_DIRNAME}). No download performed."
        )
        report.pending(
            "model.local_weights",
            f"{EXPECTED_MODEL_ID} not present locally; download pending for GPU stage",
        )
        report.pending("model.load", "skipped — weights not local")
        report.pending("model.dtype", "skipped — weights not local")
        report.pending("model.modules", "skipped — weights not local")
        for i in range(28):
            report.pending(
                f"module.layers.{i}.mlp.down_proj",
                "skipped — model not loaded",
            )
        return {"model": None, "tokenizer": None, "snapshot": None}

    print(f"  Local snapshot: {snap}")
    report.verified("model.local_weights", str(snap))

    transformers, terr = _try_import("transformers")
    torch = env.get("torch")
    if transformers is None or torch is None:
        detail = f"transformers={terr}; torch missing={torch is None}"
        print(f"  Load skipped: dependencies unavailable ({detail})")
        report.pending("model.load", detail)
        report.pending("model.dtype", "skipped — deps missing")
        report.pending("model.modules", "skipped — deps missing")
        for i in range(28):
            report.pending(
                f"module.layers.{i}.mlp.down_proj",
                "skipped — deps missing",
            )
        return {"model": None, "tokenizer": None, "snapshot": snap}

    try:
        # local_files_only prevents any network fetch
        tok = transformers.AutoTokenizer.from_pretrained(
            str(snap), local_files_only=True, trust_remote_code=True
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            str(snap),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
    except Exception as exc:  # noqa: BLE001
        report.failed("model.load", str(exc))
        print(f"  model/tokenizer load FAIL: {exc}")
        report.pending("model.dtype", "skipped — load failed")
        report.pending("model.modules", "skipped — load failed")
        for i in range(28):
            report.pending(
                f"module.layers.{i}.mlp.down_proj",
                "skipped — load failed",
            )
        return {"model": None, "tokenizer": None, "snapshot": snap}

    print("  tokenizer/model construction: OK (local_files_only)")
    report.verified("model.load", "tokenizer + AutoModelForCausalLM local_files_only")

    # dtype check — parameters should be bf16 when requested
    try:
        dtypes = {p.dtype for p in model.parameters()}
        bf16 = getattr(torch, "bfloat16")
        if bf16 in dtypes:
            report.verified("model.dtype", f"parameter dtypes include bfloat16; set={dtypes}")
            print(f"  dtype: OK (includes bfloat16); seen={dtypes}")
        else:
            # Not necessarily fatal on CPU-only builds without bf16 support
            report.pending(
                "model.dtype",
                f"requested torch_dtype=bfloat16 but parameters are {dtypes}",
            )
            print(f"  dtype: PENDING — requested bf16, parameters={dtypes}")
    except Exception as exc:  # noqa: BLE001
        report.failed("model.dtype", str(exc))
        print(f"  dtype: FAIL ({exc})")

    print("\n=== 4. Model module path (28× mlp.down_proj) ===")
    # Prefer resolving via model.model (HF Qwen nesting)
    bases = []
    if hasattr(model, "model"):
        bases.append(("model.model", model.model))
    bases.append(("model", model))

    missing = []
    for dotted in EXPECTED_MODULES:
        found = False
        last_err = None
        for _label, base in bases:
            try:
                # If base is already inner transformer, strip leading 'model.'
                path = dotted
                if _label == "model.model" and path.startswith("model."):
                    path = path[len("model.") :]
                resolve_module(base, path)
                found = True
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        short = dotted  # model.layers.N.mlp.down_proj
        check_name = short.replace("model.", "module.", 1)
        if found:
            report.verified(check_name)
            print(f"  OK  {dotted}")
        else:
            missing.append(dotted)
            report.failed(check_name, str(last_err))
            print(f"  FAIL {dotted} ({last_err})")

    if not missing:
        report.verified("model.modules", "all 28 down_proj modules present")
    else:
        report.failed("model.modules", f"{len(missing)} missing")

    return {"model": model, "tokenizer": tok, "snapshot": snap}


def check_minimal_engram(
    report: Report,
    model_bundle: dict[str, Any],
    engram_bundle: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    print("\n=== 6. Minimal get_engram integration (optional) ===")
    model = model_bundle.get("model")
    tokenizer = model_bundle.get("tokenizer")
    get_fn: Optional[Callable[..., Any]] = engram_bundle.get("get_engram")
    forget = frozen.get("forget_f01_explicit") or []
    total = frozen.get("total_explicit") or []

    prereq_missing = []
    if model is None or tokenizer is None:
        prereq_missing.append("local model/tokenizer")
    if get_fn is None:
        prereq_missing.append("get_engram")
    if not engram_bundle.get("version_ok"):
        prereq_missing.append("ai-engram==0.9.0")
    if len(forget) != 5:
        prereq_missing.append("F01 forget set (5 sentences)")
    if len(total) != 20:
        prereq_missing.append("explicit reference total (20 sentences)")

    if prereq_missing:
        detail = "skipped — missing: " + ", ".join(prereq_missing)
        print(f"  {detail}")
        report.pending("engram.minimal_get_engram", detail)
        return

    # Construct only — do not apply_engram, calibrate, or matrix.
    try:
        print(
            f"  Calling get_engram once on F01 explicit "
            f"(forget={len(forget)}, total={len(total)}); no apply."
        )
        result = get_fn(model, tokenizer, forget=forget, total=total)
        n_layers = None
        if hasattr(result, "layers"):
            try:
                n_layers = len(result.layers)
            except Exception:  # noqa: BLE001
                n_layers = None
        detail = f"get_engram returned {type(result).__name__}"
        if n_layers is not None:
            detail += f" with {n_layers} layer entries"
        print(f"  OK — {detail}")
        report.verified("engram.minimal_get_engram", detail)
        # Explicitly do not call apply_engram
        report.verified(
            "engram.no_apply_in_this_check",
            "apply_engram intentionally not called",
        )
    except Exception as exc:  # noqa: BLE001
        report.failed("engram.minimal_get_engram", str(exc))
        print(f"  FAIL — get_engram raised: {exc}")


def summarize(report: Report) -> int:
    print("\n=== Summary ===")
    counts = {"verified": 0, "pending": 0, "failed": 0}
    for c in report.checks:
        counts[c.status] = counts.get(c.status, 0) + 1
    print(
        f"verified={counts['verified']}  pending={counts['pending']}  "
        f"failed={counts['failed']}"
    )
    print("\nBy status:")
    for status in ("failed", "pending", "verified"):
        items = [c for c in report.checks if c.status == status]
        if not items:
            continue
        print(f"  [{status}]")
        for c in items:
            extra = f" — {c.detail}" if c.detail else ""
            print(f"    - {c.name}{extra}")

    print("\nBlockers before paid GPU experiment:")
    # Always list prerequisite gaps even if not 'failed'
    blockers = list(report.blockers)
    for c in report.checks:
        if c.status == "pending" and c.name in (
            "dep.torch",
            "dep.transformers",
            "dep.peft",
            "dep.ai-engram",
            "engram.import",
            "engram.version",
            "model.local_weights",
            "model.load",
            "engram.minimal_get_engram",
        ):
            blockers.append(f"{c.name}: {c.detail}" if c.detail else c.name)
    # de-dupe
    seen = set()
    uniq = []
    for b in blockers:
        if b not in seen:
            seen.add(b)
            uniq.append(b)
    if not uniq and not report.has_failure:
        print("  (none recorded — stack verified for this host)")
    else:
        for b in uniq:
            print(f"  - {b}")

    print(
        "\nNote: this is engineering validation only; it is not a research result."
    )
    if report.has_failure:
        print("RESULT: FAIL (incompatibility detected)")
        return 1
    print("RESULT: OK (no incompatibilities; see pending items for unverified steps)")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-GPU runtime validation for TraceShift (no install/download)."
    )
    parser.add_argument(
        "--research-root",
        type=Path,
        default=None,
        help="Path to research/ (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    research_root = (
        args.research_root.resolve()
        if args.research_root
        else Path(__file__).resolve().parent.parent
    )
    if not (research_root / "configs" / "experiment.yaml").is_file():
        print(f"ERROR: experiment.yaml not found under {research_root}", file=sys.stderr)
        return 2

    report = Report()
    print(f"research_root: {research_root}")
    print("Policy: no package install, no weight download, no FT/alpha/matrix runs.\n")

    env = check_environment(report)
    engram_bundle = check_engram_api(report)
    frozen = check_frozen_data_connection(research_root, report)
    model_bundle = check_model_path(report, env)
    check_minimal_engram(report, model_bundle, engram_bundle, frozen)
    return summarize(report)


if __name__ == "__main__":
    sys.exit(main())
