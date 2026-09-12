# Archive (historical / non-canonical)

Material preserved for provenance, not for primary scientific citation.

| Path | What it is |
|------|------------|
| `reports/` | Pre-GPU audit, repo preflight, live runtime verification, and the long run-status dump from the overnight GPU session |
| `scripts/` | Detached-tmux / overnight GPU orchestrators used once to produce the canonical 6-fact run (`run_full_experiment.sh`, resume/finalize wrappers). Hardcoded absolute paths from the original VM are intentional historical artifacts. |
| `runtime_verification/` | GPU smoke / host / `nvidia-smi` checks from the L4 verification session |
| `pre_gpu/` | Pre-GPU readiness STATUS before the scientific clock started |

**4-fact design outputs** were superseded by the balanced 6-fact / 3-domain design. They are **not** present as files in this tree; they exist only in Git history (see commits before the 6-fact results commit). Do not cite 4-fact matrices or the old `C(4,2)` permutation as the final experiment.

Canonical science lives under:

- `research/data/frozen/`
- `research/results/raw/` and `research/results/tables/`
- `research/reports/` (non-archive)
- `research/results/figures/final/`
