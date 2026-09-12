# TraceShift balanced 6-fact experiment status summary

**Design:** balanced_6fact_3domain (tennis / swimming / acting; two facts each)  
**Run host:** `traceshift-l4` (remote GCP NVIDIA L4)  
**Execution:** persistent tmux session `traceshift-run-6fact`  
**Scientific clock start (UTC):** 2026-09-10T22:16:42Z  
**Approx total scientific wall time:** 77910 s  
**Overnight exit code:** 0  
**Status:** COMPLETE

This document records measured outcomes and engineering notes. It is **not** a polished MATS narrative.

---

## Fact universe (frozen)

| Domain | Facts |
|--------|-------|
| Tennis | F01 Roger Federer, F02 Rafael Nadal |
| Swimming | F03 Michael Phelps, F04 Katie Ledecky |
| Acting | F05 Amitabh Bachchan, F06 Shah Rukh Khan |

Within-domain pairs: F01↔F02, F03↔F04, F05↔F06.

---

## 1. Baseline recall

| Fact | Entity | Status | Wins |
|------|--------|--------|------|
| F01 | Roger Federer | PASS | 2/3 |
| F02 | Rafael Nadal | PASS | 2/3 |
| F03 | Michael Phelps | PASS | 3/3 |
| F04 | Katie Ledecky | PASS | 3/3 |
| F05 | Amitabh Bachchan | PASS | 3/3 |
| F06 | Shah Rukh Khan | PASS | 2/3 |

Artifact: `results/raw/baseline_recall/baseline_cloze.json`

---

## 2. Alpha calibration (BASE, explicit)

| Fact | Status | Alpha |
|------|--------|-------|
| F01 | selective | 0.4 |
| F02 | selective | 0.4 |
| F03 | selective | 0.8 |
| F04 | selective | 0.8 |
| F05 | selective | 0.4 |
| F06 | selective | 0.2 |

Artifacts: `results/raw/alpha_calibration/base_alphas.yaml`

---

## 3. Matrices

- M_base 6×6: present (`model_stage=M_base`)
- M_FT tennis 6×6: present (`model_state=primary_finetuned`)
- M_FT generic 6×6: present (`model_state=control_finetuned`)

---

## 4. Fine-tuning validity

- Tennis: {'kind': 'primary', 'ok': True, 'errors': [], 'adapter': '/home/siddharth/trace-shift/research/results/raw/finetuning/qwen3-1.7b-lora-tennis-primary/final_adapter', 'expected_optimizer_steps': 30, 'n_examples_expected': 40, 'epochs_expected': 3, 'global_step': 30, 'trainer_state_path': '/home/siddharth/trace-shift/research/results/raw/finetuning/qwen3-1.7b-lora-tennis-primary/checkpoint-30/trainer_state.json'}
- Generic control: {'kind': 'control', 'ok': True, 'errors': [], 'adapter': '/home/siddharth/trace-shift/research/results/raw/finetuning/qwen3-1.7b-lora-control/final_adapter', 'expected_optimizer_steps': 30, 'n_examples_expected': 40, 'epochs_expected': 3, 'global_step': 30, 'trainer_state_path': '/home/siddharth/trace-shift/research/results/raw/finetuning/qwen3-1.7b-lora-control/checkpoint-30/trainer_state.json', 'lora_norm_max': 2.363518238067627, 'lora_norm_max_provenance': {'source': 'final_adapter/adapter_model.safetensors', 'method': 'max_frobenius_norm_over_lora_named_tensors', 'n_lora_tensors': 224, 'n_nonzero_lora_tensors': 224, 'note': 'Backfilled because Phase-05 execution_validity omitted lora_norm_max after an in-memory nonzero-norm check.'}}

---

## 5. Primary Delta-M (tennis FT) — inferential

- mean tennis within-domain ΔM: `None`
- mean unrelated within-domain ΔM: `None`
- **T_obs (T_within):** `0.022448480129241943`
- Exact one-sided p (C(6,2)=15 cell permutation): `0.13333333333333333`
- Secondary descriptive T_cross: `None`
- Within-domain cells: `{"F01->F02": -0.0017691850662231445, "F02->F01": 0.020721375942230225, "F03->F04": 0.010760307312011719, "F04->F03": -0.0028675198554992676, "F05->F06": -0.016843140125274658, "F06->F05": -0.042939186096191406}`

Null distribution (15 contrasts):
```
[
  {
    "tennis_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ]
    ],
    "unrelated_cells": [
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ]
    ],
    "control_pairs": [
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.022448480129241943,
    "is_observed_assignment": true
  },
  {
    "tennis_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ]
    ],
    "unrelated_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ]
    ],
    "control_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.014977678656578064,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "unrelated_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "control_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.004756808280944824,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "unrelated_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "control_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": -0.005724906921386719,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "unrelated_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "in_domain_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "control_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "contrast": -0.02529694139957428,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.03184559941291809,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.02162472903728485,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.011143013834953308,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F02",
        "F01"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "in_domain_pairs": [
      [
        "F02",
        "F01"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "contrast": -0.008429020643234253,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.014153927564620972,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": 0.0036722123622894287,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F03",
        "F04"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "in_domain_pairs": [
      [
        "F03",
        "F04"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "contrast": -0.015899822115898132,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "in_domain_pairs": [
      [
        "F04",
        "F03"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "contrast": -0.006548658013343811,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "in_domain_pairs": [
      [
        "F04",
        "F03"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F05",
        "F06"
      ]
    ],
    "contrast": -0.026120692491531372,
    "is_observed_assignment": false
  },
  {
    "tennis_cells": [
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "unrelated_cells": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "in_domain_pairs": [
      [
        "F05",
        "F06"
      ],
      [
        "F06",
        "F05"
      ]
    ],
    "control_pairs": [
      [
        "F01",
        "F02"
      ],
      [
        "F02",
        "F01"
      ],
      [
        "F03",
        "F04"
      ],
      [
        "F04",
        "F03"
      ]
    ],
    "contrast": -0.036602407693862915,
    "is_observed_assignment": false
  }
]
```

Limitation: exact test over six directed within-domain cells; not population-level inference.

---

## 6. Generic FT control (descriptive)

- T_generic_within: `0.03052687644958496`
- T_tennis − T_generic: `-0.008078396320343018`
- No second p-value.

Artifacts: `results/raw/alternative_hypotheses/generic_ft_control_*.json`

---

## 7. Lexical-control (descriptive)

- T_lexical_within: `None`
- Artifact: `results/raw/alternative_hypotheses/lexical_control.json`

---

## 8. Phase timings

```
[
  {
    "phase": "01_baseline_recall",
    "result": "PASS",
    "elapsed_s": 33,
    "total_scientific_s": 40,
    "notes": "baseline cloze on M_base",
    "utc": "2026-09-10T22:17:22Z"
  },
  {
    "phase": "02_alpha_calibration",
    "result": "PASS",
    "elapsed_s": 1145,
    "total_scientific_s": 4317,
    "notes": "base alpha calibration",
    "utc": "2026-09-10T23:28:39Z"
  },
  {
    "phase": "03_base_matrix",
    "result": "PASS",
    "elapsed_s": 6626,
    "total_scientific_s": 32596,
    "notes": "M_base computed",
    "utc": "2026-09-11T07:19:58Z"
  },
  {
    "phase": "04_tennis_finetune",
    "result": "FAIL",
    "elapsed_s": 80,
    "total_scientific_s": 32683,
    "notes": "tennis FT validity gate failed",
    "utc": "2026-09-11T07:21:25Z"
  },
  {
    "phase": "04_tennis_finetune",
    "result": "PASS",
    "elapsed_s": 12,
    "total_scientific_s": 47382,
    "notes": "tennis LoRA FT revalidated (no retrain)",
    "utc": "2026-09-11T11:26:24Z"
  },
  {
    "phase": "05_control_finetune",
    "result": "PASS",
    "elapsed_s": 59,
    "total_scientific_s": 47446,
    "notes": "control LoRA FT + validity gate",
    "utc": "2026-09-11T11:27:28Z"
  },
  {
    "phase": "06_ft_matrix",
    "result": "PASS",
    "elapsed_s": 6704,
    "total_scientific_s": 54153,
    "notes": "M_FT_tennis 6x6 computed",
    "utc": "2026-09-11T13:19:15Z"
  },
  {
    "phase": "07_delta_analysis",
    "result": "PASS",
    "elapsed_s": 3,
    "total_scientific_s": 54162,
    "notes": "Delta-M tennis + C(6,2)=15 exact permutation",
    "utc": "2026-09-11T13:19:24Z"
  },
  {
    "phase": "08_generic_ft_control",
    "result": "FAIL",
    "elapsed_s": 9,
    "total_scientific_s": 54171,
    "notes": "generic FT control exit nonzero",
    "utc": "2026-09-11T13:19:33Z"
  },
  {
    "phase": "08_generic_ft_control",
    "result": "PASS",
    "elapsed_s": 6697,
    "total_scientific_s": 64519,
    "notes": "M_FT_generic 6x6 + descriptive T_generic_within",
    "utc": "2026-09-11T16:12:01Z"
  },
  {
    "phase": "09_lexical_control",
    "result": "PASS",
    "elapsed_s": 13373,
    "total_scientific_s": 77899,
    "notes": "lexical-control 6x6 matrices + T_lexical_within",
    "utc": "2026-09-11T19:55:01Z"
  },
  {
    "phase": "10_consolidation",
    "result": "PASS",
    "elapsed_s": 4,
    "total_scientific_s": 77910,
    "notes": "consolidation checks (6-fact)",
    "utc": "2026-09-11T19:55:12Z"
  }
]
```

---

## 9. Consolidation

```
{
  "design": "balanced_6fact_3domain",
  "frozen_input_sha256": {
    "facts": "499c032724474fe406cf529c96c27d0303d67a510cbf1542765f44a48f44254c",
    "explicit": "0a8b4a8e63093562242ca4e90dd8be04e552483f2463bcb2b23a44148966a8ee",
    "lexical": "ac99c23f7f51dbb64c57199b4de5edccbf49bf066ea672c43d0f4fc45d9086b4",
    "reference": "bf27d0cf29c326d54096bc8cdce29c7a5959c316f05150700d343e12b72032e2",
    "cloze": "3735f7b8ec1588837faab1df857cc42656b1b183a3933ae4829373d8a9321374",
    "experiment": "54861172d490b955f497414416520e276d30ad8f3361da8e6f21aab051319663"
  },
  "required_outputs_missing": [],
  "structural_issues": [],
  "hardware": {
    "gpu": "NVIDIA L4",
    "torch": "2.14.0+cu130",
    "torch_cuda": "13.0"
  },
  "software": {
    "python": "3.12.13",
    "transformers": "5.16.1",
    "peft": "0.20.0",
    "ai-engram": "0.9.0",
    "tqdm": "4.70.0",
    "PyYAML": "6.0.3"
  },
  "all_required_present": true
}
```

---

## Integrity

Frozen 6-fact inputs were not rewritten after measurements began for outcome shopping.
Primary inferential result is the tennis within-domain vs unrelated within-domain contrast with exact p over 15 cell assignments.
Generic FT and lexical results are descriptive only.
