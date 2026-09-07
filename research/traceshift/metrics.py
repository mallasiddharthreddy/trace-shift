"""Shared trace comparison metrics matching prior TraceShift / AI-Engram matrix work.

Prior experiment code (``compare_engrams``) computed global cosine *similarity*
on concatenated float32 flattened layer projections. For TraceShift response
matrices the primary cell statistic is the corresponding cosine *distance*
(change):

    M[i, j] = 1 - cos(baseline_trace_j, post_intervention_trace_j)

where ``cos`` is the same global formula used previously.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .engram import ExtractedEngram, TraceVector, engram_to_trace


def cosine_similarity_vectors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity of two 1-D float tensors (CPU float32)."""
    aa = a.detach().to("cpu", dtype=torch.float32).reshape(-1)
    bb = b.detach().to("cpu", dtype=torch.float32).reshape(-1)
    if aa.numel() != bb.numel():
        raise ValueError(
            f"Trace length mismatch: {aa.numel()} vs {bb.numel()}"
        )
    if aa.numel() == 0:
        raise ValueError("Empty traces")
    sim = F.cosine_similarity(aa.unsqueeze(0), bb.unsqueeze(0)).item()
    return float(sim)


def cosine_distance_vectors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Primary response statistic: 1 - cosine_similarity."""
    return float(1.0 - cosine_similarity_vectors(a, b))


def cosine_similarity_engram_results(result_a: Any, result_b: Any) -> tuple[dict[str, float], float]:
    """Layer-wise + global cosine similarity (prior ``compare_engrams`` convention).

    Global similarity accumulates dots / norms across shared layer names, matching
    the previous experiment implementation.
    """
    per_layer: dict[str, float] = {}
    dot_sum = 0.0
    norm_a_sum = 0.0
    norm_b_sum = 0.0
    names = sorted(set(result_a.layers.keys()) & set(result_b.layers.keys()))
    if not names:
        raise ValueError("No shared EngramResult.layers keys to compare")
    for name in names:
        pa = result_a.layers[name].projection.detach().to("cpu", torch.float32).flatten()
        pb = result_b.layers[name].projection.detach().to("cpu", torch.float32).flatten()
        sim = F.cosine_similarity(pa.unsqueeze(0), pb.unsqueeze(0)).item()
        per_layer[name] = float(sim)
        dot_sum += torch.dot(pa, pb).item()
        norm_a_sum += pa.pow(2).sum().item()
        norm_b_sum += pb.pow(2).sum().item()
    global_sim = dot_sum / ((norm_a_sum**0.5) * (norm_b_sum**0.5) + 1e-12)
    return per_layer, float(global_sim)


def primary_response(
    baseline_trace: TraceVector,
    post_trace: TraceVector,
) -> dict[str, float]:
    """Primary matrix cell from TraceShift concatenated traces."""
    sim = cosine_similarity_vectors(baseline_trace.vector, post_trace.vector)
    return {
        "cosine_similarity": sim,
        "cosine_distance": float(1.0 - sim),
    }


def extracted_to_trace(extracted: ExtractedEngram) -> TraceVector:
    return engram_to_trace(extracted)
