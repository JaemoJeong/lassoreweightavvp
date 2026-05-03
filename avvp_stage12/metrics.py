from __future__ import annotations

import numpy as np


def norm_similarities_np(
    scores: np.ndarray,
    exclude_zero: bool = True,
    zero_eps: float = 1e-8,
) -> np.ndarray:
    """Class-axis z-score + sigmoid score normalization.

    By default, exact-zero Lasso weights are excluded from the row mean/std so
    sparse inactive classes do not dominate the normalization statistics.
    Degenerate sparse rows are handled explicitly:
      - no nonzero weights: all scores are 0
      - one/equal nonzero weights: nonzero scores are 1, zero scores are 0
    """
    scores = np.asarray(scores, dtype=np.float32)
    if not exclude_zero:
        mu = scores.mean(axis=-1, keepdims=True)
        sd = scores.std(axis=-1, keepdims=True) + zero_eps
        z = (scores - mu) / sd
        z = np.clip(z, -50.0, 50.0)
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    mask = np.abs(scores) > zero_eps
    count = mask.sum(axis=-1, keepdims=True)
    out = np.zeros_like(scores, dtype=np.float32)
    denom = np.maximum(count, 1).astype(np.float32)
    mu = (scores * mask).sum(axis=-1, keepdims=True) / denom
    var = (((scores - mu) ** 2) * mask).sum(axis=-1, keepdims=True) / denom
    sd = np.sqrt(var)

    enough = count >= 2
    if np.any(enough):
        valid = enough & (sd > zero_eps)
        if np.any(valid):
            z = (scores - mu) / np.maximum(sd, zero_eps)
            z = np.clip(z, -50.0, 50.0)
            normalized = 1.0 / (1.0 + np.exp(-z))
            out = np.where(valid, normalized, out).astype(np.float32)

    degenerate = (count > 0) & ((count < 2) | (sd <= zero_eps))
    if np.any(degenerate):
        out = np.where(degenerate & mask, 1.0, out).astype(np.float32)
    return out


def avvp_segment_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(np.int32)
    gt = gt.astype(np.int32)
    tp = (pred * gt).sum(axis=1)
    fn = ((1 - pred) * gt).sum(axis=1)
    fp = (pred * (1 - gt)).sum(axis=1)
    values = []
    for tp_i, fp_i, fn_i in zip(tp, fp, fn):
        denom = 2 * tp_i + fp_i + fn_i
        if denom > 0:
            values.append(2 * tp_i / denom)
    return float(np.mean(values)) if values else 1.0


def recon_aware_thresholds(
    recon: np.ndarray,
    base_thr: float,
    delta: float,
    mean_recon: float,
) -> np.ndarray:
    return base_thr + delta * np.maximum(0.0, mean_recon - recon)


def apply_segment_thresholds(weights: np.ndarray, seg_thresholds: np.ndarray) -> np.ndarray:
    return weights > seg_thresholds[:, :, None]
