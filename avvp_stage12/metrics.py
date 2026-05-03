from __future__ import annotations

import numpy as np


def norm_similarities_np(scores: np.ndarray) -> np.ndarray:
    mu = scores.mean(axis=-1, keepdims=True)
    sd = scores.std(axis=-1, keepdims=True) + 1e-8
    z = (scores - mu) / sd
    return 1.0 / (1.0 + np.exp(-z))


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
