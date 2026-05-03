from __future__ import annotations

import numpy as np


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def active_zscore_np(
    scores: np.ndarray,
    exclude_zero: bool = True,
    zero_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class-axis z-score used by sparse AVVP prediction.

    Exact-zero weights are rejected candidates under the v5.2 protocol. When
    exclude_zero=True, they are excluded from mean/std statistics and receive
    zero probability after scoring. Degenerate active rows are flagged so the
    caller can keep the legacy sparse behavior: active entries score 1.0.
    """
    scores = np.asarray(scores, dtype=np.float32)
    active_mask = np.abs(scores) > zero_eps
    active_count = active_mask.sum(axis=-1).astype(np.float32)

    if not exclude_zero:
        mu = scores.mean(axis=-1, keepdims=True)
        sd = scores.std(axis=-1, keepdims=True)
        valid = sd[..., 0] > zero_eps
        z = (scores - mu) / np.maximum(sd, zero_eps)
        z = np.where(valid[..., None], z, 0.0)
        return np.clip(z, -50.0, 50.0).astype(np.float32), active_mask, active_count, valid

    count = active_count[..., None]
    denom = np.maximum(count, 1.0)
    mu = (scores * active_mask).sum(axis=-1, keepdims=True) / denom
    var = (((scores - mu) ** 2) * active_mask).sum(axis=-1, keepdims=True) / denom
    sd = np.sqrt(var)
    valid = (count[..., 0] >= 2.0) & (sd[..., 0] > zero_eps)
    z = np.zeros_like(scores, dtype=np.float32)
    if np.any(valid):
        z_all = (scores - mu) / np.maximum(sd, zero_eps)
        z = np.where(valid[..., None] & active_mask, z_all, z)
    return np.clip(z, -50.0, 50.0).astype(np.float32), active_mask, active_count, valid


def sparse_weight_scores(
    W: np.ndarray,
    temperature: float | np.ndarray = 1.0,
    exclude_zero: bool = True,
    zero_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probabilities, active count K, and per-row temperature."""
    z, active_mask, active_count, valid = active_zscore_np(
        W, exclude_zero=exclude_zero, zero_eps=zero_eps
    )
    temperature_arr = np.asarray(temperature, dtype=np.float32)
    if temperature_arr.ndim == 0:
        temperature_arr = np.full(active_count.shape, float(temperature_arr), dtype=np.float32)
    probs = _sigmoid_np(z / np.maximum(temperature_arr[..., None], zero_eps))
    if exclude_zero:
        probs = np.where(active_mask, probs, 0.0).astype(np.float32)
    degenerate = (active_count > 0) & (~valid)
    if exclude_zero and np.any(degenerate):
        probs = np.where(degenerate[..., None] & active_mask, 1.0, probs).astype(np.float32)
    return probs.astype(np.float32), active_count.astype(np.float32), temperature_arr.astype(np.float32)


def score_sparse_weights(
    W: np.ndarray,
    tau: float = 0.75,
    k0: float = 16.0,
    t_min: float = 0.25,
    t_max: float = 1.25,
    exclude_zero: bool = True,
    zero_eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score sparse weights with the v5.2 active-count protocol.

    W can be shaped (N, T, C) or (num_segments, C). The returned tuple is
    (probabilities, binary predictions, active count K, temperature T).

    Protocol:
      1. zero weights are rejected candidates and excluded from z-score stats;
      2. K is the nonzero class count per segment;
      3. T = clip(K / K0, Tmin, Tmax);
      4. p = sigmoid(z / T), y = 1[p > tau].
    """
    _, active_mask, active_count, _ = active_zscore_np(
        W, exclude_zero=exclude_zero, zero_eps=zero_eps
    )
    del active_mask
    temperature = np.clip(
        active_count / (float(k0) + zero_eps),
        float(t_min),
        float(t_max),
    ).astype(np.float32)
    probs, active_count, temperature = sparse_weight_scores(
        W,
        temperature=temperature,
        exclude_zero=exclude_zero,
        zero_eps=zero_eps,
    )
    pred = (probs > float(tau)).astype(np.uint8)
    return probs, pred, active_count, temperature


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
        z, _, _, _ = active_zscore_np(scores, exclude_zero=False, zero_eps=zero_eps)
        return _sigmoid_np(z)

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
            normalized = _sigmoid_np(z)
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
