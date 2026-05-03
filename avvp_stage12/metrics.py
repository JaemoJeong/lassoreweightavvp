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


def apply_min_duration_filter(pred: np.ndarray, min_duration: int = 1) -> np.ndarray:
    """Remove per-video class predictions active for fewer than min_duration segments."""
    pred = np.asarray(pred, dtype=np.uint8)
    if min_duration <= 1:
        return pred.astype(np.uint8)
    if pred.ndim != 3:
        raise ValueError(f"min-duration filter expects (N,T,C) predictions, got {pred.shape}")
    keep = pred.sum(axis=1, keepdims=True) >= int(min_duration)
    return (pred & keep).astype(np.uint8)


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


def _binary_f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> float:
    values = []
    for tp_i, fp_i, fn_i in zip(tp, fp, fn):
        denom = 2 * tp_i + fp_i + fn_i
        if denom > 0:
            values.append(2 * tp_i / denom)
    return float(np.mean(values)) if values else 1.0


def _to_event_vec(start: int, end: int, length: int) -> np.ndarray:
    x = np.zeros(length, dtype=np.uint8)
    x[start:end] = 1
    return x


def _extract_events(seq: np.ndarray) -> list[np.ndarray] | None:
    seq = np.asarray(seq).astype(np.uint8)
    length = int(seq.shape[0])
    events: list[np.ndarray] = []
    idx = 0
    while idx < length:
        if seq[idx] != 1:
            idx += 1
            continue
        start = idx
        while idx < length and seq[idx] == 1:
            idx += 1
        events.append(_to_event_vec(start, idx, length))
    return events if events else None


def _event_wise_counts(event_p: list[np.ndarray] | None, event_gt: list[np.ndarray] | None) -> tuple[int, int, int]:
    tp = 0
    fp = 0
    fn = 0
    if event_p is not None:
        for pred_event in event_p:
            matched = False
            if event_gt is not None:
                for gt_event in event_gt:
                    inter = np.sum(pred_event * gt_event)
                    union = np.sum(pred_event + gt_event - pred_event * gt_event)
                    if inter >= 0.5 * union:
                        tp += 1
                        matched = True
                        break
            if not matched:
                fp += 1
    if event_gt is not None:
        for gt_event in event_gt:
            matched = False
            if event_p is not None:
                for pred_event in event_p:
                    inter = np.sum(gt_event * pred_event)
                    union = np.sum(gt_event + pred_event - gt_event * pred_event)
                    if inter >= 0.5 * union:
                        matched = True
                        break
            if not matched:
                fn += 1
    return tp, fp, fn


def avvp_official_segment_level(
    so_a: np.ndarray,
    so_v: np.ndarray,
    so_av: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    gt_av: np.ndarray,
) -> tuple[float, float, float, float]:
    """AV2A/AVVP segment-level metrics for one video.

    Inputs are shaped (C, T), matching AV2A_pristine/eval_metrics.py.
    Returns (F_a, F_v, F_event, F_av), where F_event is the combined
    audio+visual event metric used as Event@AV segment in the paper table.
    """
    so_a = so_a.astype(np.int32)
    so_v = so_v.astype(np.int32)
    so_av = so_av.astype(np.int32)
    gt_a = gt_a.astype(np.int32)
    gt_v = gt_v.astype(np.int32)
    gt_av = gt_av.astype(np.int32)

    tp_a = np.sum(so_a * gt_a, axis=1)
    fn_a = np.sum((1 - so_a) * gt_a, axis=1)
    fp_a = np.sum(so_a * (1 - gt_a), axis=1)

    tp_v = np.sum(so_v * gt_v, axis=1)
    fn_v = np.sum((1 - so_v) * gt_v, axis=1)
    fp_v = np.sum(so_v * (1 - gt_v), axis=1)

    tp_event = tp_a + tp_v
    fn_event = fn_a + fn_v
    fp_event = fp_a + fp_v

    tp_av = np.sum(so_av * gt_av, axis=1)
    fn_av = np.sum((1 - so_av) * gt_av, axis=1)
    fp_av = np.sum(so_av * (1 - gt_av), axis=1)

    return (
        _binary_f1_from_counts(tp_a, fp_a, fn_a),
        _binary_f1_from_counts(tp_v, fp_v, fn_v),
        _binary_f1_from_counts(tp_event, fp_event, fn_event),
        _binary_f1_from_counts(tp_av, fp_av, fn_av),
    )


def avvp_official_event_level(
    so_a: np.ndarray,
    so_v: np.ndarray,
    so_av: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    gt_av: np.ndarray,
) -> tuple[float, float, float, float]:
    """AV2A/AVVP event-level metrics for one video.

    Inputs are shaped (C, T). Contiguous positive intervals are events, and a
    predicted event is correct when temporal IoU with a GT event is at least 0.5.
    """
    num_classes = int(so_a.shape[0])
    tp_a = np.zeros(num_classes)
    fp_a = np.zeros(num_classes)
    fn_a = np.zeros(num_classes)
    tp_v = np.zeros(num_classes)
    fp_v = np.zeros(num_classes)
    fn_v = np.zeros(num_classes)
    tp_av = np.zeros(num_classes)
    fp_av = np.zeros(num_classes)
    fn_av = np.zeros(num_classes)

    for class_idx in range(num_classes):
        counts = _event_wise_counts(
            _extract_events(so_a[class_idx]), _extract_events(gt_a[class_idx])
        )
        tp_a[class_idx], fp_a[class_idx], fn_a[class_idx] = counts

        counts = _event_wise_counts(
            _extract_events(so_v[class_idx]), _extract_events(gt_v[class_idx])
        )
        tp_v[class_idx], fp_v[class_idx], fn_v[class_idx] = counts

        counts = _event_wise_counts(
            _extract_events(so_av[class_idx]), _extract_events(gt_av[class_idx])
        )
        tp_av[class_idx], fp_av[class_idx], fn_av[class_idx] = counts

    tp_event = tp_a + tp_v
    fp_event = fp_a + fp_v
    fn_event = fn_a + fn_v

    return (
        _binary_f1_from_counts(tp_a, fp_a, fn_a),
        _binary_f1_from_counts(tp_v, fp_v, fn_v),
        _binary_f1_from_counts(tp_event, fp_event, fn_event),
        _binary_f1_from_counts(tp_av, fp_av, fn_av),
    )


def avvp_official_metrics(
    pred_a: np.ndarray,
    pred_v: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    pred_av: np.ndarray | None = None,
    gt_av: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the AVVP table metrics used by AV2A.

    Inputs are shaped (N, T, C). Returned values are fractions in [0, 1], not
    percentages. The AV2A names are kept for traceability:
      - F_seg_a / F_event_a: Audio segment/event
      - F_seg_v / F_event_v: Visual segment/event
      - F_seg_av / F_event_av: Audio Visual segment/event
      - avg_type / avg_type_event: Type@AV segment/event
      - avg_event / avg_event_level: Event@AV segment/event
    """
    pred_a = pred_a.astype(np.uint8)
    pred_v = pred_v.astype(np.uint8)
    gt_a = gt_a.astype(np.uint8)
    gt_v = gt_v.astype(np.uint8)
    pred_av = (pred_a & pred_v).astype(np.uint8) if pred_av is None else pred_av.astype(np.uint8)
    gt_av = (gt_a & gt_v).astype(np.uint8) if gt_av is None else gt_av.astype(np.uint8)

    seg_a = []
    seg_v = []
    seg_event = []
    seg_av = []
    event_a = []
    event_v = []
    event_event = []
    event_av = []

    for idx in range(pred_a.shape[0]):
        so_a = pred_a[idx].T
        so_v = pred_v[idx].T
        so_av = pred_av[idx].T
        ga = gt_a[idx].T
        gv = gt_v[idx].T
        gav = gt_av[idx].T

        f_a, f_v, f_event, f_av = avvp_official_segment_level(so_a, so_v, so_av, ga, gv, gav)
        seg_a.append(f_a)
        seg_v.append(f_v)
        seg_event.append(f_event)
        seg_av.append(f_av)

        f_a, f_v, f_event, f_av = avvp_official_event_level(so_a, so_v, so_av, ga, gv, gav)
        event_a.append(f_a)
        event_v.append(f_v)
        event_event.append(f_event)
        event_av.append(f_av)

    F_seg_a = float(np.mean(seg_a))
    F_seg_v = float(np.mean(seg_v))
    F_seg = float(np.mean(seg_event))
    F_seg_av = float(np.mean(seg_av))
    F_event_a = float(np.mean(event_a))
    F_event_v = float(np.mean(event_v))
    F_event = float(np.mean(event_event))
    F_event_av = float(np.mean(event_av))
    avg_type = (F_seg_a + F_seg_v + F_seg_av) / 3.0
    avg_type_event = (F_event_a + F_event_v + F_event_av) / 3.0
    return {
        "F_seg_a": F_seg_a,
        "F_seg_v": F_seg_v,
        "F_seg": F_seg,
        "F_seg_av": F_seg_av,
        "avg_type": avg_type,
        "avg_event": F_seg,
        "F_event_a": F_event_a,
        "F_event_v": F_event_v,
        "F_event": F_event,
        "F_event_av": F_event_av,
        "avg_type_event": avg_type_event,
        "avg_event_level": F_event,
        "audio_segment_f1": F_seg_a,
        "audio_event_f1": F_event_a,
        "visual_segment_f1": F_seg_v,
        "visual_event_f1": F_event_v,
        "audio_visual_segment_f1": F_seg_av,
        "audio_visual_event_f1": F_event_av,
        "type_av_segment_f1": avg_type,
        "type_av_event_f1": avg_type_event,
        "event_av_segment_f1": F_seg,
        "event_av_event_f1": F_event,
    }


def recon_aware_thresholds(
    recon: np.ndarray,
    base_thr: float,
    delta: float,
    mean_recon: float,
) -> np.ndarray:
    return base_thr + delta * np.maximum(0.0, mean_recon - recon)


def apply_segment_thresholds(weights: np.ndarray, seg_thresholds: np.ndarray) -> np.ndarray:
    return weights > seg_thresholds[:, :, None]
