from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import DEFAULT_BACKBONE, DEFAULT_VOCAB
from .data import build_dense_gt, load_llp_cached_bundle, load_prompt_vocab
from .metrics import avvp_official_metrics, avvp_segment_f1, norm_similarities_np


DEFAULT_AV2A_METRICS_PATH = Path(
    "/home/jaemo/AV2A_pristine/runs/llp_clipclap_20260420_l2norm_full/per_class_metrics.json"
)


def _as_fraction(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def load_av2a_baseline(metrics_path: str | Path | None = DEFAULT_AV2A_METRICS_PATH) -> dict[str, object] | None:
    if not metrics_path:
        return None
    path = Path(metrics_path)
    if not path.exists():
        print(f"[warn] AV2A baseline metrics not found: {path}")
        return None
    data = json.loads(path.read_text())
    overall = data.get("overall", data)
    baseline = {
        "path": str(path),
        "audio": _as_fraction(overall.get("F_seg_a") or overall.get("audio_segment_f1")),
        "visual": _as_fraction(overall.get("F_seg_v") or overall.get("visual_segment_f1")),
        "av": _as_fraction(
            overall.get("F_seg_av")
            or overall.get("av_segment_f1")
            or overall.get("av_segment_f1_and")
        ),
    }
    for key in [
        "F_seg_a",
        "F_seg_v",
        "F_seg",
        "F_seg_av",
        "avg_type",
        "avg_event",
        "F_event_a",
        "F_event_v",
        "F_event",
        "F_event_av",
        "avg_type_event",
        "avg_event_level",
    ]:
        if key in overall:
            baseline[key] = _as_fraction(overall[key])
    if "F_seg_a" in baseline:
        baseline.update({
            "audio_segment_f1": baseline.get("F_seg_a"),
            "audio_event_f1": baseline.get("F_event_a"),
            "visual_segment_f1": baseline.get("F_seg_v"),
            "visual_event_f1": baseline.get("F_event_v"),
            "audio_visual_segment_f1": baseline.get("F_seg_av"),
            "audio_visual_event_f1": baseline.get("F_event_av"),
            "type_av_segment_f1": baseline.get("avg_type"),
            "type_av_event_f1": baseline.get("avg_type_event"),
            "event_av_segment_f1": baseline.get("avg_event"),
            "event_av_event_f1": baseline.get("avg_event_level"),
        })
    return baseline


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.clip(norm, eps, None)).astype(np.float32)


def _select_filenames(bundle: dict[str, object], filenames: list[str] | None) -> tuple[list[str], np.ndarray]:
    bundle_filenames = list(bundle["filenames"])
    if filenames is None:
        return bundle_filenames, np.arange(len(bundle_filenames), dtype=np.int64)
    index = {fn: idx for idx, fn in enumerate(bundle_filenames)}
    missing = [fn for fn in filenames if fn not in index]
    if missing:
        raise ValueError(f"{len(missing)} filenames are not present in cached bundle; first missing={missing[0]}")
    return list(filenames), np.array([index[fn] for fn in filenames], dtype=np.int64)


def _predict_from_cosine_scores(scores: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    # Raw cosine scores are dense, so use the original AV2A row-wise z-score +
    # sigmoid recipe rather than the sparse-Lasso zero-exclusion variant.
    normalized = norm_similarities_np(scores, exclude_zero=False)
    pred = (normalized > threshold).astype(np.uint8)
    return normalized.astype(np.float32), pred


def compute_zero_shot_baseline(
    filenames: list[str] | None = None,
    backbone: str = DEFAULT_BACKBONE,
    vocab: str = DEFAULT_VOCAB,
    threshold: float = 0.75,
) -> dict[str, object]:
    """Compute dense encoder zero-shot AVVP segment F1.

    ZS-CLAP uses cached audio segment embeddings against CLAP text prototypes.
    ZS-CLIP uses cached visual segment embeddings against CLIP text prototypes.
    Both use raw cosine scores followed by AV2A's per-segment class-axis
    z-score + sigmoid + fixed threshold.
    """
    bundle = load_llp_cached_bundle(backbone=backbone)
    kept_filenames, idx = _select_filenames(bundle, filenames)
    vocab_bundle = load_prompt_vocab(vocab)

    audio_segments = _l2_normalize(np.asarray(bundle["audio_segments"])[idx], axis=-1)
    visual_segments = _l2_normalize(np.asarray(bundle["visual_segments"])[idx], axis=-1)
    audio_proto = _l2_normalize(vocab_bundle["audio_rows"], axis=-1)
    visual_proto = _l2_normalize(vocab_bundle["visual_rows"], axis=-1)

    scores_a = audio_segments @ audio_proto.T
    scores_v = visual_segments @ visual_proto.T
    _, pred_a = _predict_from_cosine_scores(scores_a, threshold)
    _, pred_v = _predict_from_cosine_scores(scores_v, threshold)
    pred_av = (pred_a & pred_v).astype(np.uint8)

    gt_a = build_dense_gt(kept_filenames, "audio")
    gt_v = build_dense_gt(kept_filenames, "visual")
    gt_av = (gt_a & gt_v).astype(np.uint8)
    official = avvp_official_metrics(pred_a, pred_v, gt_a, gt_v, pred_av=pred_av, gt_av=gt_av)

    return {
        "backbone": backbone,
        "vocab": vocab,
        "threshold": float(threshold),
        "num_videos": len(kept_filenames),
        "num_segments": int(audio_segments.shape[1]),
        "score_normalization": "raw cosine -> per segment class-axis z-score -> sigmoid -> fixed threshold",
        "zs_clap_audio": avvp_segment_f1(pred_a.reshape(-1, pred_a.shape[-1]), gt_a.reshape(-1, gt_a.shape[-1])),
        "zs_clip_visual": avvp_segment_f1(pred_v.reshape(-1, pred_v.shape[-1]), gt_v.reshape(-1, gt_v.shape[-1])),
        "zs_av_and": avvp_segment_f1(pred_av.reshape(-1, pred_av.shape[-1]), gt_av.reshape(-1, gt_av.shape[-1])),
        "official_avvp": official,
        "F_seg_a": official["F_seg_a"],
        "F_seg_v": official["F_seg_v"],
        "F_seg": official["F_seg"],
        "F_seg_av": official["F_seg_av"],
        "avg_type": official["avg_type"],
        "avg_event": official["avg_event"],
        "F_event_a": official["F_event_a"],
        "F_event_v": official["F_event_v"],
        "F_event": official["F_event"],
        "F_event_av": official["F_event_av"],
        "avg_type_event": official["avg_type_event"],
        "avg_event_level": official["avg_event_level"],
        "audio_segment_f1": official["audio_segment_f1"],
        "audio_event_f1": official["audio_event_f1"],
        "visual_segment_f1": official["visual_segment_f1"],
        "visual_event_f1": official["visual_event_f1"],
        "audio_visual_segment_f1": official["audio_visual_segment_f1"],
        "audio_visual_event_f1": official["audio_visual_event_f1"],
        "type_av_segment_f1": official["type_av_segment_f1"],
        "type_av_event_f1": official["type_av_event_f1"],
        "event_av_segment_f1": official["event_av_segment_f1"],
        "event_av_event_f1": official["event_av_event_f1"],
        "audio_pred_active_mean": float(pred_a.sum(axis=-1).mean()),
        "visual_pred_active_mean": float(pred_v.sum(axis=-1).mean()),
        "av_pred_active_mean": float(pred_av.sum(axis=-1).mean()),
    }
