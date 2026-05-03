from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import LLP_CATS
from .metrics import avvp_segment_f1, norm_similarities_np


def _binary_f1_per_class(pred: np.ndarray, gt: np.ndarray) -> list[dict[str, float | int | str]]:
    pred_2d = pred.reshape(-1, pred.shape[-1]).astype(np.int32)
    gt_2d = gt.reshape(-1, gt.shape[-1]).astype(np.int32)
    out: list[dict[str, float | int | str]] = []
    for idx, label in enumerate(LLP_CATS):
        p = pred_2d[:, idx]
        g = gt_2d[:, idx]
        tp = int(((p == 1) & (g == 1)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        support = int(g.sum())
        pred_pos = int(p.sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out.append({
            "class_index": idx,
            "class": label,
            "support": support,
            "pred_pos": pred_pos,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
    return out


def _top_labels(scores: np.ndarray, top_k: int) -> str:
    if top_k <= 0:
        return ""
    idx = np.argsort(scores)[::-1][:top_k]
    return ", ".join(f"{LLP_CATS[i]}:{scores[i]:.3f}" for i in idx)


def compute_stage_predictions(
    W_a: np.ndarray,
    W_v: np.ndarray,
    threshold: float,
    score_exclude_zero: bool = True,
) -> dict[str, np.ndarray]:
    scores_a = norm_similarities_np(W_a, exclude_zero=score_exclude_zero)
    scores_v = norm_similarities_np(W_v, exclude_zero=score_exclude_zero)
    pred_a = (scores_a > threshold).astype(np.uint8)
    pred_v = (scores_v > threshold).astype(np.uint8)
    pred_av = (pred_a & pred_v).astype(np.uint8)
    return {
        "scores_a": scores_a.astype(np.float32),
        "scores_v": scores_v.astype(np.float32),
        "pred_a": pred_a,
        "pred_v": pred_v,
        "pred_av": pred_av,
    }


def compute_stage_metrics(
    stage_name: str,
    pred: dict[str, np.ndarray],
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    threshold: float,
    score_exclude_zero: bool,
) -> dict[str, object]:
    gt_av = (gt_a & gt_v).astype(np.uint8)
    pred_a = pred["pred_a"]
    pred_v = pred["pred_v"]
    pred_av = pred["pred_av"]
    metrics = {
        "stage": stage_name,
        "threshold": float(threshold),
        "score_normalization": "per segment, over class axis: sigmoid((W - mean(W_class)) / std(W_class))",
        "score_exclude_zero": bool(score_exclude_zero),
        "audio_segment_f1": avvp_segment_f1(pred_a.reshape(-1, pred_a.shape[-1]), gt_a.reshape(-1, gt_a.shape[-1])),
        "visual_segment_f1": avvp_segment_f1(pred_v.reshape(-1, pred_v.shape[-1]), gt_v.reshape(-1, gt_v.shape[-1])),
        "av_segment_f1_and": avvp_segment_f1(pred_av.reshape(-1, pred_av.shape[-1]), gt_av.reshape(-1, gt_av.shape[-1])),
        "audio_pred_active_mean": float(pred_a.sum(axis=-1).mean()),
        "visual_pred_active_mean": float(pred_v.sum(axis=-1).mean()),
        "av_pred_active_mean": float(pred_av.sum(axis=-1).mean()),
        "audio_gt_active_mean": float(gt_a.sum(axis=-1).mean()),
        "visual_gt_active_mean": float(gt_v.sum(axis=-1).mean()),
        "av_gt_active_mean": float(gt_av.sum(axis=-1).mean()),
        "per_class_audio": _binary_f1_per_class(pred_a, gt_a),
        "per_class_visual": _binary_f1_per_class(pred_v, gt_v),
        "per_class_av_and": _binary_f1_per_class(pred_av, gt_av),
    }
    return metrics


def write_segment_details(
    out_path: Path,
    stage_name: str,
    filenames: list[str],
    video_ids: list[str],
    W_a: np.ndarray,
    W_v: np.ndarray,
    pred: dict[str, np.ndarray],
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    threshold: float,
    score_exclude_zero: bool,
    max_videos: int,
    top_k: int,
    all_classes: bool,
    extra_columns: list[tuple[str, np.ndarray]] | None = None,
) -> None:
    scores_a = pred["scores_a"]
    scores_v = pred["scores_v"]
    pred_a = pred["pred_a"]
    pred_v = pred["pred_v"]
    pred_av = pred["pred_av"]
    gt_av = (gt_a & gt_v).astype(np.uint8)
    num_videos = len(filenames) if max_videos <= 0 else min(len(filenames), max_videos)
    extra_columns = [] if extra_columns is None else extra_columns
    for name, values in extra_columns:
        if values.shape[:3] != W_a.shape:
            raise ValueError(
                f"extra column {name!r} shape must start with {W_a.shape}, got {values.shape}"
            )

    lines: list[str] = []
    sep = "=" * 132
    lines.extend([
        sep,
        f"STAGE12 {stage_name.upper()} SEGMENT-LEVEL GROUND TRUTH vs PREDICTIONS",
        sep,
        "Scores: sigmoid(z-score(W over 25 LLP classes per segment)).",
        f"Zero weights excluded from z-score stats: {score_exclude_zero}.",
        f"Threshold: {threshold:.3f}. Pred_AV uses Pred_A AND Pred_V.",
        "Rows show GT/pred-active classes plus top-k classes by audio/visual score.",
        "Stage2 extra columns, when present: A_* means prior/signal used for audio reweighting; V_* means prior/signal used for visual reweighting.",
        sep,
        "",
    ])

    for vid_idx in range(num_videos):
        lines.extend([
            "",
            sep,
            f"VIDEO {vid_idx:04d}: {video_ids[vid_idx]}   filename={filenames[vid_idx]}",
            sep,
            "",
        ])
        for seg_idx in range(gt_a.shape[1]):
            top_a = _top_labels(scores_a[vid_idx, seg_idx], top_k)
            top_v = _top_labels(scores_v[vid_idx, seg_idx], top_k)
            lines.append(f"  SEGMENT {seg_idx}")
            if top_k > 0:
                lines.append(f"    topA: {top_a}")
                lines.append(f"    topV: {top_v}")
            lines.append("  " + "-" * 128)
            lines.append(
                "  "
                + f"{'Category':<28} | GT_A GT_V GT_AV | Pred_A Pred_V Pred_AV | "
                + f"A_Score V_Score | A_W V_W"
                + ("" if not extra_columns else " | " + " ".join(f"{name:>10}" for name, _ in extra_columns))
            )
            lines.append("  " + "-" * 128)

            active = (
                gt_a[vid_idx, seg_idx]
                | gt_v[vid_idx, seg_idx]
                | gt_av[vid_idx, seg_idx]
                | pred_a[vid_idx, seg_idx]
                | pred_v[vid_idx, seg_idx]
                | pred_av[vid_idx, seg_idx]
            ).astype(bool)
            if top_k > 0:
                top_idx = set(np.argsort(scores_a[vid_idx, seg_idx])[::-1][:top_k].tolist())
                top_idx.update(np.argsort(scores_v[vid_idx, seg_idx])[::-1][:top_k].tolist())
            else:
                top_idx = set()
            show_idx = list(range(len(LLP_CATS))) if all_classes else [
                i for i in range(len(LLP_CATS)) if active[i] or i in top_idx
            ]

            if not show_idx:
                lines.append("  (no active GT/pred classes)")
            for class_idx in show_idx:
                lines.append(
                    "  "
                    + f"{LLP_CATS[class_idx]:<28} | "
                    + f"  {int(gt_a[vid_idx, seg_idx, class_idx])}    "
                    + f"{int(gt_v[vid_idx, seg_idx, class_idx])}     "
                    + f"{int(gt_av[vid_idx, seg_idx, class_idx])}   | "
                    + f"   {int(pred_a[vid_idx, seg_idx, class_idx])}      "
                    + f"{int(pred_v[vid_idx, seg_idx, class_idx])}       "
                    + f"{int(pred_av[vid_idx, seg_idx, class_idx])}    | "
                    + f" {scores_a[vid_idx, seg_idx, class_idx]:.3f}   "
                    + f"{scores_v[vid_idx, seg_idx, class_idx]:.3f} | "
                    + f"{W_a[vid_idx, seg_idx, class_idx]:.4f} "
                    + f"{W_v[vid_idx, seg_idx, class_idx]:.4f}"
                    + (
                        ""
                        if not extra_columns
                        else " | "
                        + " ".join(
                            f"{values[vid_idx, seg_idx, class_idx]:10.4f}"
                            for _, values in extra_columns
                        )
                    )
                )
            lines.append("")

    if num_videos < len(filenames):
        lines.append(f"[truncated] wrote first {num_videos}/{len(filenames)} videos")
    out_path.write_text("\n".join(lines))


def write_stage_eval_outputs(
    out_dir: Path,
    stage_name: str,
    filenames: list[str],
    video_ids: list[str],
    W_a: np.ndarray,
    W_v: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    threshold: float,
    score_exclude_zero: bool,
    write_details: bool,
    detail_max_videos: int,
    detail_top_k: int,
    detail_all_classes: bool,
    detail_extra_columns: list[tuple[str, np.ndarray]] | None = None,
) -> dict[str, object]:
    pred = compute_stage_predictions(W_a, W_v, threshold, score_exclude_zero=score_exclude_zero)
    metrics = compute_stage_metrics(
        stage_name, pred, gt_a, gt_v, threshold, score_exclude_zero=score_exclude_zero
    )
    (out_dir / f"metrics_{stage_name}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False)
    )
    np.savez_compressed(
        out_dir / f"scores_preds_{stage_name}.npz",
        scores_a=pred["scores_a"],
        scores_v=pred["scores_v"],
        pred_a=pred["pred_a"],
        pred_v=pred["pred_v"],
        pred_av=pred["pred_av"],
        gt_a=gt_a.astype(np.uint8),
        gt_v=gt_v.astype(np.uint8),
        gt_av=(gt_a & gt_v).astype(np.uint8),
    )
    if write_details:
        write_segment_details(
            out_path=out_dir / f"segment_details_{stage_name}.txt",
            stage_name=stage_name,
            filenames=filenames,
            video_ids=video_ids,
            W_a=W_a,
            W_v=W_v,
            pred=pred,
            gt_a=gt_a,
            gt_v=gt_v,
            threshold=threshold,
            score_exclude_zero=score_exclude_zero,
            max_videos=detail_max_videos,
            top_k=detail_top_k,
            all_classes=detail_all_classes,
            extra_columns=detail_extra_columns,
        )
    return metrics
