from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .constants import LLP_CATS
from .metrics import (
    apply_min_duration_filter,
    avvp_official_metrics,
    avvp_segment_f1,
    score_sparse_weights,
    sparse_weight_scores,
)


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


def _top_indices(scores: np.ndarray, top_k: int, eps: float = 1e-8) -> list[int]:
    if top_k <= 0 or float(np.max(scores)) <= eps:
        return []
    return np.argsort(scores)[::-1][:top_k].tolist()


def _top_labels(scores: np.ndarray, top_k: int) -> str:
    if top_k <= 0:
        return ""
    idx = _top_indices(scores, top_k)
    if not idx:
        return "(all zero)"
    return ", ".join(f"{LLP_CATS[i]}:{scores[i]:.3f}" for i in idx)


def _underline(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if ch != " ":
            out.append("̲")
    return "".join(out)


def compute_stage_predictions(
    W_a: np.ndarray,
    W_v: np.ndarray,
    threshold: float,
    score_exclude_zero: bool = True,
    score_mode: str = "adaptive_k",
    score_k0: float = 16.0,
    score_t_min: float = 0.25,
    score_t_max: float = 1.25,
    pred_min_duration: int = 1,
) -> dict[str, np.ndarray]:
    if score_mode == "adaptive_k":
        scores_a, pred_a, K_a, T_a = score_sparse_weights(
            W_a,
            tau=threshold,
            k0=score_k0,
            t_min=score_t_min,
            t_max=score_t_max,
            exclude_zero=score_exclude_zero,
        )
        scores_v, pred_v, K_v, T_v = score_sparse_weights(
            W_v,
            tau=threshold,
            k0=score_k0,
            t_min=score_t_min,
            t_max=score_t_max,
            exclude_zero=score_exclude_zero,
        )
    elif score_mode == "fixed_t":
        scores_a, K_a, T_a = sparse_weight_scores(W_a, temperature=1.0, exclude_zero=score_exclude_zero)
        scores_v, K_v, T_v = sparse_weight_scores(W_v, temperature=1.0, exclude_zero=score_exclude_zero)
        pred_a = (scores_a > threshold).astype(np.uint8)
        pred_v = (scores_v > threshold).astype(np.uint8)
    else:
        raise ValueError(f"unknown score_mode={score_mode!r}")
    pred_a = apply_min_duration_filter(pred_a, pred_min_duration)
    pred_v = apply_min_duration_filter(pred_v, pred_min_duration)
    pred_av = (pred_a & pred_v).astype(np.uint8)
    return {
        "scores_a": scores_a.astype(np.float32),
        "scores_v": scores_v.astype(np.float32),
        "pred_a": pred_a,
        "pred_v": pred_v,
        "pred_av": pred_av,
        "K_a": K_a.astype(np.float32),
        "K_v": K_v.astype(np.float32),
        "T_a": T_a.astype(np.float32),
        "T_v": T_v.astype(np.float32),
    }


def compute_stage_metrics(
    stage_name: str,
    pred: dict[str, np.ndarray],
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    threshold: float,
    score_exclude_zero: bool,
    score_mode: str,
    score_k0: float,
    score_t_min: float,
    score_t_max: float,
    pred_min_duration: int,
) -> dict[str, object]:
    gt_av = (gt_a & gt_v).astype(np.uint8)
    pred_a = pred["pred_a"]
    pred_v = pred["pred_v"]
    pred_av = pred["pred_av"]
    official = avvp_official_metrics(pred_a, pred_v, gt_a, gt_v, pred_av=pred_av, gt_av=gt_av)
    metrics = {
        "stage": stage_name,
        "threshold": float(threshold),
        "score_normalization": (
            "adaptive_k: active z-score over nonzero coefficients, "
            "T=clip(K/K0,Tmin,Tmax), sigmoid(z/T); fixed_t: same active z-score, T=1"
        ),
        "score_mode": score_mode,
        "score_exclude_zero": bool(score_exclude_zero),
        "score_k0": float(score_k0),
        "score_t_min": float(score_t_min),
        "score_t_max": float(score_t_max),
        "pred_min_duration": int(pred_min_duration),
        "audio_segment_f1": avvp_segment_f1(pred_a.reshape(-1, pred_a.shape[-1]), gt_a.reshape(-1, gt_a.shape[-1])),
        "visual_segment_f1": avvp_segment_f1(pred_v.reshape(-1, pred_v.shape[-1]), gt_v.reshape(-1, gt_v.shape[-1])),
        "av_segment_f1_and": avvp_segment_f1(pred_av.reshape(-1, pred_av.shape[-1]), gt_av.reshape(-1, gt_av.shape[-1])),
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
        "table_audio_segment": official["audio_segment_f1"],
        "table_audio_event": official["audio_event_f1"],
        "table_visual_segment": official["visual_segment_f1"],
        "table_visual_event": official["visual_event_f1"],
        "table_audio_visual_segment": official["audio_visual_segment_f1"],
        "table_audio_visual_event": official["audio_visual_event_f1"],
        "table_type_av_segment": official["type_av_segment_f1"],
        "table_type_av_event": official["type_av_event_f1"],
        "table_event_av_segment": official["event_av_segment_f1"],
        "table_event_av_event": official["event_av_event_f1"],
        "audio_pred_active_mean": float(pred_a.sum(axis=-1).mean()),
        "visual_pred_active_mean": float(pred_v.sum(axis=-1).mean()),
        "av_pred_active_mean": float(pred_av.sum(axis=-1).mean()),
        "audio_l0_mean": float(pred["K_a"].mean()),
        "visual_l0_mean": float(pred["K_v"].mean()),
        "audio_temperature_mean": float(pred["T_a"].mean()),
        "visual_temperature_mean": float(pred["T_v"].mean()),
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
    score_mode: str,
    score_k0: float,
    score_t_min: float,
    score_t_max: float,
    pred_min_duration: int,
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
        f"Score mode: {score_mode}; K0={score_k0:.3f}; Tmin={score_t_min:.3f}; Tmax={score_t_max:.3f}.",
        f"Zero weights excluded from z-score stats: {score_exclude_zero}.",
        f"Threshold: {threshold:.3f}. Pred_AV uses Pred_A AND Pred_V.",
        f"Post-hoc min-duration filter: {pred_min_duration} segment(s) per video/class.",
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
            header = (
                f"{'Category':<28} | "
                + f"{'GT_A':>4} {'GT_V':>4} {'GT_AV':>5} | "
                + f"{'Pred_A':>6} {'Pred_V':>6} {'Pred_AV':>7} | "
                + f"{'A_Score':>7} {'V_Score':>7} | "
                + f"{'A_W':>7} {'V_W':>7}"
                + (
                    ""
                    if not extra_columns
                    else " | " + " ".join(f"{name:>10}" for name, _ in extra_columns)
                )
            )
            sep_line = "-" * len(header)
            lines.append("  " + sep_line)
            lines.append("  " + header)
            lines.append("  " + sep_line)

            active = (
                gt_a[vid_idx, seg_idx]
                | gt_v[vid_idx, seg_idx]
                | gt_av[vid_idx, seg_idx]
                | pred_a[vid_idx, seg_idx]
                | pred_v[vid_idx, seg_idx]
                | pred_av[vid_idx, seg_idx]
            ).astype(bool)
            if top_k > 0:
                top_idx = set(_top_indices(scores_a[vid_idx, seg_idx], top_k))
                top_idx.update(_top_indices(scores_v[vid_idx, seg_idx], top_k))
            else:
                top_idx = set()
            show_idx = list(range(len(LLP_CATS))) if all_classes else [
                i for i in range(len(LLP_CATS)) if active[i] or i in top_idx
            ]

            if not show_idx:
                lines.append("  (no active GT/pred classes)")
            for class_idx in show_idx:
                ga = int(gt_a[vid_idx, seg_idx, class_idx])
                gv = int(gt_v[vid_idx, seg_idx, class_idx])
                a_score_s = f"{scores_a[vid_idx, seg_idx, class_idx]:>7.3f}"
                v_score_s = f"{scores_v[vid_idx, seg_idx, class_idx]:>7.3f}"
                a_w_s = f"{W_a[vid_idx, seg_idx, class_idx]:>7.4f}"
                v_w_s = f"{W_v[vid_idx, seg_idx, class_idx]:>7.4f}"
                if ga:
                    a_score_s = _underline(a_score_s)
                    a_w_s = _underline(a_w_s)
                if gv:
                    v_score_s = _underline(v_score_s)
                    v_w_s = _underline(v_w_s)
                extra_str = ""
                if extra_columns:
                    parts = []
                    for name, values in extra_columns:
                        cell = f"{values[vid_idx, seg_idx, class_idx]:>10.4f}"
                        prefix = name[:2]
                        if (prefix == "A_" and ga) or (prefix == "V_" and gv):
                            cell = _underline(cell)
                        parts.append(cell)
                    extra_str = " | " + " ".join(parts)
                lines.append(
                    "  "
                    + f"{LLP_CATS[class_idx]:<28} | "
                    + f"{ga:>4} "
                    + f"{gv:>4} "
                    + f"{int(gt_av[vid_idx, seg_idx, class_idx]):>5} | "
                    + f"{int(pred_a[vid_idx, seg_idx, class_idx]):>6} "
                    + f"{int(pred_v[vid_idx, seg_idx, class_idx]):>6} "
                    + f"{int(pred_av[vid_idx, seg_idx, class_idx]):>7} | "
                    + f"{a_score_s} {v_score_s} | "
                    + f"{a_w_s} {v_w_s}"
                    + extra_str
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
    score_mode: str = "adaptive_k",
    score_k0: float = 16.0,
    score_t_min: float = 0.25,
    score_t_max: float = 1.25,
    pred_min_duration: int = 1,
) -> dict[str, object]:
    pred = compute_stage_predictions(
        W_a,
        W_v,
        threshold,
        score_exclude_zero=score_exclude_zero,
        score_mode=score_mode,
        score_k0=score_k0,
        score_t_min=score_t_min,
        score_t_max=score_t_max,
        pred_min_duration=pred_min_duration,
    )
    metrics = compute_stage_metrics(
        stage_name,
        pred,
        gt_a,
        gt_v,
        threshold,
        score_exclude_zero=score_exclude_zero,
        score_mode=score_mode,
        score_k0=score_k0,
        score_t_min=score_t_min,
        score_t_max=score_t_max,
        pred_min_duration=pred_min_duration,
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
        K_a=pred["K_a"],
        K_v=pred["K_v"],
        T_a=pred["T_a"],
        T_v=pred["T_v"],
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
            score_mode=score_mode,
            score_k0=score_k0,
            score_t_min=score_t_min,
            score_t_max=score_t_max,
            pred_min_duration=pred_min_duration,
            max_videos=detail_max_videos,
            top_k=detail_top_k,
            all_classes=detail_all_classes,
            extra_columns=detail_extra_columns,
        )
    return metrics
