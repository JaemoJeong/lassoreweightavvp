from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avvp_stage12.data import build_dense_gt  # noqa: E402
from avvp_stage12.metrics import avvp_segment_f1, score_sparse_weights, sparse_weight_scores  # noqa: E402


def parse_float_list(text: str) -> list[float]:
    values = [float(x) for x in text.replace(",", " ").split()]
    if not values:
        raise ValueError("empty float list")
    return values


def lam_from_meta(run_dir: Path) -> float:
    meta = json.loads((run_dir / "meta.json").read_text())
    return float(meta["config"]["lambda_a"])


def filenames_from_meta(run_dir: Path) -> list[str]:
    meta = json.loads((run_dir / "meta.json").read_text())
    return list(meta["filenames"])


def active_temp_scores(
    weights: np.ndarray,
    k0: float,
    t_min: float,
    t_max: float,
    direction: str = "direct",
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "direct":
        probs, _, _, temperature = score_sparse_weights(
            weights,
            tau=0.75,
            k0=k0,
            t_min=t_min,
            t_max=t_max,
            exclude_zero=True,
            zero_eps=eps,
        )
        return probs, temperature
    if direction == "inverse":
        _, k_count, _ = sparse_weight_scores(weights, temperature=1.0, exclude_zero=True, zero_eps=eps)
        temperature = np.clip(float(k0) / (k_count + eps), float(t_min), float(t_max)).astype(np.float32)
        probs, _, temperature = sparse_weight_scores(
            weights, temperature=temperature, exclude_zero=True, zero_eps=eps
        )
        return probs, temperature
    raise ValueError(f"unknown temperature direction: {direction!r}")


def fixed_temp_scores(weights: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    probs, k_count, temp = sparse_weight_scores(weights, temperature=1.0, exclude_zero=True, zero_eps=eps)
    del k_count
    return probs, temp


def f1_row(
    scores_a: np.ndarray,
    scores_v: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred_a = (scores_a > threshold).astype(np.uint8).reshape(-1, scores_a.shape[-1])
    pred_v = (scores_v > threshold).astype(np.uint8).reshape(-1, scores_v.shape[-1])
    pred_av = pred_a & pred_v
    gt_a_flat = gt_a.reshape(-1, gt_a.shape[-1])
    gt_v_flat = gt_v.reshape(-1, gt_v.shape[-1])
    gt_av = gt_a_flat & gt_v_flat
    audio_f1 = avvp_segment_f1(pred_a, gt_a_flat)
    visual_f1 = avvp_segment_f1(pred_v, gt_v_flat)
    av_f1 = avvp_segment_f1(pred_av, gt_av)
    return {
        "audio_f1": audio_f1,
        "visual_f1": visual_f1,
        "av_f1": av_f1,
        "mean_f1": (audio_f1 + visual_f1 + av_f1) / 3.0,
        "audio_pred_active_mean": float(pred_a.sum(axis=1).mean()),
        "visual_pred_active_mean": float(pred_v.sum(axis=1).mean()),
        "av_pred_active_mean": float(pred_av.sum(axis=1).mean()),
    }


def l0_mean(weights: np.ndarray, eps: float = 1e-8) -> float:
    return float((np.abs(weights) > eps).sum(axis=-1).mean())


def eval_fixed_and_oracle(
    weights_a: np.ndarray,
    weights_v: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    thresholds: list[float],
    fixed_threshold: float,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    scores_a, temp_a = fixed_temp_scores(weights_a)
    scores_v, temp_v = fixed_temp_scores(weights_v)
    rows = []
    for threshold in thresholds:
        row = f1_row(scores_a, scores_v, gt_a, gt_v, threshold)
        row.update(
            {
                "mode": "oracle_tau_sweep",
                "threshold": float(threshold),
                "k0": 1.0,
                "t0": 1.0,
                "t_min": 1.0,
                "t_max": 1.0,
                "audio_temperature_mean": float(temp_a.mean()),
                "visual_temperature_mean": float(temp_v.mean()),
            }
        )
        rows.append(row)
    fixed = f1_row(scores_a, scores_v, gt_a, gt_v, fixed_threshold)
    fixed.update(
        {
            "mode": "fixed_T1",
            "threshold": float(fixed_threshold),
            "k0": 1.0,
            "t0": 1.0,
            "t_min": 1.0,
            "t_max": 1.0,
            "audio_temperature_mean": float(temp_a.mean()),
            "visual_temperature_mean": float(temp_v.mean()),
        }
    )
    return rows, fixed


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in cols) + "\n")


def plot_stage(records: list[dict[str, object]], out_dir: Path, stage: str) -> None:
    stage_rows = [r for r in records if r["stage"] == stage]
    lams = sorted({float(r["lambda"]) for r in stage_rows})
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4), facecolor="white")
    specs = [
        ("audio_f1", "Audio F1"),
        ("visual_f1", "Visual F1"),
        ("av_f1", "AV-AND F1"),
        ("mean_f1", "Mean F1"),
    ]
    modes = [
        ("fixed_T1", "fixed T=1, tau=0.75", "-o"),
        ("adaptive_K_over_K0", "adaptive K/K0", "-D"),
        ("adaptive_K0_over_K", "adaptive K0/K", "-s"),
        ("oracle_tau_best", "oracle best tau", "--^"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        for mode, label, marker in modes:
            vals = []
            for lam in lams:
                candidates = [
                    r for r in stage_rows
                    if float(r["lambda"]) == lam and r["mode"] == mode
                ]
                vals.append(float(candidates[0][metric]) if candidates else np.nan)
            ax.plot(lams, vals, marker, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel(title)
        ax.set_title(f"{stage} {title}")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stage}_active_count_temperature.png", dpi=180)
    fig.savefig(out_dir / f"{stage}_active_count_temperature.pdf")
    plt.close(fig)


def plot_l0_temperature(records: list[dict[str, object]], out_dir: Path) -> None:
    rows = [r for r in records if r["mode"] == "adaptive_K_over_K0"]
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor="white")
    for stage in ["stage1", "stage2"]:
        sub = sorted([r for r in rows if r["stage"] == stage], key=lambda r: float(r["lambda"]))
        if not sub:
            continue
        lams = [float(r["lambda"]) for r in sub]
        axes[0].plot(lams, [float(r["audio_l0_mean"]) for r in sub], "-o", label=f"Audio {stage}")
        axes[0].plot(lams, [float(r["visual_l0_mean"]) for r in sub], "-s", label=f"Visual {stage}")
        axes[1].plot(lams, [float(r["audio_temperature_mean"]) for r in sub], "-o", label=f"Audio {stage}")
        axes[1].plot(lams, [float(r["visual_temperature_mean"]) for r in sub], "-s", label=f"Visual {stage}")
    for ax, ylabel in zip(axes, ["mean K = L0", "mean T"]):
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "l0_and_temperature_vs_lambda.png", dpi=180)
    fig.savefig(out_dir / "l0_and_temperature_vs_lambda.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate active-count adaptive temperature on saved Stage1/2 weights."
    )
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fixed-threshold", type=float, default=0.75)
    parser.add_argument(
        "--thresholds",
        default="0.50 0.525 0.55 0.575 0.60 0.625 0.65 0.675 0.70 0.725 0.75 0.775 0.80 0.825 0.85 0.875 0.90",
    )
    parser.add_argument("--k0", type=float, default=16.0)
    parser.add_argument("--t-min", type=float, default=0.25)
    parser.add_argument("--t-max", type=float, default=1.25)
    parser.add_argument(
        "--also-inverse",
        action="store_true",
        help="also evaluate the rejected inverse direction T=K0/K for ablation",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="also sweep several clipping ranges and report best adaptive setting",
    )
    parser.add_argument("--grid-t-mins", default="0.25 0.5 0.75")
    parser.add_argument("--grid-t-maxs", default="1.25 1.5 2.0 3.0")
    parser.add_argument("--grid-k0s", default=None)
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    out_dir = args.out_dir or (sweep_dir / "active_count_temperature")
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_float_list(args.thresholds)
    grid_t_mins = parse_float_list(args.grid_t_mins)
    grid_t_maxs = parse_float_list(args.grid_t_maxs)
    grid_k0s = [args.k0] if args.grid_k0s is None else parse_float_list(args.grid_k0s)

    records: list[dict[str, object]] = []
    oracle_records: list[dict[str, object]] = []
    grid_records: list[dict[str, object]] = []

    for run_dir in sorted(sweep_dir.glob("lam*")):
        if not (run_dir / "meta.json").exists():
            continue
        lam = lam_from_meta(run_dir)
        filenames = filenames_from_meta(run_dir)
        gt_a = build_dense_gt(filenames, "audio")
        gt_v = build_dense_gt(filenames, "visual")
        for stage in ["stage1", "stage2"]:
            path_a = run_dir / f"W_a_{stage}.npy"
            path_v = run_dir / f"W_v_{stage}.npy"
            if not path_a.exists() or not path_v.exists():
                continue
            weights_a = np.load(path_a)
            weights_v = np.load(path_v)
            base = {
                "lambda": lam,
                "stage": stage,
                "audio_l0_mean": l0_mean(weights_a),
                "visual_l0_mean": l0_mean(weights_v),
            }

            oracle_rows, fixed = eval_fixed_and_oracle(
                weights_a=weights_a,
                weights_v=weights_v,
                gt_a=gt_a,
                gt_v=gt_v,
                thresholds=thresholds,
                fixed_threshold=args.fixed_threshold,
            )
            records.append({**base, **fixed})
            for row in oracle_rows:
                oracle_records.append({**base, **row})
            best_oracle = max(oracle_rows, key=lambda r: float(r["mean_f1"]))
            records.append({**base, **best_oracle, "mode": "oracle_tau_best"})

            scores_a, temp_a = active_temp_scores(
                weights_a,
                k0=args.k0,
                t_min=args.t_min,
                t_max=args.t_max,
                direction="direct",
            )
            scores_v, temp_v = active_temp_scores(
                weights_v,
                k0=args.k0,
                t_min=args.t_min,
                t_max=args.t_max,
                direction="direct",
            )
            adaptive = f1_row(scores_a, scores_v, gt_a, gt_v, args.fixed_threshold)
            records.append(
                {
                    **base,
                    **adaptive,
                    "mode": "adaptive_K_over_K0",
                    "threshold": float(args.fixed_threshold),
                    "k0": float(args.k0),
                    "t0": 1.0,
                    "t_min": float(args.t_min),
                    "t_max": float(args.t_max),
                    "audio_temperature_mean": float(temp_a.mean()),
                    "visual_temperature_mean": float(temp_v.mean()),
                }
            )

            if args.also_inverse:
                inverse_scores_a, inverse_temp_a = active_temp_scores(
                    weights_a,
                    k0=args.k0,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    direction="inverse",
                )
                inverse_scores_v, inverse_temp_v = active_temp_scores(
                    weights_v,
                    k0=args.k0,
                    t_min=args.t_min,
                    t_max=args.t_max,
                    direction="inverse",
                )
                inverse = f1_row(inverse_scores_a, inverse_scores_v, gt_a, gt_v, args.fixed_threshold)
                records.append(
                    {
                        **base,
                        **inverse,
                        "mode": "adaptive_K0_over_K",
                        "threshold": float(args.fixed_threshold),
                        "k0": float(args.k0),
                        "t0": 1.0,
                        "t_min": float(args.t_min),
                        "t_max": float(args.t_max),
                        "audio_temperature_mean": float(inverse_temp_a.mean()),
                        "visual_temperature_mean": float(inverse_temp_v.mean()),
                    }
                )

            if args.grid:
                for grid_k0 in grid_k0s:
                    for t_min in grid_t_mins:
                        for t_max in grid_t_maxs:
                            if t_max < t_min:
                                continue
                            g_scores_a, g_temp_a = active_temp_scores(
                                weights_a,
                                k0=grid_k0,
                                t_min=t_min,
                                t_max=t_max,
                                direction="direct",
                            )
                            g_scores_v, g_temp_v = active_temp_scores(
                                weights_v,
                                k0=grid_k0,
                                t_min=t_min,
                                t_max=t_max,
                                direction="direct",
                            )
                            grid_row = f1_row(g_scores_a, g_scores_v, gt_a, gt_v, args.fixed_threshold)
                            grid_records.append(
                                {
                                    **base,
                                    **grid_row,
                                    "mode": "adaptive_K_over_K0_grid",
                                    "threshold": float(args.fixed_threshold),
                                    "k0": float(grid_k0),
                                    "t0": 1.0,
                                    "t_min": float(t_min),
                                    "t_max": float(t_max),
                                    "audio_temperature_mean": float(g_temp_a.mean()),
                                    "visual_temperature_mean": float(g_temp_v.mean()),
                                }
                            )
                            if args.also_inverse:
                                inv_scores_a, inv_temp_a = active_temp_scores(
                                    weights_a,
                                    k0=grid_k0,
                                    t_min=t_min,
                                    t_max=t_max,
                                    direction="inverse",
                                )
                                inv_scores_v, inv_temp_v = active_temp_scores(
                                    weights_v,
                                    k0=grid_k0,
                                    t_min=t_min,
                                    t_max=t_max,
                                    direction="inverse",
                                )
                                inverse_grid_row = f1_row(inv_scores_a, inv_scores_v, gt_a, gt_v, args.fixed_threshold)
                                grid_records.append(
                                    {
                                        **base,
                                        **inverse_grid_row,
                                        "mode": "adaptive_K0_over_K_grid",
                                        "threshold": float(args.fixed_threshold),
                                        "k0": float(grid_k0),
                                        "t0": 1.0,
                                        "t_min": float(t_min),
                                        "t_max": float(t_max),
                                        "audio_temperature_mean": float(inv_temp_a.mean()),
                                        "visual_temperature_mean": float(inv_temp_v.mean()),
                                    }
                                )

    records = sorted(records, key=lambda r: (float(r["lambda"]), str(r["stage"]), str(r["mode"])))
    oracle_records = sorted(
        oracle_records,
        key=lambda r: (float(r["lambda"]), str(r["stage"]), float(r["threshold"])),
    )
    grid_records = sorted(
        grid_records,
        key=lambda r: (float(r["lambda"]), str(r["stage"]), float(r["t_min"]), float(r["t_max"])),
    )

    (out_dir / "active_count_temperature.json").write_text(json.dumps(records, indent=2))
    (out_dir / "oracle_threshold_sweep.json").write_text(json.dumps(oracle_records, indent=2))
    write_tsv(out_dir / "active_count_temperature.tsv", records)
    write_tsv(out_dir / "oracle_threshold_sweep.tsv", oracle_records)
    if grid_records:
        (out_dir / "adaptive_temperature_grid.json").write_text(json.dumps(grid_records, indent=2))
        write_tsv(out_dir / "adaptive_temperature_grid.tsv", grid_records)

    for stage in ["stage1", "stage2"]:
        plot_stage(records, out_dir, stage)
    plot_l0_temperature(records, out_dir)

    if grid_records:
        best_grid = {}
        for stage in ["stage1", "stage2"]:
            sub = [r for r in grid_records if r["stage"] == stage]
            if sub:
                best_grid[stage] = max(sub, key=lambda r: float(r["mean_f1"]))
        (out_dir / "best_adaptive_grid.json").write_text(json.dumps(best_grid, indent=2))

    print(f"Saved active-count temperature outputs to {out_dir}")
    for stage in ["stage1", "stage2"]:
        sub = [r for r in records if r["stage"] == stage]
        if not sub:
            continue
        for mode in ["fixed_T1", "adaptive_K_over_K0", "adaptive_K0_over_K", "oracle_tau_best"]:
            mode_rows = [r for r in sub if r["mode"] == mode]
            if not mode_rows:
                continue
            best = max(mode_rows, key=lambda r: float(r["mean_f1"]))
            print(
                f"{stage} {mode}: best mean={best['mean_f1']:.4f} "
                f"lambda={best['lambda']} tau={best['threshold']} "
                f"A={best['audio_f1']:.4f} V={best['visual_f1']:.4f} AV={best['av_f1']:.4f}"
            )
    if grid_records:
        for stage in ["stage1", "stage2"]:
            sub = [r for r in grid_records if r["stage"] == stage]
            if not sub:
                continue
            for mode in ["adaptive_K_over_K0_grid", "adaptive_K0_over_K_grid"]:
                mode_rows = [r for r in sub if r["mode"] == mode]
                if not mode_rows:
                    continue
                best = max(mode_rows, key=lambda r: float(r["mean_f1"]))
                print(
                    f"{stage} {mode}: best mean={best['mean_f1']:.4f} "
                    f"lambda={best['lambda']} k0={best['k0']} "
                    f"t_min={best['t_min']} t_max={best['t_max']} "
                    f"A={best['audio_f1']:.4f} V={best['visual_f1']:.4f} AV={best['av_f1']:.4f}"
                )


if __name__ == "__main__":
    main()
