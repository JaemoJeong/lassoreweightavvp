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
from avvp_stage12.metrics import avvp_segment_f1, norm_similarities_np  # noqa: E402


def parse_thresholds(text: str) -> list[float]:
    values = [float(x) for x in text.replace(",", " ").split()]
    if not values:
        raise ValueError("empty threshold list")
    return values


def evaluate_weights(
    weights_a: np.ndarray,
    weights_v: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    thresholds: list[float],
    exclude_zero: bool,
) -> list[dict[str, float]]:
    scores_a = norm_similarities_np(weights_a, exclude_zero=exclude_zero)
    scores_v = norm_similarities_np(weights_v, exclude_zero=exclude_zero)
    gt_a_flat = gt_a.reshape(-1, gt_a.shape[-1])
    gt_v_flat = gt_v.reshape(-1, gt_v.shape[-1])
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        pred_a = (scores_a > threshold).astype(np.uint8).reshape(-1, weights_a.shape[-1])
        pred_v = (scores_v > threshold).astype(np.uint8).reshape(-1, weights_v.shape[-1])
        pred_av = pred_a & pred_v
        gt_av = gt_a_flat & gt_v_flat
        audio_f1 = avvp_segment_f1(pred_a, gt_a_flat)
        visual_f1 = avvp_segment_f1(pred_v, gt_v_flat)
        av_f1 = avvp_segment_f1(pred_av, gt_av)
        rows.append(
            {
                "threshold": float(threshold),
                "audio_f1": audio_f1,
                "visual_f1": visual_f1,
                "av_f1": av_f1,
                "mean_f1": (audio_f1 + visual_f1 + av_f1) / 3.0,
                "audio_pred_active_mean": float(pred_a.sum(axis=1).mean()),
                "visual_pred_active_mean": float(pred_v.sum(axis=1).mean()),
                "av_pred_active_mean": float(pred_av.sum(axis=1).mean()),
            }
        )
    return rows


def load_lambda_run(run_dir: Path) -> tuple[float, list[str]]:
    meta = json.loads((run_dir / "meta.json").read_text())
    return float(meta["config"]["lambda_a"]), list(meta["filenames"])


def best_row(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    return max(rows, key=lambda row: row[key])


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in cols) + "\n")


def plot_metric(
    records: list[dict[str, object]],
    out_dir: Path,
    stage: str,
    metric: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for exclude_zero, label, marker in [
        (True, "zero-excluded", "o"),
        (False, "zero-included", "s"),
    ]:
        sub = [r for r in records if r["stage"] == stage and r["exclude_zero"] == exclude_zero]
        lams = sorted({float(r["lambda"]) for r in sub})
        fixed = []
        best = []
        for lam in lams:
            rows = [r for r in sub if float(r["lambda"]) == lam]
            fixed_row = min(rows, key=lambda r: abs(float(r["threshold"]) - 0.75))
            fixed.append(float(fixed_row[metric]))
            best.append(max(float(r[metric]) for r in rows))
        ax.plot(lams, fixed, f"-{marker}", label=f"{label}, tau=0.75")
        ax.plot(lams, best, f"--{marker}", label=f"{label}, best tau")
    ax.set_xscale("log")
    ax.set_xlabel("lambda_base")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{stage}: {ylabel} threshold sensitivity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stage}_{metric}_threshold_sensitivity.png", dpi=180)
    fig.savefig(out_dir / f"{stage}_{metric}_threshold_sensitivity.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--thresholds",
        default="0.50 0.525 0.55 0.575 0.60 0.625 0.65 0.675 0.70 0.725 0.75 0.775 0.80 0.825 0.85 0.875 0.90",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir
    out_dir = args.out_dir or (sweep_dir / "threshold_sensitivity")
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = parse_thresholds(args.thresholds)

    records: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for run_dir in sorted(sweep_dir.glob("lam*")):
        if not (run_dir / "meta.json").exists():
            continue
        lam, filenames = load_lambda_run(run_dir)
        gt_a = build_dense_gt(filenames, "audio")
        gt_v = build_dense_gt(filenames, "visual")
        for stage in ["stage1", "stage2"]:
            weights_a = np.load(run_dir / f"W_a_{stage}.npy")
            weights_v = np.load(run_dir / f"W_v_{stage}.npy")
            for exclude_zero in [True, False]:
                rows = evaluate_weights(
                    weights_a=weights_a,
                    weights_v=weights_v,
                    gt_a=gt_a,
                    gt_v=gt_v,
                    thresholds=thresholds,
                    exclude_zero=exclude_zero,
                )
                for row in rows:
                    records.append(
                        {
                            "lambda": lam,
                            "stage": stage,
                            "exclude_zero": exclude_zero,
                            **row,
                        }
                    )
                fixed = min(rows, key=lambda row: abs(row["threshold"] - 0.75))
                best_mean = best_row(rows, "mean_f1")
                best_audio = best_row(rows, "audio_f1")
                best_visual = best_row(rows, "visual_f1")
                best_av = best_row(rows, "av_f1")
                summary.append(
                    {
                        "lambda": lam,
                        "stage": stage,
                        "exclude_zero": exclude_zero,
                        "fixed_tau": fixed["threshold"],
                        "fixed_audio_f1": fixed["audio_f1"],
                        "fixed_visual_f1": fixed["visual_f1"],
                        "fixed_av_f1": fixed["av_f1"],
                        "fixed_mean_f1": fixed["mean_f1"],
                        "best_mean_tau": best_mean["threshold"],
                        "best_mean_f1": best_mean["mean_f1"],
                        "best_mean_audio_f1": best_mean["audio_f1"],
                        "best_mean_visual_f1": best_mean["visual_f1"],
                        "best_mean_av_f1": best_mean["av_f1"],
                        "best_audio_tau": best_audio["threshold"],
                        "best_audio_f1": best_audio["audio_f1"],
                        "best_visual_tau": best_visual["threshold"],
                        "best_visual_f1": best_visual["visual_f1"],
                        "best_av_tau": best_av["threshold"],
                        "best_av_f1": best_av["av_f1"],
                    }
                )

    records = sorted(
        records,
        key=lambda r: (float(r["lambda"]), str(r["stage"]), bool(r["exclude_zero"]), float(r["threshold"])),
    )
    summary = sorted(
        summary,
        key=lambda r: (float(r["lambda"]), str(r["stage"]), bool(r["exclude_zero"])),
    )
    (out_dir / "threshold_sensitivity.json").write_text(json.dumps(records, indent=2))
    (out_dir / "threshold_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    write_tsv(out_dir / "threshold_sensitivity.tsv", records)
    write_tsv(out_dir / "threshold_sensitivity_summary.tsv", summary)

    for stage in ["stage1", "stage2"]:
        for metric, ylabel in [
            ("audio_f1", "Audio segment F1"),
            ("visual_f1", "Visual segment F1"),
            ("av_f1", "AV-AND segment F1"),
            ("mean_f1", "Mean(Audio, Visual, AV) F1"),
        ]:
            plot_metric(records, out_dir, stage, metric, ylabel)

    print(f"Saved threshold sensitivity outputs to {out_dir}")
    stage2_summary = [r for r in summary if r["stage"] == "stage2"]
    for exclude_zero in [True, False]:
        label = "zero-excluded" if exclude_zero else "zero-included"
        sub = [r for r in stage2_summary if r["exclude_zero"] == exclude_zero]
        best_mean = max(sub, key=lambda r: float(r["best_mean_f1"]))
        fixed_best = max(sub, key=lambda r: float(r["fixed_mean_f1"]))
        print(
            f"{label}: best tau-swept mean at lambda={best_mean['lambda']} "
            f"tau={best_mean['best_mean_tau']} mean={best_mean['best_mean_f1']:.4f} "
            f"(A={best_mean['best_mean_audio_f1']:.4f}, V={best_mean['best_mean_visual_f1']:.4f}, "
            f"AV={best_mean['best_mean_av_f1']:.4f}); "
            f"best fixed tau=0.75 lambda={fixed_best['lambda']} mean={fixed_best['fixed_mean_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
