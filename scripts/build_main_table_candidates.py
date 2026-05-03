from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avvp_stage12.data import build_dense_gt  # noqa: E402
from avvp_stage12.metrics import avvp_segment_f1, score_sparse_weights, sparse_weight_scores  # noqa: E402


DEFAULT_THRESHOLDS = (
    "0.50 0.525 0.55 0.575 0.60 0.625 0.65 0.675 0.70 "
    "0.725 0.75 0.775 0.80 0.825 0.85 0.875 0.90"
)


def parse_thresholds(text: str) -> list[float]:
    return [float(x) for x in text.replace(",", " ").split()]


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    cols = list(rows[0].keys())
    with path.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for row in rows:
            f.write("\t".join(str(row.get(col, "")) for col in cols) + "\n")


def f1_metrics(scores_a: np.ndarray, scores_v: np.ndarray, gt_a: np.ndarray, gt_v: np.ndarray, tau: float) -> dict[str, float]:
    pred_a = (scores_a > tau).astype(np.uint8).reshape(-1, scores_a.shape[-1])
    pred_v = (scores_v > tau).astype(np.uint8).reshape(-1, scores_v.shape[-1])
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


def l0_mean(weights: np.ndarray) -> float:
    return float((np.abs(weights) > 1e-8).sum(axis=-1).mean())


def eval_weight_pair(
    weights_a: np.ndarray,
    weights_v: np.ndarray,
    gt_a: np.ndarray,
    gt_v: np.ndarray,
    tau: float,
    k0: float,
    t_min: float,
    t_max: float,
    thresholds: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    fixed_scores_a, _, fixed_temp_a = sparse_weight_scores(
        weights_a, temperature=1.0, exclude_zero=True
    )
    fixed_scores_v, _, fixed_temp_v = sparse_weight_scores(
        weights_v, temperature=1.0, exclude_zero=True
    )
    fixed = f1_metrics(fixed_scores_a, fixed_scores_v, gt_a, gt_v, tau)
    rows.append({
        **fixed,
        "score_protocol": "fixed_T1",
        "tau": float(tau),
        "K0": "",
        "Tmin": 1.0,
        "Tmax": 1.0,
        "audio_temperature_mean": float(fixed_temp_a.mean()),
        "visual_temperature_mean": float(fixed_temp_v.mean()),
    })

    adaptive_scores_a, _, K_a, T_a = score_sparse_weights(
        weights_a,
        tau=tau,
        k0=k0,
        t_min=t_min,
        t_max=t_max,
        exclude_zero=True,
    )
    adaptive_scores_v, _, K_v, T_v = score_sparse_weights(
        weights_v,
        tau=tau,
        k0=k0,
        t_min=t_min,
        t_max=t_max,
        exclude_zero=True,
    )
    del K_a, K_v
    adaptive = f1_metrics(adaptive_scores_a, adaptive_scores_v, gt_a, gt_v, tau)
    rows.append({
        **adaptive,
        "score_protocol": "adaptive_K_over_K0",
        "tau": float(tau),
        "K0": float(k0),
        "Tmin": float(t_min),
        "Tmax": float(t_max),
        "audio_temperature_mean": float(T_a.mean()),
        "visual_temperature_mean": float(T_v.mean()),
    })

    oracle_candidates = []
    for threshold in thresholds:
        candidate = f1_metrics(fixed_scores_a, fixed_scores_v, gt_a, gt_v, threshold)
        oracle_candidates.append({
            **candidate,
            "score_protocol": "oracle_tau_fixed_T1",
            "tau": float(threshold),
            "K0": "",
            "Tmin": 1.0,
            "Tmax": 1.0,
            "audio_temperature_mean": float(fixed_temp_a.mean()),
            "visual_temperature_mean": float(fixed_temp_v.mean()),
        })
    rows.append(max(oracle_candidates, key=lambda row: float(row["mean_f1"])))
    return rows


def load_run_rows(
    sweep_dir: Path,
    run_dir: Path,
    tau: float,
    k0: float,
    t_min: float,
    t_max: float,
    thresholds: list[float],
) -> list[dict[str, object]]:
    meta = json.loads((run_dir / "meta.json").read_text())
    lam = float(meta["config"]["lambda_a"])
    kappa = float(meta["config"].get("kappa", 0.0))
    eta = float(meta["config"].get("eta", 0.0))
    filenames = list(meta["filenames"])
    gt_a = build_dense_gt(filenames, "audio")
    gt_v = build_dense_gt(filenames, "visual")
    rows: list[dict[str, object]] = []
    for stage, stage_label in [("stage1", "Stage1 sparse"), ("stage2", "Stage2 prior-guided")]:
        path_a = run_dir / f"W_a_{stage}.npy"
        path_v = run_dir / f"W_v_{stage}.npy"
        if not path_a.exists() or not path_v.exists():
            continue
        weights_a = np.load(path_a)
        weights_v = np.load(path_v)
        for row in eval_weight_pair(
            weights_a=weights_a,
            weights_v=weights_v,
            gt_a=gt_a,
            gt_v=gt_v,
            tau=tau,
            k0=k0,
            t_min=t_min,
            t_max=t_max,
            thresholds=thresholds,
        ):
            rows.append({
                "method": f"{stage_label}, {row['score_protocol']}",
                "stage": stage,
                "score_protocol": row["score_protocol"],
                "sweep": sweep_dir.name,
                "lambda_base": lam,
                "eta": eta,
                "kappa": kappa,
                "K0": row["K0"],
                "Tmin": row["Tmin"],
                "Tmax": row["Tmax"],
                "tau": row["tau"],
                "audio_f1": row["audio_f1"],
                "visual_f1": row["visual_f1"],
                "av_f1": row["av_f1"],
                "mean_f1": row["mean_f1"],
                "audio_l0_mean": l0_mean(weights_a),
                "visual_l0_mean": l0_mean(weights_v),
                "audio_temperature_mean": row["audio_temperature_mean"],
                "visual_temperature_mean": row["visual_temperature_mean"],
                "audio_pred_active_mean": row["audio_pred_active_mean"],
                "visual_pred_active_mean": row["visual_pred_active_mean"],
                "av_pred_active_mean": row["av_pred_active_mean"],
            })
    return rows


def baseline_rows(sweep_dirs: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sweep_dir in sweep_dirs:
        zs_path = sweep_dir / "zs_baseline.json"
        if zs_path.exists():
            zs = json.loads(zs_path.read_text())
            audio = float(zs["zs_clap_audio"])
            visual = float(zs["zs_clip_visual"])
            av = float(zs["zs_av_and"])
            rows.append({
                "method": "ZS dense cosine",
                "stage": "baseline",
                "score_protocol": "dense_zs",
                "sweep": sweep_dir.name,
                "lambda_base": "",
                "eta": "",
                "kappa": "",
                "K0": "",
                "Tmin": "",
                "Tmax": "",
                "tau": zs.get("threshold", 0.75),
                "audio_f1": audio,
                "visual_f1": visual,
                "av_f1": av,
                "mean_f1": (audio + visual + av) / 3.0,
                "audio_l0_mean": "",
                "visual_l0_mean": "",
                "audio_temperature_mean": "",
                "visual_temperature_mean": "",
                "audio_pred_active_mean": zs.get("audio_pred_active_mean", ""),
                "visual_pred_active_mean": zs.get("visual_pred_active_mean", ""),
                "av_pred_active_mean": zs.get("av_pred_active_mean", ""),
            })
            break
    for sweep_dir in sweep_dirs:
        av2a_path = sweep_dir / "av2a_baseline.json"
        if av2a_path.exists():
            av2a = json.loads(av2a_path.read_text())
            audio = float(av2a["audio"])
            visual = float(av2a["visual"])
            av = float(av2a["av"])
            rows.append({
                "method": "AV2A baseline",
                "stage": "baseline",
                "score_protocol": "av2a",
                "sweep": sweep_dir.name,
                "lambda_base": "",
                "eta": "",
                "kappa": "",
                "K0": "",
                "Tmin": "",
                "Tmax": "",
                "tau": "",
                "audio_f1": audio,
                "visual_f1": visual,
                "av_f1": av,
                "mean_f1": (audio + visual + av) / 3.0,
                "audio_l0_mean": "",
                "visual_l0_mean": "",
                "audio_temperature_mean": "",
                "visual_temperature_mean": "",
                "audio_pred_active_mean": "",
                "visual_pred_active_mean": "",
                "av_pred_active_mean": "",
            })
            break
    return rows


def select_best_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    best: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["sweep"]), str(row["stage"]), str(row["score_protocol"]))
        if row["stage"] == "baseline":
            key = ("baseline", str(row["stage"]), str(row["score_protocol"]))
        grouped.setdefault(key, []).append(row)
    for key in sorted(grouped):
        candidates = grouped[key]
        best.append(max(candidates, key=lambda row: float(row["mean_f1"])))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dirs",
        nargs="+",
        type=Path,
        default=[
            ROOT / "results" / "sweep_lambda_k4_e2",
            ROOT / "results" / "sweep_lambda_k8_e2",
        ],
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "main_table_candidates.tsv")
    parser.add_argument("--all-out", type=Path, default=ROOT / "results" / "main_table_candidates_all.tsv")
    parser.add_argument("--tau", type=float, default=0.75)
    parser.add_argument("--k0", type=float, default=16.0)
    parser.add_argument("--t-min", type=float, default=0.25)
    parser.add_argument("--t-max", type=float, default=1.25)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    sweep_dirs = [path.resolve() for path in args.sweep_dirs]
    rows: list[dict[str, object]] = baseline_rows(sweep_dirs)
    for sweep_dir in sweep_dirs:
        for run_dir in sorted(sweep_dir.glob("lam*")):
            if (run_dir / "meta.json").exists():
                rows.extend(load_run_rows(sweep_dir, run_dir, args.tau, args.k0, args.t_min, args.t_max, thresholds))

    best_rows = select_best_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.all_out, rows)
    write_tsv(args.out, best_rows)

    print(f"Wrote {args.out}")
    print(f"Wrote {args.all_out}")
    for row in best_rows:
        print(
            f"{row['sweep']} | {row['method']} | mean={float(row['mean_f1']):.4f} "
            f"A={float(row['audio_f1']):.4f} V={float(row['visual_f1']):.4f} "
            f"AV={float(row['av_f1']):.4f} lambda={row['lambda_base']}"
        )


if __name__ == "__main__":
    main()
