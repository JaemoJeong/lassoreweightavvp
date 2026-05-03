"""
λ sweep driver + evaluator + plotter for the clean stage1/2 pipeline.

This script does NOT modify any code under avvp_stage12/. It only:
  1. Calls run_llp_stage12.py as a subprocess for each (κ, η, λ) combination.
  2. Loads the saved W_*_stage{1,2}.npy + meta.json from each run.
  3. Builds dense GT via avvp_stage12.data.build_dense_gt.
  4. Computes a fixed-threshold AVVP segment F1 using avvp_stage12.metrics helpers,
     uniformly across all (λ, modality, stage).
  5. Writes a single-file plot summary (PNG + PDF + sweep_results.json).

Convention:
  W (n, 10, K) → norm_similarities_np (label-axis z-score + sigmoid;
                  exact-zero weights excluded from z-score stats by default)
              → > THR (default 0.75) → binary pred (n, 10, K)
              → avvp_segment_f1 against dense GT (n, 10, K)
              → mean over n*10 segments (clip-segment macro-style F1)

Sanity sweep (η=0):
  Stage 2 should be (essentially) identical to Stage 1 because the weighted
  penalty collapses to exp(0) * λ_base = λ_base everywhere.
  We log max-abs(W_stage2 - W_stage1) and the F1 delta to confirm.

NOTE on transductive scope (per agreement):
  • centering mean is computed on LLP test features (test-feature mean,
    label-free, transductive). Recorded in meta_transductive.json.
  • norm_similarities mean/std is per-sample over the class axis,
    not transductive (no leakage).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add parent so we can `import avvp_stage12.*` from this script (read-only access)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from avvp_stage12.baselines import compute_zero_shot_baseline
from avvp_stage12.data import build_dense_gt, load_llp_metadata
from avvp_stage12.metrics import (
    norm_similarities_np,
    avvp_segment_f1,
)
from avvp_stage12.constants import LLP_CATS

THR = 0.75
RUNNER = ROOT / "run_llp_stage12.py"
PYBIN = "/home/jaemo/miniconda3/envs/av2a_fresh/bin/python"
DEFAULT_AV2A_METRICS_PATH = Path(
    "/home/jaemo/AV2A_pristine/runs/llp_clipclap_20260420_l2norm_full/per_class_metrics.json"
)


def lam_tag(lam: float) -> str:
    return f"lam{('%.4f' % lam).rstrip('0').rstrip('.').replace('.', 'p')}"


def run_one(out_dir: Path, lam: float, kappa: float, eta: float,
            rho_min: float, rho_max: float, fista_iters: int, device: str,
            mean_source: str, audio_mean_path: str | None, visual_mean_path: str | None,
            limit_videos: int, score_include_zero: bool, stage2_prior_mode: str, dry_run: bool) -> None:
    cmd = [
        PYBIN, str(RUNNER),
        "--out-dir", str(out_dir),
        "--lambda-base", str(lam),
        "--kappa", str(kappa),
        "--eta", str(eta),
        "--rho-min", str(rho_min),
        "--rho-max", str(rho_max),
        "--stage2-prior-mode", stage2_prior_mode,
        "--fista-iters", str(fista_iters),
        "--device", device,
        "--mean-source", mean_source,
        "--no-details",
    ]
    if score_include_zero:
        cmd += ["--score-include-zero"]
    if audio_mean_path:
        cmd += ["--audio-mean-path", audio_mean_path]
    if visual_mean_path:
        cmd += ["--visual-mean-path", visual_mean_path]
    if limit_videos > 0:
        cmd += ["--limit-videos", str(limit_videos)]
    print("[run]", " ".join(cmd[2:]))
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "run.log"
    with log_file.open("w") as lf:
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"run failed for {out_dir} (see {log_file})")


def fixed_threshold_pred(weights: np.ndarray, thr: float = THR, exclude_zero: bool = True) -> np.ndarray:
    """W: (n, 10, K) → binary pred (n, 10, K) via label-axis norm + threshold."""
    probs = norm_similarities_np(weights, exclude_zero=exclude_zero)  # axis=-1 over K
    return (probs > thr).astype(np.uint8)


def f1_from_W(W: np.ndarray, GT: np.ndarray, exclude_zero: bool = True) -> float:
    """Flatten (n, 10, K) → (n*10, K) for avvp_segment_f1 (per-row macro-style)."""
    pred = fixed_threshold_pred(W, exclude_zero=exclude_zero).reshape(-1, W.shape[-1])
    gt = GT.reshape(-1, GT.shape[-1])
    return avvp_segment_f1(pred, gt)


def evaluate_run(out_dir: Path, filenames: list[str], score_exclude_zero: bool) -> dict[str, float]:
    """Load saved npy + GT, compute fixed-thr F1 + read recon/L0 from meta."""
    meta = json.loads((out_dir / "meta.json").read_text())
    Wa1 = np.load(out_dir / "W_a_stage1.npy")
    Wv1 = np.load(out_dir / "W_v_stage1.npy")
    Wa2 = np.load(out_dir / "W_a_stage2.npy")
    Wv2 = np.load(out_dir / "W_v_stage2.npy")

    GT_A = build_dense_gt(filenames, "audio")
    GT_V = build_dense_gt(filenames, "visual")
    if GT_A.shape[0] != Wa1.shape[0]:
        # bundle keeps only valid videos; rebuild GT for kept filenames
        kept = meta["filenames"]
        GT_A = build_dense_gt(kept, "audio")
        GT_V = build_dense_gt(kept, "visual")

    f1_a1 = f1_from_W(Wa1, GT_A, exclude_zero=score_exclude_zero)
    f1_a2 = f1_from_W(Wa2, GT_A, exclude_zero=score_exclude_zero)
    f1_v1 = f1_from_W(Wv1, GT_V, exclude_zero=score_exclude_zero)
    f1_v2 = f1_from_W(Wv2, GT_V, exclude_zero=score_exclude_zero)

    # Sanity diagnostics: stage1 vs stage2 W max-abs diff (only meaningful at η=0)
    diff_a = float(np.max(np.abs(Wa2 - Wa1)))
    diff_v = float(np.max(np.abs(Wv2 - Wv1)))

    return {
        "f1_audio_stage1": f1_a1,
        "f1_audio_stage2": f1_a2,
        "f1_visual_stage1": f1_v1,
        "f1_visual_stage2": f1_v2,
        "recon_audio_stage1": meta["audio_summary_stage1"]["recon_mean"],
        "recon_audio_stage2": meta["audio_summary_stage2"]["recon_mean"],
        "recon_visual_stage1": meta["visual_summary_stage1"]["recon_mean"],
        "recon_visual_stage2": meta["visual_summary_stage2"]["recon_mean"],
        "l0_audio_stage1": meta["audio_summary_stage1"]["l0_mean"],
        "l0_audio_stage2": meta["audio_summary_stage2"]["l0_mean"],
        "l0_visual_stage1": meta["visual_summary_stage1"]["l0_mean"],
        "l0_visual_stage2": meta["visual_summary_stage2"]["l0_mean"],
        "max_abs_W_diff_audio": diff_a,
        "max_abs_W_diff_visual": diff_v,
        "score_exclude_zero": bool(score_exclude_zero),
    }


def write_mean_meta(out_root: Path, mean_source: str, audio_mean_path: str | None, visual_mean_path: str | None) -> None:
    if mean_source == "llp":
        info = {
            "mean_source": "llp_test_dataset",
            "segment_mean_scope": "all valid LLP test segments (per modality, post L2-norm)",
            "video_mean_scope": "all valid LLP test videos (per modality, post L2-norm)",
            "scope_label": "test-feature mean / transductive / label-free",
            "purpose": "exploratory λ sweep — compare against external reference means",
        }
        out_name = "meta_transductive.json"
    else:
        info = {
            "mean_source": "external_reference",
            "audio_mean_path": audio_mean_path,
            "visual_mean_path": visual_mean_path,
            "scope_label": "backbone reference mean / non-LLP",
            "purpose": "mean-source comparison against LLP test-feature mean",
        }
        out_name = "meta_reference.json"
    (out_root / out_name).write_text(json.dumps(info, indent=2))


def mean_source_label(mean_source: str, audio_mean_path: str | None, visual_mean_path: str | None) -> str:
    if mean_source == "llp":
        return "test-feature mean, transductive"
    audio = Path(audio_mean_path).stem if audio_mean_path else "default audio reference"
    visual = Path(visual_mean_path).stem if visual_mean_path else "default visual reference"
    return f"external mean: audio={audio}, visual={visual}"


def _as_fraction(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def load_av2a_baseline(metrics_path: str | None) -> dict[str, object] | None:
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
        "av": _as_fraction(overall.get("F_seg_av") or overall.get("av_segment_f1") or overall.get("av_segment_f1_and")),
    }
    return baseline


def _draw_av2a_baselines(ax, baseline: dict[str, object] | None) -> None:
    if baseline is None:
        return
    specs = [
        ("audio", "#1f77b4", "AV2A audio"),
        ("visual", "#2ca02c", "AV2A visual"),
        ("av", "#666666", "AV2A AV"),
    ]
    for key, color, label in specs:
        value = baseline.get(key)
        if value is None:
            continue
        ax.axhline(float(value), color=color, linestyle=":", linewidth=1.6, alpha=0.8, label=label)


def _draw_zs_baselines(ax, baseline: dict[str, object] | None) -> None:
    if baseline is None:
        return
    specs = [
        ("zs_clap_audio", "#1f77b4", "ZS-CLAP audio"),
        ("zs_clip_visual", "#2ca02c", "ZS-CLIP visual"),
    ]
    for key, color, label in specs:
        value = baseline.get(key)
        if value is None:
            continue
        ax.axhline(float(value), color=color, linestyle="-.", linewidth=1.5, alpha=0.75, label=label)


def make_plots(
    records: list[dict],
    out_root: Path,
    sanity: dict | None,
    av2a_baseline: dict[str, object] | None,
    zs_baseline: dict[str, object] | None,
) -> None:
    records_main = sorted([r for r in records if r["is_main"]], key=lambda r: r["lambda"])
    lams = [r["lambda"] for r in records_main]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="white")

    # --- F1 plot
    ax = axes[0]
    ax.plot(lams, [r["f1_audio_stage1"] for r in records_main], "-o", color="#1f77b4", label="Audio · stage1")
    ax.plot(lams, [r["f1_audio_stage2"] for r in records_main], "--o", color="#1f77b4", label="Audio · stage2")
    ax.plot(lams, [r["f1_visual_stage1"] for r in records_main], "-s", color="#2ca02c", label="Visual · stage1")
    ax.plot(lams, [r["f1_visual_stage2"] for r in records_main], "--s", color="#2ca02c", label="Visual · stage2")
    _draw_av2a_baselines(ax, av2a_baseline)
    _draw_zs_baselines(ax, zs_baseline)
    ax.set_xscale("log")
    ax.set_xlabel("λ_base")
    ax.set_ylabel(f"AVVP segment F1 (fixed thr={THR})")
    ax.set_title(
        f"F1 vs λ   (κ={records_main[0]['kappa']:.2f}, η={records_main[0]['eta']:.2f}, "
        f"prior={records_main[0].get('stage2_prior_mode', 'full')}, fixed thr {THR})",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)

    # --- recon plot
    ax = axes[1]
    ax.plot(lams, [r["recon_audio_stage1"] for r in records_main], "-o", color="#1f77b4", label="Audio · stage1")
    ax.plot(lams, [r["recon_audio_stage2"] for r in records_main], "--o", color="#1f77b4", label="Audio · stage2")
    ax.plot(lams, [r["recon_visual_stage1"] for r in records_main], "-s", color="#2ca02c", label="Visual · stage1")
    ax.plot(lams, [r["recon_visual_stage2"] for r in records_main], "--s", color="#2ca02c", label="Visual · stage2")
    ax.set_xscale("log")
    ax.set_xlabel("λ_base")
    ax.set_ylabel("recon cos(ẑ, z) — SpLiCE Eq.(2)")
    ax.set_title("recon vs λ   (Step-4 ẑ = σ(σ(C̃·w) + μ_z), uncentered)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)

    # --- L0 plot
    ax = axes[2]
    ax.plot(lams, [r["l0_audio_stage1"] for r in records_main], "-o", color="#1f77b4", label="Audio · stage1")
    ax.plot(lams, [r["l0_audio_stage2"] for r in records_main], "--o", color="#1f77b4", label="Audio · stage2")
    ax.plot(lams, [r["l0_visual_stage1"] for r in records_main], "-s", color="#2ca02c", label="Visual · stage1")
    ax.plot(lams, [r["l0_visual_stage2"] for r in records_main], "--s", color="#2ca02c", label="Visual · stage2")
    ax.set_xscale("log")
    ax.set_xlabel("λ_base")
    ax.set_ylabel("L0 (mean #nonzero per segment)")
    ax.set_title("L0 vs λ", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)

    zero_label = "zero-excluded score" if records_main[0].get("score_exclude_zero", True) else "zero-included score"
    title = (
        f"stage1/2 λ sweep · LLP test "
        f"({records_main[0]['mean_source_label']}, {zero_label}, "
        f"prior={records_main[0].get('stage2_prior_mode', 'full')})"
    )
    if av2a_baseline is not None:
        title += f"\nAV2A baseline: {Path(str(av2a_baseline['path'])).parent.name}"
    if zs_baseline is not None:
        title += (
            f"\nZS baseline: CLAP-audio={float(zs_baseline['zs_clap_audio']):.4f}, "
            f"CLIP-visual={float(zs_baseline['zs_clip_visual']):.4f}, "
            f"AV={float(zs_baseline['zs_av_and']):.4f}"
        )
    if sanity is not None:
        title += (f"\nSanity (η=0, λ={sanity['lambda']}): "
                  f"max|W₂−W₁| audio={sanity['max_abs_W_diff_audio']:.2e}, "
                  f"visual={sanity['max_abs_W_diff_visual']:.2e}  ·  "
                  f"F1 audio Δ={sanity['f1_audio_stage2']-sanity['f1_audio_stage1']:+.4f}, "
                  f"visual Δ={sanity['f1_visual_stage2']-sanity['f1_visual_stage1']:+.4f}")
    fig.suptitle(title, fontsize=10.5, y=1.02)

    out_png = out_root / "sweep_lambda.png"
    out_pdf = out_root / "sweep_lambda.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[plot] saved → {out_png}")
    print(f"[plot] saved → {out_pdf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=ROOT / "results" / "sweep_lambda")
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--rho-min", type=float, default=0.1)
    ap.add_argument("--rho-max", type=float, default=1.0)
    ap.add_argument(
        "--stage2-prior-mode",
        choices=["full", "video"],
        default="full",
        help="'full': video-level prior * segment prior; 'video': video-level prior only",
    )
    ap.add_argument("--fista-iters", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--mean-source", choices=["llp", "external"], default="llp")
    ap.add_argument("--audio-mean-path", type=str, default=None)
    ap.add_argument("--visual-mean-path", type=str, default=None)
    ap.add_argument("--av2a-metrics-path", type=str, default=str(DEFAULT_AV2A_METRICS_PATH))
    ap.add_argument("--no-av2a-baseline", action="store_true")
    ap.add_argument("--no-zs-baseline", action="store_true")
    ap.add_argument(
        "--score-include-zero",
        action="store_true",
        help="include exact-zero Lasso weights in class-axis z-score stats; default excludes zeros",
    )
    ap.add_argument("--sanity-lambda", type=float, default=0.05,
                    help="run an extra (η=0, λ=this) sanity to confirm stage1≈stage2")
    ap.add_argument("--skip-run", action="store_true",
                    help="skip subprocess execution, only re-evaluate from existing dirs")
    ap.add_argument("--dry-run", action="store_true",
                    help="print commands without executing or evaluating")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    write_mean_meta(args.out_root, args.mean_source, args.audio_mean_path, args.visual_mean_path)
    score_exclude_zero = not args.score_include_zero

    # ---------- 1. run all configs ----------
    main_dirs = []
    for lam in args.lambdas:
        d = args.out_root / lam_tag(lam)
        main_dirs.append((lam, d))
        if not args.skip_run:
            run_one(d, lam, args.kappa, args.eta, args.rho_min, args.rho_max,
                    args.fista_iters, args.device, args.mean_source,
                    args.audio_mean_path, args.visual_mean_path,
                    args.limit_videos, args.score_include_zero, args.stage2_prior_mode, args.dry_run)

    sanity_dir = args.out_root / f"sanity_eta0_{lam_tag(args.sanity_lambda)}"
    if not args.skip_run:
        run_one(sanity_dir, args.sanity_lambda, args.kappa, 0.0, args.rho_min, args.rho_max,
                args.fista_iters, args.device, args.mean_source,
                args.audio_mean_path, args.visual_mean_path,
                args.limit_videos, args.score_include_zero, args.stage2_prior_mode, args.dry_run)

    if args.dry_run:
        print("[dry-run] done; no eval / no plot.")
        return

    # ---------- 2. evaluate ----------
    df = load_llp_metadata()
    filenames = df.filename.tolist()
    label = mean_source_label(args.mean_source, args.audio_mean_path, args.visual_mean_path)

    records = []
    for lam, d in main_dirs:
        ev = evaluate_run(d, filenames, score_exclude_zero=score_exclude_zero)
        ev.update({
            "lambda": lam,
            "kappa": args.kappa,
            "eta": args.eta,
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "stage2_prior_mode": args.stage2_prior_mode,
            "is_main": True,
            "out_dir": str(d),
            "mean_source": args.mean_source,
            "mean_source_label": label,
        })
        records.append(ev)
        print(f"[eval] λ={lam}  F1 a1={ev['f1_audio_stage1']:.4f} a2={ev['f1_audio_stage2']:.4f}  "
              f"v1={ev['f1_visual_stage1']:.4f} v2={ev['f1_visual_stage2']:.4f}  "
              f"recon_a1={ev['recon_audio_stage1']:.3f} recon_v1={ev['recon_visual_stage1']:.3f}  "
              f"L0_a1={ev['l0_audio_stage1']:.2f} L0_v1={ev['l0_visual_stage1']:.2f}")

    sanity_ev = None
    if (sanity_dir / "meta.json").exists():
        sanity_ev = evaluate_run(sanity_dir, filenames, score_exclude_zero=score_exclude_zero)
        sanity_ev.update({"lambda": args.sanity_lambda, "kappa": args.kappa, "eta": 0.0,
                           "rho_min": args.rho_min, "rho_max": args.rho_max,
                           "stage2_prior_mode": args.stage2_prior_mode,
                           "is_main": False, "out_dir": str(sanity_dir),
                           "mean_source": args.mean_source,
                           "mean_source_label": label})
        records.append(sanity_ev)
        print(f"[sanity η=0, λ={args.sanity_lambda}]  "
              f"F1 a1={sanity_ev['f1_audio_stage1']:.4f} a2={sanity_ev['f1_audio_stage2']:.4f}  "
              f"v1={sanity_ev['f1_visual_stage1']:.4f} v2={sanity_ev['f1_visual_stage2']:.4f}  "
              f"max|W₂−W₁| a={sanity_ev['max_abs_W_diff_audio']:.2e} v={sanity_ev['max_abs_W_diff_visual']:.2e}")
    elif args.skip_run:
        print(f"[warn] sanity run missing under {sanity_dir}; plotting without sanity annotation")
    else:
        raise FileNotFoundError(f"sanity run missing under {sanity_dir}")

    (args.out_root / "sweep_results.json").write_text(json.dumps(records, indent=2))
    print(f"[json] saved → {args.out_root / 'sweep_results.json'}")

    # ---------- 3. plot ----------
    av2a_baseline = None if args.no_av2a_baseline else load_av2a_baseline(args.av2a_metrics_path)
    if av2a_baseline is not None:
        (args.out_root / "av2a_baseline.json").write_text(json.dumps(av2a_baseline, indent=2))
        def _fmt_metric(key: str) -> str:
            value = av2a_baseline.get(key)
            return "NA" if value is None else f"{float(value):.4f}"
        print(
            "[av2a] "
            f"audio={_fmt_metric('audio')} "
            f"visual={_fmt_metric('visual')} "
            f"av={_fmt_metric('av')} "
            f"from {av2a_baseline.get('path')}"
        )
    zs_baseline = None
    if not args.no_zs_baseline:
        kept_filenames = json.loads((Path(records[0]["out_dir"]) / "meta.json").read_text())["filenames"]
        zs_baseline = compute_zero_shot_baseline(
            filenames=kept_filenames,
            threshold=THR,
        )
        (args.out_root / "zs_baseline.json").write_text(json.dumps(zs_baseline, indent=2))
        print(
            "[zs] "
            f"CLAP-audio={float(zs_baseline['zs_clap_audio']):.4f} "
            f"CLIP-visual={float(zs_baseline['zs_clip_visual']):.4f} "
            f"AV={float(zs_baseline['zs_av_and']):.4f} "
            f"(N={zs_baseline['num_videos']}, thr={zs_baseline['threshold']})"
        )
    make_plots(records, args.out_root, sanity_ev, av2a_baseline, zs_baseline)


if __name__ == "__main__":
    main()
