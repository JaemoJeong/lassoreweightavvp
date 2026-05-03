from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from avvp_stage12.constants import DEFAULT_BACKBONE, DEFAULT_MEAN_SOURCE, DEFAULT_VOCAB
from avvp_stage12.data import build_dense_gt, load_llp_cached_bundle, load_prompt_vocab, load_reference_means
from avvp_stage12.pipeline import Stage12Config, prepare_modality, run_stage12
from avvp_stage12.reporting import compute_stage_predictions
from avvp_stage12.reporting import write_stage_eval_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean LLP-25 stage1/stage2 AVVP implementation."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB)
    parser.add_argument("--lambda-base", type=float, default=0.05)
    parser.add_argument("--lambda-a", type=float, default=None)
    parser.add_argument("--lambda-v", type=float, default=None)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rho-min", type=float, default=None)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument(
        "--lambda-min-factor",
        type=float,
        default=None,
        help="legacy alias for --rho-min",
    )
    parser.add_argument("--fista-iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--max-stage", type=int, choices=[1, 2], default=2)
    parser.add_argument(
        "--stage2-prior-mode",
        choices=["full", "video"],
        default="full",
        help="'full': video-level prior * segment prior; 'video': video-level prior only",
    )
    parser.add_argument(
        "--video-prior-only",
        action="store_true",
        help="shortcut for --stage2-prior-mode video",
    )
    parser.add_argument("--mean-source", choices=["llp", "external"], default=DEFAULT_MEAN_SOURCE)
    parser.add_argument("--audio-mean-path", type=Path, default=None)
    parser.add_argument("--visual-mean-path", type=Path, default=None)
    parser.add_argument("--score-thr", type=float, default=0.75)
    parser.add_argument(
        "--score-mode",
        choices=["adaptive_k", "fixed_t"],
        default="adaptive_k",
        help="v5.2 default is adaptive_k: active z-score + T=clip(K/K0,Tmin,Tmax)",
    )
    parser.add_argument("--score-k0", type=float, default=16.0)
    parser.add_argument("--score-t-min", type=float, default=0.25)
    parser.add_argument("--score-t-max", type=float, default=1.25)
    parser.add_argument(
        "--score-include-zero",
        action="store_true",
        help="include exact-zero Lasso weights in class-axis z-score stats; default excludes zeros",
    )
    parser.add_argument("--no-details", action="store_true")
    parser.add_argument("--detail-max-videos", type=int, default=0)
    parser.add_argument("--detail-top-k", type=int, default=5)
    parser.add_argument("--detail-all-classes", action="store_true")
    return parser.parse_args()


def resolve_lambda(override: float | None, base: float) -> float:
    return float(base if override is None else override)


def maybe_limit(arr: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        return arr
    return arr[:limit]


def resolve_rho_min(rho_min: float | None, legacy_min_factor: float | None) -> float:
    if rho_min is not None:
        return float(rho_min)
    if legacy_min_factor is not None:
        return float(legacy_min_factor)
    return 0.1


def broadcast_video_prior(video_prior: np.ndarray, num_segments: int) -> np.ndarray:
    return np.broadcast_to(video_prior[:, None, :], (video_prior.shape[0], num_segments, video_prior.shape[1])).astype(np.float32)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_llp_cached_bundle(backbone=args.backbone)
    vocab = load_prompt_vocab(args.vocab)
    means = load_reference_means(
        mean_source=args.mean_source,
        audio_mean_path=args.audio_mean_path,
        visual_mean_path=args.visual_mean_path,
    )

    limit = args.limit_videos if args.limit_videos > 0 else len(bundle["filenames"])
    filenames = bundle["filenames"][:limit]
    video_ids = bundle["video_ids"][:limit]
    audio_segments = maybe_limit(bundle["audio_segments"], args.limit_videos)
    visual_segments = maybe_limit(bundle["visual_segments"], args.limit_videos)
    audio_video = maybe_limit(bundle["audio_video"], args.limit_videos)
    visual_video = maybe_limit(bundle["visual_video"], args.limit_videos)

    audio = prepare_modality(
        "audio",
        segment_raw=audio_segments,
        video_raw=audio_video,
        proto_raw=vocab["audio_rows"],
        segment_mean_override=None if means is None else means["audio"],
        video_mean_override=None if means is None else means["audio"],
    )
    visual = prepare_modality(
        "visual",
        segment_raw=visual_segments,
        video_raw=visual_video,
        proto_raw=vocab["visual_rows"],
        segment_mean_override=None if means is None else means["visual"],
        video_mean_override=None if means is None else means["visual"],
    )

    cfg = Stage12Config(
        lambda_a=resolve_lambda(args.lambda_a, args.lambda_base),
        lambda_v=resolve_lambda(args.lambda_v, args.lambda_base),
        kappa=float(args.kappa),
        eta=float(args.eta),
        rho_min=resolve_rho_min(args.rho_min, args.lambda_min_factor),
        rho_max=float(args.rho_max),
        fista_iters=int(args.fista_iters),
        device=args.device,
        max_stage=int(args.max_stage),
        prior_mode="video" if args.video_prior_only else args.stage2_prior_mode,
    )

    results = run_stage12(audio, visual, cfg)

    audio_results = results["audio"]
    visual_results = results["visual"]

    np.save(args.out_dir / "W_a_stage1.npy", audio_results["stage1"]["weights"])
    np.save(args.out_dir / "W_v_stage1.npy", visual_results["stage1"]["weights"])
    np.save(args.out_dir / "recon_a_stage1.npy", audio_results["stage1"]["recon"])
    np.save(args.out_dir / "recon_v_stage1.npy", visual_results["stage1"]["recon"])
    np.save(args.out_dir / "recon_center_a_stage1.npy", audio_results["stage1"]["recon_center"])
    np.save(args.out_dir / "recon_center_v_stage1.npy", visual_results["stage1"]["recon_center"])
    np.save(args.out_dir / "P_a.npy", audio_results["presence"])
    np.save(args.out_dir / "P_v.npy", visual_results["presence"])

    if "stage2" in audio_results:
        np.save(args.out_dir / "sparse_confidence_a.npy", audio_results["sparse_confidence"])
        np.save(args.out_dir / "sparse_confidence_v.npy", visual_results["sparse_confidence"])
        np.save(args.out_dir / "reconstruction_quality_a.npy", audio_results["reconstruction_quality"])
        np.save(args.out_dir / "reconstruction_quality_v.npy", visual_results["reconstruction_quality"])
        np.save(args.out_dir / "reliable_confidence_a.npy", audio_results["reliable_confidence"])
        np.save(args.out_dir / "reliable_confidence_v.npy", visual_results["reliable_confidence"])
        np.save(args.out_dir / "video_prior_a.npy", audio_results["video_prior"])
        np.save(args.out_dir / "video_prior_v.npy", visual_results["video_prior"])
        # Backward-compatible aliases used by earlier notes/scripts.
        np.save(args.out_dir / "reliability_a.npy", audio_results["reliability"])
        np.save(args.out_dir / "reliability_v.npy", visual_results["reliability"])
        np.save(args.out_dir / "local_support_a.npy", audio_results["local_support"])
        np.save(args.out_dir / "local_support_v.npy", visual_results["local_support"])
        np.save(args.out_dir / "plausibility_a.npy", audio_results["plausibility"])
        np.save(args.out_dir / "plausibility_v.npy", visual_results["plausibility"])
        np.save(args.out_dir / "segment_prior_v_to_a.npy", visual_results["local_support"])
        np.save(args.out_dir / "segment_prior_a_to_v.npy", audio_results["local_support"])
        np.save(args.out_dir / "video_prior_v_to_a.npy", broadcast_video_prior(visual_results["plausibility"], audio.num_segments))
        np.save(args.out_dir / "video_prior_a_to_v.npy", broadcast_video_prior(audio_results["plausibility"], visual.num_segments))
        np.save(args.out_dir / "H_v_to_a.npy", audio_results["prior_from_visual"])
        np.save(args.out_dir / "H_a_to_v.npy", visual_results["prior_from_audio"])
        np.save(args.out_dir / "penalty_scale_a.npy", audio_results["penalty_scale"])
        np.save(args.out_dir / "penalty_scale_v.npy", visual_results["penalty_scale"])
        # Backward-compatible alias: evidence now means the v5 cross-modal prior H(t,c).
        np.save(args.out_dir / "evidence_v_to_a.npy", audio_results["evidence_from_visual"])
        np.save(args.out_dir / "evidence_a_to_v.npy", visual_results["evidence_from_audio"])
        np.save(args.out_dir / "lambda_a_weighted.npy", audio_results["weighted_lambda"])
        np.save(args.out_dir / "lambda_v_weighted.npy", visual_results["weighted_lambda"])
        np.save(args.out_dir / "W_a_stage2.npy", audio_results["stage2"]["weights"])
        np.save(args.out_dir / "W_v_stage2.npy", visual_results["stage2"]["weights"])
        np.save(args.out_dir / "recon_a_stage2.npy", audio_results["stage2"]["recon"])
        np.save(args.out_dir / "recon_v_stage2.npy", visual_results["stage2"]["recon"])

    gt_a = build_dense_gt(filenames, "audio")
    gt_v = build_dense_gt(filenames, "visual")
    score_exclude_zero = not args.score_include_zero
    metrics = {
        "threshold": float(args.score_thr),
        "score_normalization": (
            "adaptive_k: active z-score over nonzero coefficients, "
            "T=clip(K/K0,Tmin,Tmax), sigmoid(z/T); fixed_t: same active z-score, T=1"
        ),
        "score_mode": args.score_mode,
        "score_exclude_zero": bool(score_exclude_zero),
        "score_k0": float(args.score_k0),
        "score_t_min": float(args.score_t_min),
        "score_t_max": float(args.score_t_max),
        "pred_av_rule": "Pred_AV = Pred_A AND Pred_V",
        "stages": {},
    }
    metrics["stages"]["stage1"] = write_stage_eval_outputs(
        out_dir=args.out_dir,
        stage_name="stage1",
        filenames=filenames,
        video_ids=video_ids,
        W_a=audio_results["stage1"]["weights"],
        W_v=visual_results["stage1"]["weights"],
        gt_a=gt_a,
        gt_v=gt_v,
        threshold=float(args.score_thr),
        score_exclude_zero=score_exclude_zero,
        write_details=not args.no_details,
        detail_max_videos=int(args.detail_max_videos),
        detail_top_k=int(args.detail_top_k),
        detail_all_classes=bool(args.detail_all_classes),
        score_mode=args.score_mode,
        score_k0=float(args.score_k0),
        score_t_min=float(args.score_t_min),
        score_t_max=float(args.score_t_max),
    )
    if "stage2" in audio_results:
        stage1_pred = compute_stage_predictions(
            audio_results["stage1"]["weights"],
            visual_results["stage1"]["weights"],
            threshold=float(args.score_thr),
            score_exclude_zero=score_exclude_zero,
            score_mode=args.score_mode,
            score_k0=float(args.score_k0),
            score_t_min=float(args.score_t_min),
            score_t_max=float(args.score_t_max),
        )
        stage1_scores_a = stage1_pred["scores_a"].astype(np.float32)
        stage1_scores_v = stage1_pred["scores_v"].astype(np.float32)
        video_prior_v_to_a = broadcast_video_prior(visual_results["plausibility"], audio.num_segments)
        video_prior_a_to_v = broadcast_video_prior(audio_results["plausibility"], visual.num_segments)
        metrics["stages"]["stage2"] = write_stage_eval_outputs(
            out_dir=args.out_dir,
            stage_name="stage2",
            filenames=filenames,
            video_ids=video_ids,
            W_a=audio_results["stage2"]["weights"],
            W_v=visual_results["stage2"]["weights"],
            gt_a=gt_a,
            gt_v=gt_v,
            threshold=float(args.score_thr),
            score_exclude_zero=score_exclude_zero,
            write_details=not args.no_details,
            detail_max_videos=int(args.detail_max_videos),
            detail_top_k=int(args.detail_top_k),
            detail_all_classes=bool(args.detail_all_classes),
            score_mode=args.score_mode,
            score_k0=float(args.score_k0),
            score_t_min=float(args.score_t_min),
            score_t_max=float(args.score_t_max),
            detail_extra_columns=[
                ("A_S1Score", stage1_scores_a),
                ("V_S1Score", stage1_scores_v),
                ("A_VidPrior", video_prior_v_to_a),
                ("A_SegPrior", visual_results["local_support"]),
                ("A_H", audio_results["prior_from_visual"]),
                ("V_VidPrior", video_prior_a_to_v),
                ("V_SegPrior", audio_results["local_support"]),
                ("V_H", visual_results["prior_from_audio"]),
            ],
        )
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    meta = {
        "backbone": args.backbone,
        "vocab": args.vocab,
        "max_stage": int(args.max_stage),
        "num_videos": len(filenames),
        "num_segments": int(audio.num_segments),
        "audio_shape": list(audio_segments.shape),
        "visual_shape": list(visual_segments.shape),
        "audio_video_shape": list(audio_video.shape),
        "visual_video_shape": list(visual_video.shape),
        "filenames": filenames,
        "video_ids": video_ids,
        "config": results["config"],
        "mean_source": args.mean_source,
        "mean_info": {
            "audio_path": None if means is None else means["audio_path"],
            "visual_path": None if means is None else means["visual_path"],
            "audio_dim": int(audio.segment_mean.shape[0]),
            "visual_dim": int(visual.segment_mean.shape[0]),
        },
        "audio_summary_stage1": audio_results["summary_stage1"],
        "audio_summary_stage2": audio_results.get("summary_stage2"),
        "audio_summary_prior": audio_results.get("summary_prior"),
        "visual_summary_stage1": visual_results["summary_stage1"],
        "visual_summary_stage2": visual_results.get("summary_stage2"),
        "visual_summary_prior": visual_results.get("summary_prior"),
        "metrics": {
            stage: {
                "audio_segment_f1": vals["audio_segment_f1"],
                "visual_segment_f1": vals["visual_segment_f1"],
                "av_segment_f1_and": vals["av_segment_f1_and"],
                "audio_pred_active_mean": vals["audio_pred_active_mean"],
                "visual_pred_active_mean": vals["visual_pred_active_mean"],
                "av_pred_active_mean": vals["av_pred_active_mean"],
            }
            for stage, vals in metrics["stages"].items()
        },
        "notes": {
            "presence": "P[c] = max_t w_stage1[t, c]",
            "sparse_confidence": "s_m(t,c)=w_m(t,c)/(max_j w_m(t,j)+eps)",
            "reconstruction_quality": "q_m(t)=max(0, cos(z_tilde_m(t), normalize(C_tilde_m w_m(t))))",
            "reliable_confidence": "g_m(t,c)=s_m(t,c)*q_m(t)",
            "segment_prior": "h_m(t,c)=g_m(t,c)",
            "video_prior": "pi_m(c)=max_t g_m(t,c)",
            "legacy_names": "reliability=q, local_support/reliable_confidence=g, plausibility/video_prior=pi",
            "cross_modal_prior": (
                "full: H_source_to_target(t,c)=pi_source(c)*(1+kappa*g_source(t,c)); "
                "video: H_source_to_target(t,c)=pi_source(c)"
            ),
            "stage2_prior_mode": results["config"]["prior_mode"],
            "weighted_lambda": "lambda_target^c(t)=lambda_base*clip(exp(-eta*H_source_to_target(t,c)), rho_min, rho_max)",
            "score": (
                "adaptive_k: active z-score over nonzero coefficients, "
                "T=clip(K/K0,Tmin,Tmax), sigmoid(z/T); fixed_t: active z-score with T=1"
            ),
            "score_mode": args.score_mode,
            "score_k0": float(args.score_k0),
            "score_t_min": float(args.score_t_min),
            "score_t_max": float(args.score_t_max),
            "score_exclude_zero": bool(score_exclude_zero),
            "prediction_threshold": float(args.score_thr),
            "pred_av": "Pred_A AND Pred_V",
            "stage2_scope": "AVVP_Paper_Draft_KR_v5 main formulation; no explicit absence term",
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
