from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from avvp_stage12.constants import DEFAULT_BACKBONE, DEFAULT_VOCAB
from avvp_stage12.data import load_llp_cached_bundle, load_prompt_vocab
from avvp_stage12.pipeline import Stage12Config, prepare_modality, run_stage12


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
    parser.add_argument("--lambda-video-a", type=float, default=None)
    parser.add_argument("--lambda-video-v", type=float, default=None)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--lambda-min-factor", type=float, default=0.1)
    parser.add_argument("--fista-iters", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-videos", type=int, default=0)
    return parser.parse_args()


def resolve_lambda(override: float | None, base: float) -> float:
    return float(base if override is None else override)


def maybe_limit(arr: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0:
        return arr
    return arr[:limit]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_llp_cached_bundle(backbone=args.backbone)
    vocab = load_prompt_vocab(args.vocab)

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
    )
    visual = prepare_modality(
        "visual",
        segment_raw=visual_segments,
        video_raw=visual_video,
        proto_raw=vocab["visual_rows"],
    )

    cfg = Stage12Config(
        lambda_a=resolve_lambda(args.lambda_a, args.lambda_base),
        lambda_v=resolve_lambda(args.lambda_v, args.lambda_base),
        lambda_video_a=resolve_lambda(args.lambda_video_a, args.lambda_base),
        lambda_video_v=resolve_lambda(args.lambda_video_v, args.lambda_base),
        beta=float(args.beta),
        gamma=float(args.gamma),
        lambda_min_factor=float(args.lambda_min_factor),
        fista_iters=int(args.fista_iters),
        device=args.device,
    )

    results = run_stage12(audio, visual, cfg)

    np.save(args.out_dir / "W_a_stage1.npy", results["audio"]["stage1"]["weights"])
    np.save(args.out_dir / "W_v_stage1.npy", results["visual"]["stage1"]["weights"])
    np.save(args.out_dir / "recon_a_stage1.npy", results["audio"]["stage1"]["recon"])
    np.save(args.out_dir / "recon_v_stage1.npy", results["visual"]["stage1"]["recon"])
    np.save(args.out_dir / "P_a.npy", results["audio"]["presence"])
    np.save(args.out_dir / "P_v.npy", results["visual"]["presence"])
    np.save(args.out_dir / "W_a_video.npy", results["audio"]["video"]["weights"])
    np.save(args.out_dir / "W_v_video.npy", results["visual"]["video"]["weights"])
    np.save(args.out_dir / "A_a.npy", results["audio"]["absence"])
    np.save(args.out_dir / "A_v.npy", results["visual"]["absence"])
    np.save(args.out_dir / "evidence_v_to_a.npy", results["audio"]["evidence_from_visual"])
    np.save(args.out_dir / "evidence_a_to_v.npy", results["visual"]["evidence_from_audio"])
    np.save(args.out_dir / "lambda_a_weighted.npy", results["audio"]["weighted_lambda"])
    np.save(args.out_dir / "lambda_v_weighted.npy", results["visual"]["weighted_lambda"])
    np.save(args.out_dir / "W_a_stage2.npy", results["audio"]["stage2"]["weights"])
    np.save(args.out_dir / "W_v_stage2.npy", results["visual"]["stage2"]["weights"])
    np.save(args.out_dir / "recon_a_stage2.npy", results["audio"]["stage2"]["recon"])
    np.save(args.out_dir / "recon_v_stage2.npy", results["visual"]["stage2"]["recon"])

    meta = {
        "backbone": args.backbone,
        "vocab": args.vocab,
        "num_videos": len(filenames),
        "num_segments": int(audio.num_segments),
        "audio_shape": list(audio_segments.shape),
        "visual_shape": list(visual_segments.shape),
        "audio_video_shape": list(audio_video.shape),
        "visual_video_shape": list(visual_video.shape),
        "filenames": filenames,
        "video_ids": video_ids,
        "config": results["config"],
        "audio_summary_stage1": results["audio"]["summary_stage1"],
        "audio_summary_stage2": results["audio"]["summary_stage2"],
        "audio_summary_video": results["audio"]["summary_video"],
        "visual_summary_stage1": results["visual"]["summary_stage1"],
        "visual_summary_stage2": results["visual"]["summary_stage2"],
        "visual_summary_video": results["visual"]["summary_video"],
        "notes": {
            "presence": "P[c] = max_t w_stage1[t, c]",
            "absence": "A[c] = max(0, max_k w_video[k] - w_video[c])",
            "stage2_scope": "confidence omitted on purpose; only presence + video sparse absence used",
        },
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
