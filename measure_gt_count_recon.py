"""
Measure GT-count-bucket Step-4 reconstruction means for paper §3.2.4 update.

Loads existing W npy from sweep results (lam=0.05, β=γ=0.5 default), rebuilds GT,
recomputes ẑ = σ(σ(C̃·w) + μ_z), and reports cos(ẑ, z_n) bucketed by per-segment
GT label count for both stage1 (independent) and stage2 (cross-modal hint) for
both modalities.

Does NOT modify any code under avvp_stage12/. Pure read-only analysis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from avvp_stage12.data import build_dense_gt, load_llp_cached_bundle, load_prompt_vocab
from avvp_stage12.pipeline import prepare_modality
from avvp_stage12.solver import step4_reconstruction_cosine
from avvp_stage12.constants import LLP_CATS

# default lam from sweep + paper context
DEFAULT_LAM = 0.05
SWEEP_DIR = ROOT / "results/sweep_lambda/lam0p05"


def bucket_recon(recon_flat: np.ndarray, gt_count_flat: np.ndarray) -> dict[int, dict[str, float]]:
    """Group recon values by per-segment GT count and return mean/std/n per bucket."""
    out = {}
    for k in sorted(set(gt_count_flat.tolist())):
        m = gt_count_flat == k
        if m.sum() == 0:
            continue
        v = recon_flat[m]
        out[int(k)] = {"n": int(m.sum()), "mean": float(v.mean()),
                        "std": float(v.std()), "median": float(np.median(v))}
    return out


def report(name: str, buckets: dict[int, dict[str, float]]) -> str:
    lines = [f"=== {name} ==="]
    lines.append(f"{'GT count':>10}{'n':>10}{'mean':>10}{'std':>9}{'median':>10}")
    for k, st in sorted(buckets.items()):
        lines.append(f"{k:>10}{st['n']:>10}{st['mean']:>10.4f}{st['std']:>9.4f}{st['median']:>10.4f}")
    return "\n".join(lines)


def main():
    print(f"Loading bundle, vocab, GT...")
    bundle = load_llp_cached_bundle(backbone="ClipClap")
    vocab = load_prompt_vocab("v25")
    filenames = bundle["filenames"]
    n_videos = len(filenames)
    print(f"  videos: {n_videos}")

    audio = prepare_modality("audio", bundle["audio_segments"], bundle["audio_video"], vocab["audio_rows"])
    visual = prepare_modality("visual", bundle["visual_segments"], bundle["visual_video"], vocab["visual_rows"])

    GT_A = build_dense_gt(filenames, "audio")     # (n, 10, 25)
    GT_V = build_dense_gt(filenames, "visual")
    cnt_A = GT_A.sum(-1).reshape(-1)              # (n*10,)
    cnt_V = GT_V.sum(-1).reshape(-1)

    Wa1 = np.load(SWEEP_DIR / "W_a_stage1.npy")   # (n, 10, 25)
    Wv1 = np.load(SWEEP_DIR / "W_v_stage1.npy")
    Wa2 = np.load(SWEEP_DIR / "W_a_stage2.npy")
    Wv2 = np.load(SWEEP_DIR / "W_v_stage2.npy")

    # recompute Step-4 recon for each (W, modality)
    z_n_a_flat = audio.segment_n.reshape(-1, audio.d_dim)
    z_n_v_flat = visual.segment_n.reshape(-1, visual.d_dim)

    def step4_recon(W3d, z_n_flat, c_center, z_mean):
        W_flat = W3d.reshape(-1, W3d.shape[-1])
        return step4_reconstruction_cosine(W_flat, z_n_flat, c_center, z_mean)

    r_a1 = step4_recon(Wa1, z_n_a_flat, audio.proto_center, audio.segment_mean)
    r_a2 = step4_recon(Wa2, z_n_a_flat, audio.proto_center, audio.segment_mean)
    r_v1 = step4_recon(Wv1, z_n_v_flat, visual.proto_center, visual.segment_mean)
    r_v2 = step4_recon(Wv2, z_n_v_flat, visual.proto_center, visual.segment_mean)

    print(f"\nλ={DEFAULT_LAM}, β=γ=0.5  (stage2 = cross-modal weighted Lasso)\n")
    bucket_a1 = bucket_recon(r_a1, cnt_A)
    bucket_a2 = bucket_recon(r_a2, cnt_A)
    bucket_v1 = bucket_recon(r_v1, cnt_V)
    bucket_v2 = bucket_recon(r_v2, cnt_V)

    print(report("AUDIO  stage1  (recon vs GT_A count)", bucket_a1))
    print(report("AUDIO  stage2  (recon vs GT_A count)", bucket_a2))
    print(report("VISUAL stage1  (recon vs GT_V count)", bucket_v1))
    print(report("VISUAL stage2  (recon vs GT_V count)", bucket_v2))

    out = {
        "lam": DEFAULT_LAM,
        "n_videos": n_videos,
        "n_segments": int(cnt_A.size),
        "audio_stage1": bucket_a1,
        "audio_stage2": bucket_a2,
        "visual_stage1": bucket_v1,
        "visual_stage2": bucket_v2,
    }
    out_path = ROOT / "results/sweep_lambda/gt_count_recon.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    main()
