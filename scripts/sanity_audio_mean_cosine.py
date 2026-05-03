"""
Sanity check: pairwise cosine between LLP test audio mean and external means
(VGGSound, FSD50K, DCASE2017).

Per handoff doc:
  1차 기준: cos(LLP audio mean, VGGSound mean) > 0.391  (FSD50K's 기준)
  2차 기준: unified sweep audio v25 best F1 > 38.13
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from avvp_stage12.data import load_llp_cached_bundle


def l2(x):
    return x / max(np.linalg.norm(x), 1e-8)


def cos(a, b):
    return float(l2(a) @ l2(b))


def main() -> None:
    print("[loading] LLP test audio segments...")
    bundle = load_llp_cached_bundle(backbone="ClipClap")
    audio_seg = bundle["audio_segments"]                       # (n, 10, 512)
    print(f"  {audio_seg.shape}")

    # LLP mean: per-row L2-norm → average
    flat = audio_seg.reshape(-1, audio_seg.shape[-1]).astype(np.float32)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    llp_n = flat / np.clip(norms, 1e-8, None)
    llp_mean = llp_n.mean(0).astype(np.float32)
    print(f"  LLP audio mean ||μ||={np.linalg.norm(llp_mean):.4f}  N={flat.shape[0]} segments")

    means_dir = Path("/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means")
    candidates = sorted(means_dir.glob("clap_HTSAT-tiny_audio_*.npy"))
    print(f"\n[external means] found {len(candidates)} files")

    means: dict[str, np.ndarray] = {"LLP_test (proxy)": llp_mean}
    for p in candidates:
        m = np.load(p).astype(np.float32)
        means[p.stem] = m
        print(f"  {p.name:<60} ||μ||={np.linalg.norm(m):.4f}")

    print(f"\n=== pairwise cosine ===")
    keys = list(means.keys())
    print(f"{'':<48}" + "".join(f"{k[:18]:>22}" for k in keys))
    for ki in keys:
        row = f"{ki[:46]:<48}"
        for kj in keys:
            row += f"{cos(means[ki], means[kj]):>22.4f}"
        print(row)

    # Highlight LLP comparisons (handoff judgment criteria)
    print(f"\n=== handoff 1차 기준: LLP ↔ external ===")
    for k, m in means.items():
        if k.startswith("LLP_test"): continue
        c = cos(llp_mean, m)
        verdict = ""
        if "vggsound" in k.lower():
            verdict = "  ✓ PASS (>0.391)" if c > 0.391 else "  ✗ FAIL (<=0.391)"
        elif "fsd50k" in k.lower():
            verdict = "  (reference: 0.391 baseline)"
        print(f"  {k:<60}  cos(LLP, ·) = {c:>+.4f}{verdict}")


if __name__ == "__main__":
    main()
