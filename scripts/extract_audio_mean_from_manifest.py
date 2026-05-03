from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract CLAP HTSAT-tiny mean embedding from a manifest of audio files."
    )
    ap.add_argument(
        "--manifest-csv",
        type=Path,
        required=True,
        help="CSV containing at least one column of absolute audio paths.",
    )
    ap.add_argument(
        "--audio-col",
        default="audio_path",
        help="Column name containing file paths.",
    )
    ap.add_argument(
        "--dataset-tag",
        default="fsd50ktrain",
        help="Tag used in the output filename.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means"),
        help="Output directory for the .npy mean file.",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="Optional clip limit for smoke runs.")
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.manifest_csv)
    if args.audio_col not in df.columns:
        raise SystemExit(f"manifest missing audio column: {args.audio_col}")

    audio_paths = [str(Path(p)) for p in df[args.audio_col].tolist() if isinstance(p, str) and p]
    if args.limit > 0:
        audio_paths = audio_paths[: args.limit]
    if not audio_paths:
        raise SystemExit("no audio paths found in manifest")

    import laion_clap

    print("[loading] laion_clap CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny') ...", flush=True)
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", device=args.device)
    model.load_ckpt()
    model.eval()

    sum_vec = np.zeros(512, dtype=np.float64)
    n_done = 0
    t0 = time.time()
    bs = args.batch_size

    for start in range(0, len(audio_paths), bs):
        batch = audio_paths[start : start + bs]
        batch = [p for p in batch if Path(p).exists()]
        if not batch:
            continue
        try:
            emb = model.get_audio_embedding_from_filelist(batch, use_tensor=True)
            emb = F.normalize(emb, dim=-1)
        except Exception as exc:
            print(f"[skip batch {start}] {exc}", flush=True)
            continue

        sum_vec += emb.detach().cpu().numpy().sum(axis=0).astype(np.float64)
        n_done += len(batch)
        if (start // bs) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1e-6)
            eta = (len(audio_paths) - n_done) / max(rate, 1e-6)
            print(
                f"  [{n_done}/{len(audio_paths)}] rate={rate:.1f} clips/s ETA={eta/60:.1f}m",
                flush=True,
            )

    mean_vec = (sum_vec / max(n_done, 1)).astype(np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"clap_HTSAT-tiny_audio_{args.dataset_tag}_N{n_done}.npy"
    np.save(out_path, mean_vec)
    print(
        f"[saved] {out_path}  shape={mean_vec.shape}  norm={np.linalg.norm(mean_vec):.4f}  N={n_done}",
        flush=True,
    )


if __name__ == "__main__":
    main()
