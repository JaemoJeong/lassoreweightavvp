"""
Extract CLAP HTSAT-tiny audio mean directly from HF CLAPv2/vggsound_formatted_batch_1
(train split, raw audio in parquet — no intermediate wav files).

Usage:
  python extract_vggsound_audio_mean_streaming.py --target-n 10000

Pipeline per sample:
  1. Stream from HF (no full download).
  2. audio.array (float64, 48kHz mono, ~10s) → float32 numpy.
  3. CLAP HTSAT-tiny .get_audio_embedding_from_data → L2-normalize → accumulate.
  4. After N successful samples, save mean to
     /mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clap_HTSAT-tiny_audio_vggsoundtrain_N{N}.npy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="CLAPv2/vggsound_formatted_batch_1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--target-n", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[loading] HF streaming dataset: {args.dataset_id} (split={args.split})", flush=True)
    from datasets import load_dataset
    ds = load_dataset(args.dataset_id, split=args.split, streaming=True)

    print(f"[loading] laion_clap CLAP_Module(enable_fusion=False, amodel='HTSAT-tiny') ...", flush=True)
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", device=args.device)
    model.load_ckpt()
    model.eval()
    EMB_DIM = 512

    sum_vec = np.zeros(EMB_DIM, dtype=np.float64)
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    bs = args.batch_size

    batch_arrays: list[np.ndarray] = []
    target_sr = 48000

    def flush(batch: list[np.ndarray]) -> None:
        nonlocal sum_vec, n_done
        if not batch:
            return
        # CLAP expects (B, T) float32 tensor
        # Pad/truncate each to 10s = 480000 samples
        T = target_sr * 10
        padded = []
        for arr in batch:
            arr = arr[:T]
            if len(arr) < T:
                arr = np.pad(arr, (0, T - len(arr)))
            padded.append(arr)
        x = torch.tensor(np.stack(padded, 0), dtype=torch.float32, device=args.device)
        with torch.no_grad():
            emb = model.get_audio_embedding_from_data(x=x, use_tensor=True)
            emb = F.normalize(emb, dim=-1)
        sum_vec += emb.detach().cpu().numpy().sum(axis=0).astype(np.float64)
        n_done += len(batch)

    for i, sample in enumerate(ds):
        if n_done >= args.target_n:
            break
        try:
            audio = sample["audio"]
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio.get("sampling_rate", 48000))
            if sr != target_sr:
                # cheap resample via numpy interp (rare path; CLAP expects 48k)
                ratio = target_sr / sr
                new_len = int(round(len(arr) * ratio))
                if new_len <= 0:
                    n_skipped += 1
                    continue
                xs_old = np.linspace(0.0, 1.0, len(arr), endpoint=False)
                xs_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
                arr = np.interp(xs_new, xs_old, arr).astype(np.float32)
            if len(arr) < target_sr:  # < 1 second
                n_skipped += 1
                continue
            batch_arrays.append(arr)
        except Exception as exc:
            n_skipped += 1
            if n_skipped < 10 or n_skipped % 100 == 0:
                print(f"  [skip {n_skipped}] {exc}", flush=True)
            continue

        if len(batch_arrays) >= bs:
            try:
                flush(batch_arrays)
            except Exception as exc:
                n_skipped += len(batch_arrays)
                print(f"  [batch fail] {exc}", flush=True)
            batch_arrays = []
            if n_done > 0 and (n_done % 200) == 0:
                el = time.time() - t0
                rate = n_done / max(el, 1e-6)
                eta_min = (args.target_n - n_done) / max(rate, 1e-6) / 60
                print(f"  [{n_done}/{args.target_n}] rate={rate:.1f} clips/s "
                      f"skipped={n_skipped} ETA={eta_min:.1f}m", flush=True)

    # final flush
    flush(batch_arrays[: max(0, args.target_n - n_done)])

    if n_done == 0:
        raise SystemExit("no samples processed")

    mean_vec = (sum_vec / n_done).astype(np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"clap_HTSAT-tiny_audio_vggsoundtrain_N{n_done}.npy"
    np.save(out_path, mean_vec)
    print(f"\n[saved] {out_path}")
    print(f"  shape={mean_vec.shape}  norm={np.linalg.norm(mean_vec):.4f}  "
          f"N={n_done}  skipped={n_skipped}  total_time={(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
