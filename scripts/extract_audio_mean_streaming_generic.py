"""
Generic CLAP HTSAT-tiny audio mean extractor that streams from any HF dataset
with audio modality. Supports both legacy dict-based audio and new
torchcodec AudioDecoder API.

Output: /mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means/clap_HTSAT-tiny_audio_<tag>_N{N}.npy
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def get_audio_array_sr(audio_obj) -> tuple[np.ndarray, int]:
    """Return (array float32 mono, sample_rate). Handles both APIs."""
    if hasattr(audio_obj, "get"):  # legacy dict API
        arr = np.asarray(audio_obj.get("array", []), dtype=np.float32)
        sr = int(audio_obj.get("sampling_rate", 48000))
        return arr, sr
    if hasattr(audio_obj, "get_all_samples"):  # new torchcodec API
        samples = audio_obj.get_all_samples()
        data = samples.data  # (channels, T) tensor
        if data.dim() == 2 and data.shape[0] > 1:
            data = data.mean(0, keepdim=True)
        arr = data.squeeze(0).cpu().numpy().astype(np.float32)
        return arr, int(samples.sample_rate)
    raise RuntimeError(f"unknown audio object type: {type(audio_obj)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--audio-key", default="audio")
    ap.add_argument("--tag", required=True, help="output filename tag")
    ap.add_argument("--target-n", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/mnt/hdd4tb/jaemo/AVVP_vocab_sweep/means"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"[loading] {args.dataset_id} (config={args.config}, split={args.split})", flush=True)
    from datasets import load_dataset
    if args.config:
        ds = load_dataset(args.dataset_id, args.config, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.dataset_id, split=args.split, streaming=True)

    print("[loading] CLAP HTSAT-tiny ...", flush=True)
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-tiny", device=args.device)
    model.load_ckpt()
    model.eval()
    EMB_DIM = 512
    TARGET_SR = 48000
    T = TARGET_SR * 10  # 10s window

    sum_vec = np.zeros(EMB_DIM, dtype=np.float64)
    n_done = 0
    n_skipped = 0
    t0 = time.time()
    bs = args.batch_size

    def flush(batch: list[np.ndarray]) -> None:
        nonlocal sum_vec, n_done
        if not batch:
            return
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

    batch: list[np.ndarray] = []
    for sample in ds:
        if n_done >= args.target_n:
            break
        try:
            a = sample[args.audio_key]
            arr, sr = get_audio_array_sr(a)
            if sr != TARGET_SR and len(arr) > 0:
                ratio = TARGET_SR / sr
                new_len = int(round(len(arr) * ratio))
                if new_len <= 0:
                    n_skipped += 1; continue
                xs_old = np.linspace(0.0, 1.0, len(arr), endpoint=False)
                xs_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
                arr = np.interp(xs_new, xs_old, arr).astype(np.float32)
            if len(arr) < TARGET_SR:
                n_skipped += 1; continue
            batch.append(arr)
        except Exception as exc:
            n_skipped += 1
            if n_skipped < 5 or n_skipped % 100 == 0:
                print(f"  [skip {n_skipped}] {exc}", flush=True)
            continue

        if len(batch) >= bs:
            try:
                flush(batch)
            except Exception as exc:
                n_skipped += len(batch)
                print(f"  [batch fail] {exc}", flush=True)
            batch = []
            if n_done > 0 and (n_done % 200) == 0:
                el = time.time() - t0
                rate = n_done / max(el, 1e-6)
                eta_min = (args.target_n - n_done) / max(rate, 1e-6) / 60
                print(f"  [{n_done}/{args.target_n}] rate={rate:.1f} clips/s skipped={n_skipped} ETA={eta_min:.1f}m", flush=True)

    flush(batch[: max(0, args.target_n - n_done)])

    if n_done == 0:
        raise SystemExit("no samples processed")

    mean_vec = (sum_vec / n_done).astype(np.float32)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"clap_HTSAT-tiny_audio_{args.tag}_N{n_done}.npy"
    np.save(out_path, mean_vec)
    print(f"\n[saved] {out_path}")
    print(f"  shape={mean_vec.shape}  norm={np.linalg.norm(mean_vec):.4f}  N={n_done}  skipped={n_skipped}  total_time={(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
