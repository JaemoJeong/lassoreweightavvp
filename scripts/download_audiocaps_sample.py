"""
Download N random WAV samples from inogii/audiocaps (AudioCaps/AudioSet WAVs)
to /mnt/hdd4tb/jaemo/AudioCaps/audio/ via huggingface_hub.

Output:
  - /mnt/hdd4tb/jaemo/AudioCaps/audio/<basename>.wav    (downloaded WAVs)
  - /mnt/hdd4tb/jaemo/AudioCaps/audiocaps_manifest.csv  (audio_path,youtube_id)
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="inogii/audiocaps")
    ap.add_argument("--repo-type", default="dataset")
    ap.add_argument("--target-n", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("/mnt/hdd4tb/jaemo/AudioCaps/audio"))
    ap.add_argument("--manifest", type=Path, default=Path("/mnt/hdd4tb/jaemo/AudioCaps/audiocaps_manifest.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    print(f"[list] {args.repo_id}")
    files = api.list_repo_files(args.repo_id, repo_type=args.repo_type)
    wavs = sorted([f for f in files if f.endswith(".wav")])
    print(f"  total wavs: {len(wavs)}")

    random.seed(args.seed)
    random.shuffle(wavs)
    candidates = wavs[: args.target_n * 2]  # 2x oversample buffer
    print(f"  sampled candidates: {len(candidates)} (target {args.target_n})")

    successes = []
    failures = []
    t0 = time.time()

    def fetch(remote: str) -> tuple[str, str | None, str | None]:
        basename = Path(remote).name
        local = args.out_dir / basename
        if local.exists() and local.stat().st_size > 1000:
            return remote, str(local), None
        try:
            p = hf_hub_download(
                repo_id=args.repo_id, filename=remote, repo_type=args.repo_type,
                local_dir=None,  # use HF cache then we copy
            )
            # symlink/copy into out_dir for clean manifest
            if not local.exists():
                local.write_bytes(Path(p).read_bytes())
            return remote, str(local), None
        except Exception as exc:
            return remote, None, str(exc)[:200]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, f): f for f in candidates}
        for fut in as_completed(futs):
            remote = futs[fut]
            r, path, err = fut.result()
            if path:
                yid = Path(remote).stem.lstrip("Y")
                successes.append({"audio_path": path, "youtube_id": yid, "remote": remote})
            else:
                failures.append({"remote": remote, "error": err})
            n_total = len(successes) + len(failures)
            if n_total % 100 == 0:
                el = time.time() - t0
                rate = n_total / max(el, 1e-6)
                eta = (args.target_n - len(successes)) / max(rate * len(successes) / max(n_total, 1), 1e-6) / 60
                print(f"  [{n_total}] succ={len(successes)} fail={len(failures)} "
                      f"rate={rate:.1f}/s ETA={eta:.1f}m", flush=True)
            if len(successes) >= args.target_n:
                for f2 in futs:
                    if not f2.done():
                        f2.cancel()
                break

    pd.DataFrame(successes).to_csv(args.manifest, index=False)
    print(f"\n[done] {len(successes)} success / {len(failures)} fail in {(time.time()-t0)/60:.1f}m")
    print(f"[saved] {args.manifest}")
    if failures:
        fp = args.manifest.with_name(args.manifest.stem + "_failures.csv")
        pd.DataFrame(failures).to_csv(fp, index=False)
        print(f"[saved] {fp}")


if __name__ == "__main__":
    main()
