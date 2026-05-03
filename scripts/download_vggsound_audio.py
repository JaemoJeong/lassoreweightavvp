"""
Download VGGSound audio clips (10s starting at `start_second`) via yt-dlp + ffmpeg.

Reads sample CSV (youtube_id, start, label, split), downloads each 10s slice
in parallel, and writes successful paths into a manifest CSV.

Outputs:
  /mnt/hdd4tb/jaemo/VGGSound/audio/train/<yid>_<start>.wav        # extracted slices
  /mnt/hdd4tb/jaemo/VGGSound/vggsound_train_manifest.csv          # successful manifest
  /mnt/hdd4tb/jaemo/VGGSound/vggsound_train_failures.csv          # failure log
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SR = 48000  # CLAP HTSAT-tiny operates at 48 kHz mono


def fetch_one(yid: str, start: int, label: str, audio_dir: Path) -> tuple[str, str | None, str | None]:
    """Download one 10s clip. Returns (yid, audio_path or None, error or None)."""
    out_path = audio_dir / f"{yid}_{start}.wav"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return yid, str(out_path), None
    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        tmp_audio = tmpd / f"{yid}.m4a"
        # yt-dlp: extract audio only, no playlist, fail fast
        cmd = [
            "yt-dlp",
            "-q", "--no-warnings",
            "-f", "bestaudio[ext=m4a]/bestaudio/best",
            "-o", str(tmp_audio),
            "--no-playlist",
            "--socket-timeout", "30",
            f"https://www.youtube.com/watch?v={yid}",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            if r.returncode != 0 or not tmp_audio.exists():
                return yid, None, (r.stderr.decode("utf8", "ignore")[:200] or "yt-dlp failed")
        except subprocess.TimeoutExpired:
            return yid, None, "yt-dlp timeout"
        except Exception as exc:
            return yid, None, f"yt-dlp exc: {exc}"

        # ffmpeg: cut [start, start+10), 48kHz mono wav
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start),
            "-i", str(tmp_audio),
            "-t", "10",
            "-ac", "1",
            "-ar", str(SR),
            str(out_path),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            if r.returncode != 0 or not out_path.exists() or out_path.stat().st_size < 1000:
                return yid, None, (r.stderr.decode("utf8", "ignore")[:200] or "ffmpeg failed")
        except subprocess.TimeoutExpired:
            return yid, None, "ffmpeg timeout"
        except Exception as exc:
            return yid, None, f"ffmpeg exc: {exc}"

    return yid, str(out_path), None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-csv", type=Path,
                    default=Path("/mnt/hdd4tb/jaemo/VGGSound/vggsound_train_sample30k.csv"))
    ap.add_argument("--audio-dir", type=Path,
                    default=Path("/mnt/hdd4tb/jaemo/VGGSound/audio/train"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("/mnt/hdd4tb/jaemo/VGGSound/vggsound_train_manifest.csv"))
    ap.add_argument("--failures", type=Path,
                    default=Path("/mnt/hdd4tb/jaemo/VGGSound/vggsound_train_failures.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-n", type=int, default=10000,
                    help="stop early after this many successes")
    ap.add_argument("--max-attempts", type=int, default=30000,
                    help="upper bound on attempted downloads")
    args = ap.parse_args()

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.sample_csv)
    df = df.head(args.max_attempts)
    print(f"[start] candidates={len(df)}  workers={args.workers}  target={args.target_n}")

    # Resume support: scan existing audio
    existing = {p.stem.rsplit("_", 1)[0]: str(p) for p in args.audio_dir.glob("*.wav") if p.stat().st_size > 1000}
    print(f"[resume] existing wavs: {len(existing)}")

    successes = []
    failures = []
    t0 = time.time()
    n_attempted = 0

    def _make_task(row):
        return fetch_one(row.youtube_id, int(row.start), row.label, args.audio_dir)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        rows = list(df.itertuples(index=False))
        for r in rows:
            # if already exists with right naming, skip and add to successes
            if (args.audio_dir / f"{r.youtube_id}_{int(r.start)}.wav").exists() and \
               (args.audio_dir / f"{r.youtube_id}_{int(r.start)}.wav").stat().st_size > 1000:
                successes.append({"audio_path": str(args.audio_dir / f"{r.youtube_id}_{int(r.start)}.wav"),
                                   "label": r.label, "youtube_id": r.youtube_id,
                                   "start": int(r.start), "split": r.split})
                if len(successes) >= args.target_n:
                    break
                continue
            futures[ex.submit(_make_task, r)] = r

        for fut in as_completed(futures):
            r = futures[fut]
            n_attempted += 1
            yid, path, err = fut.result()
            if path:
                successes.append({"audio_path": path, "label": r.label, "youtube_id": r.youtube_id,
                                   "start": int(r.start), "split": r.split})
            else:
                failures.append({"youtube_id": r.youtube_id, "start": int(r.start),
                                 "label": r.label, "error": (err or "unknown")[:200]})
            if (n_attempted % 50) == 0:
                el = time.time() - t0
                rate = n_attempted / max(el, 1e-6)
                succ_rate = 100.0 * len(successes) / max(n_attempted + len(existing), 1)
                eta_min = (args.target_n - len(successes)) / max(rate * succ_rate / 100, 1e-6) / 60
                print(f"  [{n_attempted}] succ={len(successes)} fail={len(failures)} "
                      f"rate={rate:.1f}/s succ%={succ_rate:.1f}% ETA={eta_min:.1f}m", flush=True)
            if len(successes) >= args.target_n:
                # cancel remaining
                for f2 in futures:
                    if not f2.done():
                        f2.cancel()
                break

    # Save manifests
    pd.DataFrame(successes).to_csv(args.manifest, index=False)
    pd.DataFrame(failures).to_csv(args.failures, index=False)
    print(f"\n[done] successes={len(successes)}  failures={len(failures)}  attempted={n_attempted}")
    print(f"[saved] {args.manifest}  ({args.manifest.stat().st_size if args.manifest.exists() else 0} bytes)")
    print(f"[saved] {args.failures}  ({args.failures.stat().st_size if args.failures.exists() else 0} bytes)")


if __name__ == "__main__":
    main()
