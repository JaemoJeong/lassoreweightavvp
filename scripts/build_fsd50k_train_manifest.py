from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ALLOWED_EXTS = {".wav", ".ogg", ".flac", ".mp3", ".m4a"}


def build_audio_index(audio_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in audio_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXTS:
            continue
        index[path.stem] = str(path.resolve())
    return index


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a train-only manifest for official FSD50K dev split."
    )
    ap.add_argument(
        "--ground-truth-csv",
        type=Path,
        default=Path("/mnt/hdd4tb/jaemo/FSD50K/FSD50K.ground_truth/dev.csv"),
        help="Official FSD50K.ground_truth/dev.csv path.",
    )
    ap.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/mnt/hdd4tb/jaemo/FSD50K/FSD50K.dev_audio"),
        help="Root directory containing FSD50K dev audio files.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/mnt/hdd4tb/jaemo/FSD50K/fsd50k_train_manifest.csv"),
        help="Output CSV path.",
    )
    args = ap.parse_args()

    if not args.ground_truth_csv.exists():
        raise SystemExit(f"missing ground-truth csv: {args.ground_truth_csv}")
    if not args.audio_root.exists():
        raise SystemExit(f"missing audio root: {args.audio_root}")

    df = pd.read_csv(args.ground_truth_csv)
    required = {"fname", "labels", "mids", "split"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"dev.csv missing required columns: {sorted(missing)}")

    train_df = df[df["split"].astype(str).str.lower() == "train"].copy()
    audio_index = build_audio_index(args.audio_root)

    train_df["fname"] = train_df["fname"].astype(str)
    train_df["audio_path"] = train_df["fname"].map(audio_index)
    found_df = train_df[train_df["audio_path"].notna()].copy()
    missing_df = train_df[train_df["audio_path"].isna()].copy()

    out_cols = ["audio_path", "fname", "labels", "mids", "split"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    found_df[out_cols].to_csv(args.out_csv, index=False)

    print(f"[train rows] {len(train_df)}")
    print(f"[audio found] {len(found_df)}")
    print(f"[audio missing] {len(missing_df)}")
    print(f"[saved] {args.out_csv}")
    if len(missing_df):
        miss_path = args.out_csv.with_name(args.out_csv.stem + "_missing.csv")
        missing_df[["fname", "labels", "mids", "split"]].to_csv(miss_path, index=False)
        print(f"[saved missing] {miss_path}")


if __name__ == "__main__":
    main()
