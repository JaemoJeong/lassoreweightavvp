from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .constants import (
    CACHE_ROOT,
    DATA_ROOT,
    DEFAULT_AUDIO_MEAN_FILE,
    DEFAULT_BACKBONE,
    DEFAULT_MEAN_SOURCE,
    DEFAULT_VISUAL_MEAN_FILE,
    DEFAULT_VOCAB,
    EVAL_AUDIO_CSV,
    EVAL_VISUAL_CSV,
    LLP_CATS,
    LLP_IDX,
    MEANS_ROOT,
    TEST_CSV,
    VOCAB_ROOT,
)


def parse_video_id(filename: str) -> str:
    return "_".join(filename.split("_")[:-2])


def load_llp_metadata(csv_path: Path = TEST_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path, sep="\t")


def _peek_shape(cache_dir: Path, filenames: list[str]) -> tuple[int, ...]:
    for fn in filenames:
        path = cache_dir / parse_video_id(fn) / "0_10.pt"
        if path.exists():
            tensor = torch.load(path, map_location="cpu")
            return tuple(tensor.shape)
    raise FileNotFoundError(f"no cached tensors found under {cache_dir}")


def load_segment_cache(
    modality: str,
    backbone: str = DEFAULT_BACKBONE,
    csv_path: Path = TEST_CSV,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = load_llp_metadata(csv_path)
    filenames = df.filename.tolist()
    cache_dir = CACHE_ROOT / modality / backbone
    shape = _peek_shape(cache_dir, filenames)
    if len(shape) != 2:
        raise ValueError(f"expected segment cache to have shape (T, D), got {shape} at {cache_dir}")
    t_dim, d_dim = shape
    features = np.zeros((len(filenames), t_dim, d_dim), dtype=np.float32)
    valid = np.zeros(len(filenames), dtype=bool)
    for idx, fn in enumerate(filenames):
        path = cache_dir / parse_video_id(fn) / "0_10.pt"
        if not path.exists():
            continue
        tensor = torch.load(path, map_location="cpu").float().numpy().astype(np.float32)
        features[idx] = tensor
        valid[idx] = True
    return features, valid, filenames


def load_global_audio_cache(
    backbone: str = DEFAULT_BACKBONE,
    csv_path: Path = TEST_CSV,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = load_llp_metadata(csv_path)
    filenames = df.filename.tolist()
    cache_dir = CACHE_ROOT / "global_audio" / backbone
    shape = _peek_shape(cache_dir, filenames)
    if len(shape) != 2:
        raise ValueError(f"expected global audio cache to have shape (1, D), got {shape} at {cache_dir}")
    _, d_dim = shape
    features = np.zeros((len(filenames), d_dim), dtype=np.float32)
    valid = np.zeros(len(filenames), dtype=bool)
    for idx, fn in enumerate(filenames):
        path = cache_dir / parse_video_id(fn) / "0_10.pt"
        if not path.exists():
            continue
        tensor = torch.load(path, map_location="cpu").float().numpy().astype(np.float32)
        features[idx] = tensor.reshape(-1)
        valid[idx] = True
    return features, valid, filenames


def load_llp_cached_bundle(
    backbone: str = DEFAULT_BACKBONE,
    csv_path: Path = TEST_CSV,
) -> dict[str, object]:
    audio_segments, valid_a, filenames = load_segment_cache("audio", backbone, csv_path)
    visual_segments, valid_v, _ = load_segment_cache("image", backbone, csv_path)
    global_audio, valid_ga, _ = load_global_audio_cache(backbone, csv_path)
    valid = valid_a & valid_v & valid_ga
    kept_filenames = [fn for fn, keep in zip(filenames, valid.tolist()) if keep]
    kept_video_ids = [parse_video_id(fn) for fn in kept_filenames]
    kept_audio_segments = audio_segments[valid]
    kept_visual_segments = visual_segments[valid]
    return {
        "filenames": kept_filenames,
        "video_ids": kept_video_ids,
        "valid_mask": valid,
        "audio_segments": kept_audio_segments,
        "visual_segments": kept_visual_segments,
        "audio_video": global_audio[valid],
        "visual_video": kept_visual_segments.mean(axis=1),
    }


def load_prompt_vocab(vocab: str = DEFAULT_VOCAB) -> dict[str, object]:
    meta = json.loads((VOCAB_ROOT / f"{vocab}.json").read_text())
    visual = np.load(VOCAB_ROOT / f"{vocab}_clip.npy").T.astype(np.float32)
    audio = np.load(VOCAB_ROOT / f"{vocab}_clap.npy").T.astype(np.float32)
    return {
        "name": vocab,
        "labels": meta["labels"],
        "meta": meta,
        "audio_rows": audio,
        "visual_rows": visual,
    }


def load_reference_means(
    mean_source: str = DEFAULT_MEAN_SOURCE,
    audio_mean_path: Path | None = None,
    visual_mean_path: Path | None = None,
) -> dict[str, np.ndarray] | None:
    if mean_source == "llp":
        return None
    if mean_source != "external":
        raise ValueError(f"unknown mean_source: {mean_source}")

    audio_path = audio_mean_path or (MEANS_ROOT / DEFAULT_AUDIO_MEAN_FILE)
    visual_path = visual_mean_path or (MEANS_ROOT / DEFAULT_VISUAL_MEAN_FILE)
    audio_mean = np.load(audio_path).astype(np.float32).reshape(-1)
    visual_mean = np.load(visual_path).astype(np.float32).reshape(-1)
    return {
        "audio": audio_mean,
        "visual": visual_mean,
        "audio_path": str(audio_path),
        "visual_path": str(visual_path),
    }


def build_dense_gt(filenames: list[str], modality: str) -> np.ndarray:
    if modality == "audio":
        csv_path = EVAL_AUDIO_CSV
    elif modality == "visual":
        csv_path = EVAL_VISUAL_CSV
    else:
        raise ValueError(f"unknown modality: {modality}")

    df = pd.read_csv(csv_path, sep="\t")
    fn_to_idx = {fn: idx for idx, fn in enumerate(filenames)}
    gt = np.zeros((len(filenames), 10, len(LLP_CATS)), dtype=np.uint8)
    for row in df.itertuples(index=False):
        idx = fn_to_idx.get(row.filename)
        if idx is None:
            continue
        if row.event_labels not in LLP_IDX:
            continue
        gt[idx, int(row.onset):int(row.offset), LLP_IDX[row.event_labels]] = 1
    return gt


def build_weak_video_labels(filenames: list[str], csv_path: Path = TEST_CSV) -> np.ndarray:
    df = load_llp_metadata(csv_path)
    row_map = {row.filename: row for row in df.itertuples(index=False)}
    weak = np.zeros((len(filenames), len(LLP_CATS)), dtype=np.uint8)
    for idx, fn in enumerate(filenames):
        row = row_map[fn]
        if not isinstance(row.event_labels, str) or not row.event_labels.strip():
            continue
        for label in row.event_labels.split(","):
            label = label.strip()
            if label in LLP_IDX:
                weak[idx, LLP_IDX[label]] = 1
    return weak
