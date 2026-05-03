from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .solver import (
    center_rows,
    centered_reconstruction_cosine,
    nonnegative_lasso_fista,
    prepare_dictionary,
    step4_reconstruction_cosine,
)


@dataclass
class Stage12Config:
    lambda_a: float
    lambda_v: float
    lambda_video_a: float
    lambda_video_v: float
    beta: float
    gamma: float
    lambda_min_factor: float
    fista_iters: int
    device: str


@dataclass
class PreparedModality:
    name: str
    segment_raw: np.ndarray
    video_raw: np.ndarray
    proto_raw: np.ndarray
    segment_n: np.ndarray
    segment_center: np.ndarray
    segment_mean: np.ndarray
    video_n: np.ndarray
    video_center: np.ndarray
    video_mean: np.ndarray
    proto_n: np.ndarray
    proto_center: np.ndarray
    proto_mean: np.ndarray
    num_videos: int
    num_segments: int
    num_classes: int
    d_dim: int


def prepare_modality(name: str, segment_raw: np.ndarray, video_raw: np.ndarray, proto_raw: np.ndarray) -> PreparedModality:
    num_videos, num_segments, d_dim = segment_raw.shape
    if video_raw.shape != (num_videos, d_dim):
        raise ValueError(
            f"{name}: expected video shape {(num_videos, d_dim)}, got {video_raw.shape}"
        )
    segment_flat = segment_raw.reshape(-1, d_dim)
    segment_n, segment_center, segment_mean = center_rows(segment_flat)
    video_n, video_center, video_mean = center_rows(video_raw)
    proto_n, proto_center, proto_mean = prepare_dictionary(proto_raw)
    return PreparedModality(
        name=name,
        segment_raw=segment_raw.astype(np.float32),
        video_raw=video_raw.astype(np.float32),
        proto_raw=proto_raw.astype(np.float32),
        segment_n=segment_n.astype(np.float32),
        segment_center=segment_center.astype(np.float32),
        segment_mean=segment_mean.astype(np.float32),
        video_n=video_n.astype(np.float32),
        video_center=video_center.astype(np.float32),
        video_mean=video_mean.astype(np.float32),
        proto_n=proto_n.astype(np.float32),
        proto_center=proto_center.astype(np.float32),
        proto_mean=proto_mean.astype(np.float32),
        num_videos=num_videos,
        num_segments=num_segments,
        num_classes=proto_raw.shape[0],
        d_dim=d_dim,
    )


def _segment_stats(weights_flat: np.ndarray, recon_flat: np.ndarray, num_videos: int, num_segments: int) -> dict[str, np.ndarray]:
    weights = weights_flat.reshape(num_videos, num_segments, -1)
    recon = recon_flat.reshape(num_videos, num_segments)
    l0 = (weights > 1e-6).sum(axis=2).astype(np.float32)
    return {
        "weights": weights,
        "recon": recon,
        "l0": l0,
    }


def run_segment_decomposition(modality: PreparedModality, lam: float, iters: int, device: str) -> dict[str, np.ndarray]:
    weights_flat = nonnegative_lasso_fista(
        modality.segment_center,
        modality.proto_center,
        penalty=lam,
        n_iter=iters,
        device=device,
    )
    # SpLiCE Step-4 recon cos: cos(ẑ, z_n) where ẑ = σ(σ(C̃·w) + μ_z)
    z_n_flat = modality.segment_n.reshape(-1, modality.d_dim)
    recon_flat = step4_reconstruction_cosine(
        weights_flat, z_n_flat, modality.proto_center, modality.segment_mean
    )
    return _segment_stats(weights_flat, recon_flat, modality.num_videos, modality.num_segments)


def run_video_decomposition(modality: PreparedModality, lam: float, iters: int, device: str) -> dict[str, np.ndarray]:
    weights = nonnegative_lasso_fista(
        modality.video_center,
        modality.proto_center,
        penalty=lam,
        n_iter=iters,
        device=device,
    )
    recon = step4_reconstruction_cosine(
        weights, modality.video_n, modality.proto_center, modality.video_mean
    )
    l0 = (weights > 1e-6).sum(axis=1).astype(np.float32)
    return {
        "weights": weights,
        "recon": recon,
        "l0": l0,
    }


def compute_presence(stage1_weights: np.ndarray) -> np.ndarray:
    return stage1_weights.max(axis=1).astype(np.float32)


def compute_active_rejection(video_weights: np.ndarray) -> np.ndarray:
    top = video_weights.max(axis=1, keepdims=True)
    return np.maximum(0.0, top - video_weights).astype(np.float32)


def build_weighted_penalty(
    source_presence: np.ndarray,
    source_absence: np.ndarray,
    base_lambda: float,
    beta: float,
    gamma: float,
    lambda_min_factor: float,
    num_segments: int,
) -> tuple[np.ndarray, np.ndarray]:
    evidence = beta * source_presence[:, None, :] - gamma * source_absence[:, None, :]
    evidence = np.broadcast_to(
        evidence,
        (source_presence.shape[0], num_segments, source_presence.shape[1]),
    ).astype(np.float32)
    penalty_scale = np.maximum(lambda_min_factor, 1.0 - evidence).astype(np.float32)
    penalty = (base_lambda * penalty_scale).reshape(-1, source_presence.shape[1]).astype(np.float32)
    return evidence.astype(np.float32), penalty


def run_weighted_segment_decomposition(
    modality: PreparedModality,
    penalty_flat: np.ndarray,
    iters: int,
    device: str,
) -> dict[str, np.ndarray]:
    weights_flat = nonnegative_lasso_fista(
        modality.segment_center,
        modality.proto_center,
        penalty=penalty_flat,
        n_iter=iters,
        device=device,
    )
    z_n_flat = modality.segment_n.reshape(-1, modality.d_dim)
    recon_flat = step4_reconstruction_cosine(
        weights_flat, z_n_flat, modality.proto_center, modality.segment_mean
    )
    return _segment_stats(weights_flat, recon_flat, modality.num_videos, modality.num_segments)


def summarize_results(results: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "recon_mean": float(results["recon"].mean()),
        "recon_std": float(results["recon"].std()),
        "l0_mean": float(results["l0"].mean()),
        "l0_std": float(results["l0"].std()),
    }


def run_stage12(audio: PreparedModality, visual: PreparedModality, cfg: Stage12Config) -> dict[str, object]:
    if audio.num_videos != visual.num_videos or audio.num_segments != visual.num_segments:
        raise ValueError("audio/visual sample layout mismatch")

    audio_stage1 = run_segment_decomposition(audio, cfg.lambda_a, cfg.fista_iters, cfg.device)
    visual_stage1 = run_segment_decomposition(visual, cfg.lambda_v, cfg.fista_iters, cfg.device)

    presence_a = compute_presence(audio_stage1["weights"])
    presence_v = compute_presence(visual_stage1["weights"])

    audio_video = run_video_decomposition(audio, cfg.lambda_video_a, cfg.fista_iters, cfg.device)
    visual_video = run_video_decomposition(visual, cfg.lambda_video_v, cfg.fista_iters, cfg.device)

    absence_a = compute_active_rejection(audio_video["weights"])
    absence_v = compute_active_rejection(visual_video["weights"])

    evidence_v_to_a, penalty_a = build_weighted_penalty(
        source_presence=presence_v,
        source_absence=absence_v,
        base_lambda=cfg.lambda_a,
        beta=cfg.beta,
        gamma=cfg.gamma,
        lambda_min_factor=cfg.lambda_min_factor,
        num_segments=audio.num_segments,
    )
    evidence_a_to_v, penalty_v = build_weighted_penalty(
        source_presence=presence_a,
        source_absence=absence_a,
        base_lambda=cfg.lambda_v,
        beta=cfg.beta,
        gamma=cfg.gamma,
        lambda_min_factor=cfg.lambda_min_factor,
        num_segments=visual.num_segments,
    )

    audio_stage2 = run_weighted_segment_decomposition(audio, penalty_a, cfg.fista_iters, cfg.device)
    visual_stage2 = run_weighted_segment_decomposition(visual, penalty_v, cfg.fista_iters, cfg.device)

    return {
        "config": asdict(cfg),
        "audio": {
            "stage1": audio_stage1,
            "video": audio_video,
            "presence": presence_a,
            "absence": absence_a,
            "evidence_from_visual": evidence_v_to_a,
            "weighted_lambda": penalty_a.reshape(audio.num_videos, audio.num_segments, audio.num_classes),
            "stage2": audio_stage2,
            "summary_stage1": summarize_results(audio_stage1),
            "summary_stage2": summarize_results(audio_stage2),
            "summary_video": summarize_results(audio_video),
        },
        "visual": {
            "stage1": visual_stage1,
            "video": visual_video,
            "presence": presence_v,
            "absence": absence_v,
            "evidence_from_audio": evidence_a_to_v,
            "weighted_lambda": penalty_v.reshape(visual.num_videos, visual.num_segments, visual.num_classes),
            "stage2": visual_stage2,
            "summary_stage1": summarize_results(visual_stage1),
            "summary_stage2": summarize_results(visual_stage2),
            "summary_video": summarize_results(visual_video),
        },
    }
