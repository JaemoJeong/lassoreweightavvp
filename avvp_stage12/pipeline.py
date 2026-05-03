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
    kappa: float
    eta: float
    rho_min: float
    rho_max: float
    fista_iters: int
    device: str
    max_stage: int = 2
    prior_mode: str = "full"


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


def prepare_modality(
    name: str,
    segment_raw: np.ndarray,
    video_raw: np.ndarray,
    proto_raw: np.ndarray,
    segment_mean_override: np.ndarray | None = None,
    video_mean_override: np.ndarray | None = None,
) -> PreparedModality:
    num_videos, num_segments, d_dim = segment_raw.shape
    if video_raw.shape != (num_videos, d_dim):
        raise ValueError(
            f"{name}: expected video shape {(num_videos, d_dim)}, got {video_raw.shape}"
        )
    segment_flat = segment_raw.reshape(-1, d_dim)
    segment_n, segment_center, segment_mean = center_rows(segment_flat, mean_vec=segment_mean_override)
    video_n, video_center, video_mean = center_rows(video_raw, mean_vec=video_mean_override)
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
    recon_center_flat = centered_reconstruction_cosine(
        weights_flat, modality.segment_center, modality.proto_center
    )
    out = _segment_stats(weights_flat, recon_flat, modality.num_videos, modality.num_segments)
    out["recon_center"] = recon_center_flat.reshape(modality.num_videos, modality.num_segments)
    return out


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


def compute_sparse_confidence(stage1_weights: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    top = stage1_weights.max(axis=2, keepdims=True)
    return (stage1_weights / (top + eps)).astype(np.float32)


def compute_reliable_sparse_confidence(
    stage1_weights: np.ndarray,
    recon_center: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sparse_confidence = compute_sparse_confidence(stage1_weights, eps=eps)
    # q_m(t) is reconstruction quality. We keep only positive evidence so a
    # poor decomposition weakens the prior rather than becoming negative support.
    reconstruction_quality = np.clip(recon_center, 0.0, 1.0).astype(np.float32)
    reliable_confidence = (
        sparse_confidence * reconstruction_quality[:, :, None]
    ).astype(np.float32)
    return sparse_confidence, reconstruction_quality, reliable_confidence


def compute_local_support(
    stage1_weights: np.ndarray,
    recon_center: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Backward-compatible alias for older callers.
    return compute_reliable_sparse_confidence(stage1_weights, recon_center, eps=eps)


def compute_video_prior(reliable_confidence: np.ndarray) -> np.ndarray:
    return reliable_confidence.max(axis=1).astype(np.float32)


def compute_video_plausibility(local_support: np.ndarray) -> np.ndarray:
    # Backward-compatible alias for older naming.
    return compute_video_prior(local_support)


def build_cross_modal_prior(
    source_segment_prior: np.ndarray,
    source_video_prior: np.ndarray,
    kappa: float,
    prior_mode: str = "full",
) -> np.ndarray:
    video_prior = source_video_prior[:, None, :]
    if prior_mode == "video":
        return np.broadcast_to(video_prior, source_segment_prior.shape).astype(np.float32)
    if prior_mode != "full":
        raise ValueError(f"unknown prior_mode={prior_mode!r}; expected 'full' or 'video'")
    return (video_prior * (1.0 + float(kappa) * source_segment_prior)).astype(np.float32)


def build_prior_weighted_penalty(
    cross_modal_prior: np.ndarray,
    base_lambda: float,
    eta: float,
    rho_min: float,
    rho_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    penalty_scale = np.clip(
        np.exp(-float(eta) * cross_modal_prior),
        float(rho_min),
        float(rho_max),
    ).astype(np.float32)
    penalty = (float(base_lambda) * penalty_scale).reshape(-1, cross_modal_prior.shape[2])
    return penalty_scale.astype(np.float32), penalty.astype(np.float32)


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


def summarize_prior(prior: np.ndarray, penalty_scale: np.ndarray) -> dict[str, float]:
    return {
        "prior_mean": float(prior.mean()),
        "prior_std": float(prior.std()),
        "prior_max": float(prior.max()),
        "penalty_scale_mean": float(penalty_scale.mean()),
        "penalty_scale_std": float(penalty_scale.std()),
        "penalty_scale_min": float(penalty_scale.min()),
        "penalty_scale_max": float(penalty_scale.max()),
    }


def run_stage12(audio: PreparedModality, visual: PreparedModality, cfg: Stage12Config) -> dict[str, object]:
    if audio.num_videos != visual.num_videos or audio.num_segments != visual.num_segments:
        raise ValueError("audio/visual sample layout mismatch")
    if cfg.max_stage not in (1, 2):
        raise ValueError(f"max_stage must be 1 or 2, got {cfg.max_stage}")
    if cfg.prior_mode not in ("full", "video"):
        raise ValueError(f"prior_mode must be 'full' or 'video', got {cfg.prior_mode!r}")

    audio_stage1 = run_segment_decomposition(audio, cfg.lambda_a, cfg.fista_iters, cfg.device)
    visual_stage1 = run_segment_decomposition(visual, cfg.lambda_v, cfg.fista_iters, cfg.device)

    presence_a = compute_presence(audio_stage1["weights"])
    presence_v = compute_presence(visual_stage1["weights"])

    results: dict[str, object] = {
        "config": asdict(cfg),
        "audio": {
            "stage1": audio_stage1,
            "presence": presence_a,
            "summary_stage1": summarize_results(audio_stage1),
        },
        "visual": {
            "stage1": visual_stage1,
            "presence": presence_v,
            "summary_stage1": summarize_results(visual_stage1),
        },
    }

    if cfg.max_stage == 1:
        return results

    sparse_conf_a, quality_a, reliable_conf_a = compute_reliable_sparse_confidence(
        audio_stage1["weights"],
        audio_stage1["recon_center"],
    )
    sparse_conf_v, quality_v, reliable_conf_v = compute_reliable_sparse_confidence(
        visual_stage1["weights"],
        visual_stage1["recon_center"],
    )
    video_prior_a = compute_video_prior(reliable_conf_a)
    video_prior_v = compute_video_prior(reliable_conf_v)

    prior_v_to_a = build_cross_modal_prior(
        source_segment_prior=reliable_conf_v,
        source_video_prior=video_prior_v,
        kappa=cfg.kappa,
        prior_mode=cfg.prior_mode,
    )
    prior_a_to_v = build_cross_modal_prior(
        source_segment_prior=reliable_conf_a,
        source_video_prior=video_prior_a,
        kappa=cfg.kappa,
        prior_mode=cfg.prior_mode,
    )

    penalty_scale_a, penalty_a = build_prior_weighted_penalty(
        cross_modal_prior=prior_v_to_a,
        base_lambda=cfg.lambda_a,
        eta=cfg.eta,
        rho_min=cfg.rho_min,
        rho_max=cfg.rho_max,
    )
    penalty_scale_v, penalty_v = build_prior_weighted_penalty(
        cross_modal_prior=prior_a_to_v,
        base_lambda=cfg.lambda_v,
        eta=cfg.eta,
        rho_min=cfg.rho_min,
        rho_max=cfg.rho_max,
    )

    audio_stage2 = run_weighted_segment_decomposition(audio, penalty_a, cfg.fista_iters, cfg.device)
    visual_stage2 = run_weighted_segment_decomposition(visual, penalty_v, cfg.fista_iters, cfg.device)

    results["audio"].update({
        "sparse_confidence": sparse_conf_a,
        "reconstruction_quality": quality_a,
        "reliable_confidence": reliable_conf_a,
        "video_prior": video_prior_a,
        # Backward-compatible aliases for previous experiment artifacts.
        "reliability": quality_a,
        "local_support": reliable_conf_a,
        "plausibility": video_prior_a,
        "prior_from_visual": prior_v_to_a,
        "penalty_scale": penalty_scale_a,
        "evidence_from_visual": prior_v_to_a,
        "weighted_lambda": penalty_a.reshape(audio.num_videos, audio.num_segments, audio.num_classes),
        "stage2": audio_stage2,
        "summary_stage2": summarize_results(audio_stage2),
        "summary_prior": summarize_prior(prior_v_to_a, penalty_scale_a),
    })
    results["visual"].update({
        "sparse_confidence": sparse_conf_v,
        "reconstruction_quality": quality_v,
        "reliable_confidence": reliable_conf_v,
        "video_prior": video_prior_v,
        # Backward-compatible aliases for previous experiment artifacts.
        "reliability": quality_v,
        "local_support": reliable_conf_v,
        "plausibility": video_prior_v,
        "prior_from_audio": prior_a_to_v,
        "penalty_scale": penalty_scale_v,
        "evidence_from_audio": prior_a_to_v,
        "weighted_lambda": penalty_v.reshape(visual.num_videos, visual.num_segments, visual.num_classes),
        "stage2": visual_stage2,
        "summary_stage2": summarize_results(visual_stage2),
        "summary_prior": summarize_prior(prior_a_to_v, penalty_scale_v),
    })
    return results
