from __future__ import annotations

import numpy as np
import torch


def l2_np(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(denom, eps, None)


def center_rows(
    rows: np.ndarray,
    mean_vec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_n = l2_np(rows.astype(np.float32), axis=1)
    if mean_vec is None:
        mean_vec = rows_n.mean(axis=0).astype(np.float32)
    centered = l2_np(rows_n - mean_vec[None, :], axis=1)
    return rows_n.astype(np.float32), centered.astype(np.float32), mean_vec.astype(np.float32)


def prepare_dictionary(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_n = l2_np(rows.astype(np.float32), axis=1)
    mean_vec = rows_n.mean(axis=0).astype(np.float32)
    centered = l2_np(rows_n - mean_vec[None, :], axis=1)
    return rows_n.astype(np.float32), centered.astype(np.float32), mean_vec


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _prepare_penalty_matrix(
    penalty: float | np.ndarray,
    num_rows: int,
    num_cols: int,
) -> np.ndarray:
    if np.isscalar(penalty):
        return np.full((num_rows, num_cols), float(penalty), dtype=np.float32)
    arr = np.asarray(penalty, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != num_cols:
            raise ValueError(f"1D penalty length mismatch: expected {num_cols}, got {arr.shape[0]}")
        return np.broadcast_to(arr[None, :], (num_rows, num_cols)).astype(np.float32)
    if arr.shape != (num_rows, num_cols):
        raise ValueError(f"penalty shape mismatch: expected {(num_rows, num_cols)}, got {arr.shape}")
    return arr.astype(np.float32)


def nonnegative_lasso_fista(
    z_rows: np.ndarray,
    c_rows: np.ndarray,
    penalty: float | np.ndarray,
    n_iter: int = 200,
    device: str = "cuda",
) -> np.ndarray:
    """Solve argmin_{W>=0} ||Z - W C||^2 + sum penalty * |W|."""
    if z_rows.ndim != 2 or c_rows.ndim != 2:
        raise ValueError("z_rows and c_rows must be 2D")
    num_rows, d_dim = z_rows.shape
    k_dim, c_dim = c_rows.shape
    if d_dim != c_dim:
        raise ValueError(f"dimension mismatch: Z is {z_rows.shape}, C is {c_rows.shape}")

    penalty_mat = _prepare_penalty_matrix(penalty, num_rows, k_dim)
    torch_device = _resolve_device(device)

    z_t = torch.as_tensor(z_rows, dtype=torch.float32, device=torch_device)
    c_t = torch.as_tensor(c_rows, dtype=torch.float32, device=torch_device)
    penalty_t = torch.as_tensor(penalty_mat, dtype=torch.float32, device=torch_device)

    gram = c_t @ c_t.T
    l_max = torch.linalg.eigvalsh(gram).max().item()
    lipschitz = 2.0 * l_max * 1.05 + 1e-6
    inv_l = 1.0 / lipschitz

    w = torch.zeros((num_rows, k_dim), dtype=torch.float32, device=torch_device)
    y = w.clone()
    t_k = 1.0

    for _ in range(n_iter):
        grad = 2.0 * (y @ c_t - z_t) @ c_t.T
        w_next = torch.clamp(y - inv_l * grad - inv_l * penalty_t, min=0.0)
        t_next = 0.5 * (1.0 + (1.0 + 4.0 * t_k * t_k) ** 0.5)
        y = w_next + ((t_k - 1.0) / t_next) * (w_next - w)
        w = w_next
        t_k = t_next

    return w.detach().cpu().numpy().astype(np.float32)


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1).astype(np.float32)


def centered_reconstruction_cosine(
    weights: np.ndarray,
    z_center: np.ndarray,
    c_center: np.ndarray,
) -> np.ndarray:
    recon_center = l2_np(weights @ c_center, axis=1)
    return rowwise_cosine(recon_center, z_center)


def step4_reconstruct(
    weights: np.ndarray,
    c_center: np.ndarray,
    z_mean: np.ndarray,
) -> np.ndarray:
    recon_center = l2_np(weights @ c_center, axis=1)
    return l2_np(recon_center + z_mean[None, :], axis=1)
