"""
Accelerated oversampled stamp fill helpers (``os_precompute``).

For oversample>1, precompute LR maps of (template ⊛ kernel_basis_k)
block-sum-downsampled to science resolution, then gather per stamp/region.

Convolution uses ``scipy.signal.fftconvolve`` (float64), which matches
``jit_convolve_patch`` to ~1e-15 relative. A JAX FFT prototype was too
noisy for ill-conditioned Alard local solves with large ``deg_fixe``, so
this module intentionally does not use JAX.

Bases are convolved in a thread pool (``HOTPANTS_OS_N_JOBS``).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.signal import fftconvolve


def block_sum_downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Block-sum downsample; faster than the pure-Python utils.downsample_image."""
    if factor == 1:
        return np.asarray(image)
    img = np.asarray(image)
    ny, nx = img.shape
    new_ny, new_nx = ny // factor, nx // factor
    return (
        img[: new_ny * factor, : new_nx * factor]
        .reshape(new_ny, factor, new_nx, factor)
        .sum(axis=(1, 3))
    )


def _basis_lr_from_conv(
    conv_valid: np.ndarray, half_r: int, oversample: int, lr_ny: int, lr_nx: int
) -> np.ndarray:
    """
    Pad valid HR convolution by half_r so standard block-sum aligns with
    populate_*_vectors patch indexing, then downsample to LR.
    """
    padded = np.pad(
        conv_valid,
        ((half_r, half_r), (half_r, half_r)),
        mode="constant",
        constant_values=np.nan,
    )
    F = int(oversample)
    expect = (lr_ny * F, lr_nx * F)
    if padded.shape != expect:
        out = np.full(expect, np.nan, dtype=np.float64)
        hy = min(padded.shape[0], expect[0])
        hx = min(padded.shape[1], expect[1])
        out[:hy, :hx] = padded[:hy, :hx]
        padded = out
    return block_sum_downsample(padded, F).astype(np.float64, copy=False)


def precompute_basis_lr_maps(
    template_hr: np.ndarray,
    kernel_vecs,
    oversample: int,
) -> np.ndarray:
    """
    Parameters
    ----------
    template_hr : (ny*F, nx*F)
    kernel_vecs : list or (n_ker, kh, kw)
    oversample : F > 1

    Returns
    -------
    basis_lr : (n_ker, ny, nx) float64
    """
    F = int(oversample)
    if F <= 1:
        raise ValueError("precompute_basis_lr_maps is for oversample>1")

    tpl = np.ascontiguousarray(template_hr, dtype=np.float64)
    kstack = np.ascontiguousarray(np.asarray(kernel_vecs, dtype=np.float64))
    if kstack.ndim != 3:
        raise ValueError(f"kernel_vecs must be (n,kh,kw), got {kstack.shape}")

    n_ker, kh, kw = kstack.shape
    half_r = kw // 2
    hr_ny, hr_nx = tpl.shape
    lr_ny, lr_nx = hr_ny // F, hr_nx // F

    out = np.empty((n_ker, lr_ny, lr_nx), dtype=np.float64)
    n_workers = min(
        n_ker, max(1, int(os.environ.get("HOTPANTS_OS_N_JOBS", os.cpu_count() or 4)))
    )

    def _one(k: int) -> np.ndarray:
        conv = fftconvolve(tpl, kstack[k], mode="valid")
        return _basis_lr_from_conv(np.asarray(conv, dtype=np.float64), half_r, F, lr_ny, lr_nx)

    if n_workers <= 1 or n_ker < 4:
        for k in range(n_ker):
            out[k] = _one(k)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for k, lr in enumerate(ex.map(_one, range(n_ker))):
                out[k] = lr
    return out


def gather_basis_vectors(basis_lr: np.ndarray, ys, xs) -> np.ndarray:
    """Gather (n_ker, n_pix) from basis_lr[k, ys, xs]."""
    ys = np.asarray(ys, dtype=np.int64)
    xs = np.asarray(xs, dtype=np.int64)
    return np.ascontiguousarray(basis_lr[:, ys, xs], dtype=np.float64)
