"""Regression tests for os_precompute's cached-template-FFT convolution.

precompute_basis_lr_maps convolves every basis kernel against the *same*
template; the current implementation caches the template's forward FFT once
and reuses it for all n_ker kernels instead of recomputing it per kernel
(each via an independent scipy.signal.fftconvolve call, the prior
behavior). These tests pin numerical parity against that prior behavior
(reproduced here as _fftconvolve_valid, unaffected by the caching) and
exercise both the serial (n_ker < 4) and threaded (n_ker >= 4) code paths.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hotpants.pure.os_precompute import (
    _basis_lr_from_conv,
    _fftconvolve_valid,
    precompute_basis_lr_maps,
)


def _reference_basis_lr_maps(template_hr, kernel_vecs, oversample):
    """Old (uncached) behavior: independent fftconvolve per kernel."""
    F = int(oversample)
    tpl = np.ascontiguousarray(template_hr, dtype=np.float64)
    kstack = np.ascontiguousarray(np.asarray(kernel_vecs, dtype=np.float64))
    n_ker, kh, kw = kstack.shape
    half_r = kw // 2
    hr_ny, hr_nx = tpl.shape
    lr_ny, lr_nx = hr_ny // F, hr_nx // F
    out = np.empty((n_ker, lr_ny, lr_nx), dtype=np.float64)
    for k in range(n_ker):
        conv = _fftconvolve_valid(tpl, kstack[k])
        out[k] = _basis_lr_from_conv(conv, half_r, F, lr_ny, lr_nx)
    return out


@pytest.mark.parametrize("n_ker", [1, 3, 5, 8])
def test_matches_reference_uncached_convolution(n_ker):
    rng = np.random.default_rng(0)
    F = 4
    ny, nx = 5, 5  # LR shape; HR template is (ny*F, nx*F)
    tpl = rng.normal(size=(ny * F, nx * F))
    kh = kw = 9
    kernels = rng.normal(size=(n_ker, kh, kw))

    got = precompute_basis_lr_maps(tpl, kernels, F)
    want = _reference_basis_lr_maps(tpl, kernels, F)

    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, atol=1e-10, rtol=1e-10, equal_nan=True)


def test_output_shape_and_dtype():
    rng = np.random.default_rng(1)
    F = 2
    tpl = rng.normal(size=(20, 24))
    kernels = rng.normal(size=(6, 5, 5))
    out = precompute_basis_lr_maps(tpl, kernels, F)
    assert out.shape == (6, 10, 12)
    assert out.dtype == np.float64


def test_rejects_oversample_le_1():
    tpl = np.zeros((10, 10))
    kernels = np.zeros((2, 3, 3))
    with pytest.raises(ValueError):
        precompute_basis_lr_maps(tpl, kernels, 1)


def test_rejects_bad_kernel_vecs_shape():
    tpl = np.zeros((10, 10))
    with pytest.raises(ValueError):
        precompute_basis_lr_maps(tpl, np.zeros((3, 3)), 2)


def test_threaded_path_matches_serial_path_with_env_cap(monkeypatch):
    # n_ker >= 4 takes the ThreadPoolExecutor branch; force n_workers=1 via
    # HOTPANTS_OS_N_JOBS and confirm identical output to the natural
    # multi-worker run (same shared cached template FFT either way).
    rng = np.random.default_rng(2)
    F = 3
    tpl = rng.normal(size=(15, 18))
    kernels = rng.normal(size=(6, 5, 5))

    multi = precompute_basis_lr_maps(tpl, kernels, F)
    monkeypatch.setenv("HOTPANTS_OS_N_JOBS", "1")
    serial = precompute_basis_lr_maps(tpl, kernels, F)

    np.testing.assert_allclose(multi, serial, atol=1e-10, rtol=1e-10, equal_nan=True)
