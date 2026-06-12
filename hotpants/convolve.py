"""
Standalone template convolution using a saved HOTPANTS kernel solution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np

from .config import HotpantsConfig
from .core import _get_ext

if TYPE_CHECKING:
    from .core import Hotpants


@dataclass
class KernelModel:
    """
    Portable kernel fit result for standalone template convolution.

    Attributes:
        kernel_solution: 1D coefficient vector of length ``n_comp_total + 1``.
            Includes both convolution-kernel and background polynomial terms;
            only the kernel portion is used by :func:`convolve_template`.
        config: Configuration used during fitting. Kernel-critical fields
            (``rkernel``, ``ko``, ``bgo``, ``ngauss``, ``deg_fixe``,
            ``sigma_gauss``, ``use_pca``) must match the fit.
        fit_shape: ``(ny, nx)`` image shape the kernel was fit on.
    """

    kernel_solution: np.ndarray
    config: HotpantsConfig
    fit_shape: Tuple[int, int]

    @classmethod
    def from_hotpants(cls, hp: Hotpants) -> KernelModel:
        """Build a :class:`KernelModel` from a fitted :class:`Hotpants` instance."""
        if "kernel_solution" not in hp.results:
            raise ValueError(
                "kernel_solution not found. Run iterative_fit_and_clip() before extracting a KernelModel."
            )
        return cls(
            kernel_solution=np.asarray(hp.results["kernel_solution"], dtype=np.float64),
            config=hp.config,
            fit_shape=hp.template_data.shape,
        )


def _config_for_convolution(config: HotpantsConfig, template: np.ndarray) -> HotpantsConfig:
    """Adapt a saved config for convolution on the given template."""
    ny, nx = template.shape
    params = config.to_dict()
    params["nx"] = nx
    params["ny"] = ny
    params["tuthresh"] = float(np.nanmax(template))
    params["tlthresh"] = float(np.nanmin(template))
    params["iuthresh"] = params["tuthresh"]
    params["ilthresh"] = params["tlthresh"]
    params["tuktresh"] = params["tuthresh"]
    params["iuktresh"] = params["iuthresh"]
    return HotpantsConfig(**params)


def _validate_kernel_config(config: HotpantsConfig) -> None:
    if len(config.deg_fixe) != config.ngauss or len(config.sigma_gauss) != config.ngauss:
        raise ValueError(
            f"ngauss ({config.ngauss}) must match lengths of "
            f"deg_fixe ({len(config.deg_fixe)}) and sigma_gauss ({len(config.sigma_gauss)})."
        )


def convolve_template(
    template: np.ndarray,
    kernel: Union[KernelModel, np.ndarray],
    config: HotpantsConfig | None = None,
) -> np.ndarray:
    """
    Convolve a template image with a saved HOTPANTS kernel solution.

    Returns the raw ``spatial_convolve`` output only (no spatial background
    polynomial is added). The template must have the same shape as the image
    the kernel was fit on.

    Args:
        template: 2D image to convolve.
        kernel: A :class:`KernelModel` or raw ``kernel_solution`` array.
        config: Required when ``kernel`` is a raw array; ignored when
            ``kernel`` is a :class:`KernelModel`.

    Returns:
        Convolved template as a ``float32`` array with the same shape as
        ``template``.
    """
    if isinstance(kernel, KernelModel):
        kernel_solution = kernel.kernel_solution
        config = kernel.config
        fit_shape = kernel.fit_shape
    else:
        if config is None:
            raise ValueError("config is required when kernel is a raw ndarray.")
        kernel_solution = kernel
        fit_shape = None

    if template.ndim != 2:
        raise ValueError(f"template must be a 2D array, got {template.ndim} dimensions.")

    template = np.ascontiguousarray(template, dtype=np.float32)
    ny, nx = template.shape

    if fit_shape is not None and template.shape != fit_shape:
        raise ValueError(
            f"template shape {template.shape} does not match kernel fit_shape {fit_shape}. "
            "Spatial kernel variation is tied to the image dimensions used during fitting."
        )

    _validate_kernel_config(config)

    kernel_solution = np.ascontiguousarray(kernel_solution, dtype=np.float64)
    expected_len = config.n_comp_total + 1
    if kernel_solution.size != expected_len:
        raise ValueError(
            f"kernel_solution length {kernel_solution.size} does not match "
            f"expected length {expected_len} (n_comp_total + 1)."
        )

    conv_config = _config_for_convolution(config, template)
    ext = _get_ext()
    state = ext.HotpantsState(nx, ny, conv_config.to_dict())

    noise_sq = np.zeros((ny, nx), dtype=np.float32)
    convolved, _, _ = ext.apply_kernel(state, template, kernel_solution, noise_sq)
    return convolved
