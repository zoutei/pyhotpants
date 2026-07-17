# hotpants.py
"""
HOTPANTS Python Wrapper - Modular Implementation

This module provides a complete Python interface to the HOTPANTS image differencing
algorithms. Each step of the pipeline is exposed as an individual method for
fine-grained control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
from astropy.io import fits
import getpass
import socket
from datetime import datetime

from .config import HotpantsConfig
from .models import Substamp, SubstampStatus
from . import functions as pyhotpants
from . import pure

# C extension will be imported dynamically
hotpants_ext = None


def _get_ext():
    """Lazy import of the C extension module."""
    global hotpants_ext
    if hotpants_ext is None:
        try:
            # This relative import assumes the compiled C extension is in the same package
            from . import hotpants_ext as ext_module

            hotpants_ext = ext_module
        except ImportError as e:
            raise RuntimeError(f"Could not import C extension: {e}")
    return hotpants_ext


__version__ = "0.1.2"


class HotpantsError(Exception):
    """Exception raised for HOTPANTS-specific errors."""

    pass


class Hotpants:
    """
    The central HOTPANTS object for stateful image differencing.

    This class orchestrates the entire image subtraction pipeline. It holds the
    input data, configuration, and intermediate results. The public methods
    correspond to the major steps of the pipeline, allowing for either a full
    end-to-end run or step-by-step execution for detailed analysis.

    Example:
        >>> from hotpants import Hotpants, HotpantsConfig
        >>> # Create a custom configuration
        >>> config = HotpantsConfig(rkernel=15, normalize='i')
        >>> # Initialize with FITS file paths
        >>> hp = Hotpants('template.fits', 'science.fits', config=config)
        >>> # Run the entire pipeline
        >>> results = hp.run_pipeline()
        >>> # Access the difference image
        >>> diff_image = results['diff_image']
    """

    def __init__(
        self,
        template_data: Union[np.ndarray, str],
        image_data: Union[np.ndarray, str],
        t_mask: Optional[np.ndarray] = None,
        i_mask: Optional[np.ndarray] = None,
        t_error: Optional[np.ndarray] = None,
        i_error: Optional[np.ndarray] = None,
        star_catalog: Optional[np.ndarray] = None,
        config: Optional[HotpantsConfig] = None,
        output_header: Optional[fits.Header] = None,
        use_c_extension: bool = True,
        oversample: int = 1, # Feature: High-Res Template
    ):
        """
        Initializes the Hotpants object and performs pre-processing.

        This sets up the pipeline by loading images, creating initial masks,
        and generating noise models.

        Args:
            template_data: The template image data as a 2D NumPy array or a
                path to a FITS file.
            image_data: The science image data as a 2D NumPy array or a
                path to a FITS file.
            t_mask: An optional integer mask for the template image where
                pixels with values > 0 are considered bad.
            i_mask: An optional integer mask for the science image.
            t_error: An optional error map for the template.
                If not provided, a noise model is generated automatically.
            i_error: An optional error map for the science image.
            star_catalog: A pre-existing array of star positions (shape [N, 2])
                to use for kernel fitting, bypassing the stamp search.
                Coordinates should be 1-based (FITS convention).
            config: A custom `HotpantsConfig` object. If None, default
                parameters are used.
            output_header: An `astropy.io.fits.Header` object to use for all
                output FITS files. If loading from FITS files, the header of
                the science image is used by default.
            use_c_extension: boolean to use C extension or pure python
        """
        self.use_c_extension = use_c_extension
        self.oversample = int(oversample)

        if self.use_c_extension:
            self.ext = _get_ext()
            if self.oversample > 1:
                 raise NotImplementedError("Oversampled templates are only supported in Pure Python mode currently.")
        else:
            self.ext = None
            
        self.output_header = output_header

        if isinstance(template_data, str):
            self.template_path = template_data
            template_data, self.template_header = self._load_fits(template_data)
        else:
            self.template_path = "in-memory"
            self.template_header = None

        if isinstance(image_data, str):
            self.image_path = image_data
            image_data, self.image_header = self._load_fits(image_data)
            if self.output_header is None:
                self.output_header = self.image_header
        else:
            self.image_path = "in-memory"
            self.image_header = None

        if self.oversample == 1:
            self._validate_images(template_data, image_data, "template and image")
        else:
            if template_data.ndim != 2 or image_data.ndim != 2:
                 raise HotpantsError("template and image must be 2D arrays")
            
            # Strict dimension check for oversampled templates
            expected_shape = (image_data.shape[0] * self.oversample, image_data.shape[1] * self.oversample)
            if template_data.shape != expected_shape:
                raise HotpantsError(
                    f"Template shape {template_data.shape} does not match "
                    f"Science shape {image_data.shape} * oversample {self.oversample} = {expected_shape}"
                )

        self.template_data = np.ascontiguousarray(template_data, dtype=np.float32)
        self.image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        # Template (possibly HR) dims; science dims are always LR.
        self.ny, self.nx = self.template_data.shape
        self.ny_lr, self.nx_lr = self.image_data.shape

        # Oversampled mode only supports convolving the template → science.
        if self.oversample > 1:
            fc = str(getattr(config, "force_convolve", "b") if config is not None else "b")
            if fc == "i":
                raise HotpantsError(
                    "oversample>1 only supports convolving the template (force_convolve='t'); "
                    "direction 'i' is not supported."
                )

        # Size stamp grid / fwstamp from science (LR) pixels, not HR template pixels.
        self.config = config if config is not None else HotpantsConfig(nx=self.nx_lr, ny=self.ny_lr)
        if self.oversample > 1:
            if self.config.force_convolve in ("b", "i"):
                if self.config.verbose >= 1 and self.config.force_convolve == "b":
                    print("oversample>1: forcing conv_direction='t' (image→template not supported).")
                self.config.force_convolve = "t"
        # Always recompute fwstamp from LR dims (HotpantsConfig defaults assume 2048²).
        fwstamp_est = min(
            self.nx_lr / self.config.nregx / self.config.nstampx,
            self.ny_lr / self.config.nregy / self.config.nstampy,
        )
        fwstamp_est -= self.config.fwkernel
        fwstamp_est -= 1 if int(fwstamp_est) % 2 == 0 else 0
        self.config.fwstamp = int(max(fwstamp_est, self.config.fwksstamp + self.config.fwkernel))

        if t_error is not None:
            t_error_arr = np.asarray(t_error)
            self._validate_aux_array(t_error_arr, "t_error")
            self._t_error_input = np.ascontiguousarray(t_error_arr, dtype=np.float32)
        else:
            self._t_error_input = None

        if i_error is not None:
            i_error_arr = np.asarray(i_error)
            self._validate_aux_array(i_error_arr, "i_error")
            self._i_error_input = np.ascontiguousarray(i_error_arr, dtype=np.float32)
        else:
            self._i_error_input = None

        if t_mask is not None:
            t_mask_arr = np.asarray(t_mask)
            self._validate_aux_array(t_mask_arr, "t_mask")
            self._t_mask_input = np.ascontiguousarray(t_mask_arr, dtype=np.int32)
        else:
            self._t_mask_input = None

        if i_mask is not None:
            i_mask_arr = np.asarray(i_mask)
            self._validate_aux_array(i_mask_arr, "i_mask")
            self._i_mask_input = np.ascontiguousarray(i_mask_arr, dtype=np.int32)
        else:
            self._i_mask_input = None

        # Leave NaNs in arrays (C / oversample=1 parity). Masking flags them later.
        self._t_nan_mask = np.isnan(self.template_data)
        self._i_nan_mask = np.isnan(self.image_data)

        if star_catalog is not None:
            if not isinstance(star_catalog, np.ndarray) or star_catalog.ndim != 2 or star_catalog.shape[1] != 2:
                raise HotpantsError("star_catalog must be a 2D NumPy array with shape (N, 2).")
            self.star_catalog = np.ascontiguousarray(star_catalog, dtype=np.float32) - 1
        else:
            self.star_catalog = None

        if str(getattr(self.config, "stamp_mode", "grid")) == "connected_regions":
            if self.use_c_extension:
                raise HotpantsError(
                    "stamp_mode='connected_regions' requires use_c_extension=False (pure Python only)."
                )
            if self.star_catalog is None:
                raise HotpantsError(
                    "stamp_mode='connected_regions' requires a star_catalog."
                )

        self.results = {"stage_timings": {}}
        # New master lists for substamp objects
        self.template_substamps: List[Substamp] = []
        self.image_substamps: List[Substamp] = []
        # Cached pure-Python kernel bases / OS LR basis maps (shared FOM+fit+apply)
        self._cached_kernel_basis = None
        self._cached_kernel_basis_scale = None
        self._cached_basis_lr_maps = None
        self._fom_stamps_by_region: Dict[int, Any] = {}

        # Dynamically set thresholds if not provided (nan-aware for masked JWST/FITS data)
        if self.config.tuthresh is None:
            self.config.tuthresh = np.nanmax(self.template_data)
        if self.config.tuktresh is None:
            self.config.tuktresh = self.config.tuthresh
        if self.config.tlthresh is None:
            self.config.tlthresh = np.nanmin(self.template_data)
        if self.config.iuthresh is None:
            self.config.iuthresh = np.nanmax(self.image_data)
        if self.config.iuktresh is None:
            self.config.iuktresh = self.config.iuthresh
        if self.config.ilthresh is None:
            self.config.ilthresh = np.nanmin(self.image_data)

        # Initialize C state object and pre-compute masks and noise images
        if self.use_c_extension:
            self._c_state = self.ext.HotpantsState(self.nx, self.ny, self.config.to_dict())
            if self.config.verbose >= 1:
                print(f"Initialized HOTPANTS state: {self._c_state}")
        else:
            self._c_state = None
            if self.config.verbose >= 1:
                print(f"Initialized HOTPANTS (Pure Python Mode)")

        # 1. Create the initial input mask (int32 bitflags; HR when oversample>1).
        if self.use_c_extension:
            input_mask = self.ext.make_input_mask(
                self._c_state, self.template_data, self.image_data,
                self._t_mask_input, self._i_mask_input,
            )
            input_mask_lr = input_mask
        else:
            input_mask = pure.utils.make_input_mask(
                self.template_data,
                self.image_data,
                self.config,
                self._t_mask_input,
                self._i_mask_input,
                oversample=self.oversample,
            )
            if self.oversample > 1:
                input_mask_lr = pure.utils.downsample_hr_mask_to_lr(input_mask, self.oversample)
            else:
                input_mask_lr = input_mask

        if self.config.verbose >= 1:
            print(f"Input mask created with shape: {input_mask.shape}, dtype: {input_mask.dtype}")
        self.results["input_mask"] = input_mask
        self.results["input_mask_lr"] = input_mask_lr

        # 2. Generate noise images using C extension if not provided by the user.
        if self._t_error_input is not None:
            # Use user-provided noise image, squared.
            t_noise_sq = np.ascontiguousarray(self._t_error_input**2, dtype=np.float32)
        else:
            # Generate noise image from scratch and square it.
            if self.use_c_extension:
                t_noise_sq = self.ext.calculate_noise_image(self._c_state, self.template_data, True)
            else:
                rn = self.config.trdnoise
                gain = self.config.tgain
                t_noise_sq = (rn / gain)**2 + np.abs(self.template_data) / gain

        if self._i_error_input is not None:
            # Use user-provided noise image, squared.
            i_noise_sq = np.ascontiguousarray(self._i_error_input**2, dtype=np.float32)
        else:
            # Generate noise image from scratch and square it.
            if self.use_c_extension:
                i_noise_sq = self.ext.calculate_noise_image(self._c_state, self.image_data, False)
            else:
                rn = self.config.irdnoise
                gain = self.config.igain
                i_noise_sq = (rn / gain)**2 + np.abs(self.image_data) / gain

        # 3. Store the squared noise images for later use.
        self.results["t_noise_sq"] = t_noise_sq
        self.results["i_noise_sq"] = i_noise_sq
        if self.oversample > 1 and not self.use_c_extension:
            t_lr = pure.utils.downsample_image(t_noise_sq, self.oversample)
            self.results["combined_noise_sq_lr"] = t_lr + i_noise_sq
        else:
            self.results["combined_noise_sq_lr"] = t_noise_sq + i_noise_sq

    @staticmethod
    def _load_fits(filename: str) -> Tuple[np.ndarray, fits.Header]:
        """Loads FITS data, preferring extension 1, then primary."""
        with fits.open(filename) as hdul:
            if len(hdul) > 1:
                try:
                    data = hdul[1].data
                    header = hdul[1].header
                    if data is None:  # Check if extension has no data
                        data = hdul[0].data
                        header = hdul[0].header
                except IndexError:
                    data = hdul[0].data
                    header = hdul[0].header
            else:
                data = hdul[0].data
                header = hdul[0].header
        if data is None:
            raise HotpantsError(f"No image data found in FITS file: {filename}")
        return data.astype(np.float32), header

    def __del__(self):
        """Ensures the C state object is properly deallocated."""
        if hasattr(self, "_c_state") and self._c_state:
            # The C deallocator will be called automatically when the Python object is garbage collected.
            self._c_state = None

    def _validate_images(self, a1: np.ndarray, a2: np.ndarray, names: str):
        """Checks if two arrays are 2D and have the same shape."""
        if a1.ndim != 2 or a2.ndim != 2:
            raise HotpantsError(f"{names} must be 2D arrays")
        if a1.shape != a2.shape:
            raise HotpantsError(f"{names} must have the same dimensions")

    def _validate_aux_array(self, arr: np.ndarray, name: str):
        """Checks if an auxiliary array matches the image shape."""
        if arr.ndim != 2:
            raise HotpantsError(f"{name} must be a 2D array")
        if self.oversample == 1:
            expected = (self.ny, self.nx)
        elif name.startswith("t_"):
            expected = (self.ny, self.nx)
        else:
            # Science-side auxiliaries stay at native (non-oversampled) resolution.
            expected = (self.ny_lr, self.nx_lr)
        if arr.shape != expected:
            raise HotpantsError(f"{name} must have shape {expected}")

    def _ensure_kernel_basis(self, scale: int):
        """Return cached kernel basis list for the given oversample scale."""
        scale = int(scale)
        if self._cached_kernel_basis is not None and self._cached_kernel_basis_scale == scale:
            return self._cached_kernel_basis
        hr_rkernel = self.config.rkernel * scale
        k_size = 2 * hr_rkernel + 1
        # calculate_kernel_basis uses sigma_gauss as C-style coeffs in
        # exp(-x^2 * coeff). Physical Gaussian width must grow with
        # oversample F, so coeffs scale as 1/F^2 (NOT *F). Scaling by F
        # made OS>=4 bases far too sharp and produced ring residuals.
        if scale > 1:
            scaled_sigmas = [s / (scale ** 2) for s in self.config.sigma_gauss]
        else:
            scaled_sigmas = list(self.config.sigma_gauss)
        basis = pure.kernel.calculate_kernel_basis(
            (k_size, k_size), scaled_sigmas, self.config.deg_fixe
        )
        self._cached_kernel_basis = basis
        self._cached_kernel_basis_scale = scale
        return basis

    def _ensure_basis_lr_maps(self, template_hr, scale: int):
        """
        Precompute LR maps of template ⊛ each kernel basis (oversample>1 only).
        SciPy FFT convolution + thread pool; result cached on the instance.
        """
        scale = int(scale)
        if scale <= 1:
            return None
        if self._cached_basis_lr_maps is not None and self._cached_kernel_basis_scale == scale:
            return self._cached_basis_lr_maps
        import time as _time

        t0 = _time.perf_counter()
        basis = self._ensure_kernel_basis(scale)
        from .pure.os_precompute import precompute_basis_lr_maps

        maps = precompute_basis_lr_maps(template_hr, basis, scale)
        self._cached_basis_lr_maps = maps
        self.results.setdefault("stage_timings", {})["precompute_basis_lr"] = (
            _time.perf_counter() - t0
        )
        if self.config.verbose >= 1:
            print(
                f"Precomputed OS={scale} LR basis maps {maps.shape} "
                f"in {self.results['stage_timings']['precompute_basis_lr']:.2f}s",
                flush=True,
            )
        return maps

    def find_stamps(self) -> Tuple[List[Substamp], List[Substamp]]:
        """
        Step 1: Finds potential substamp coordinates for kernel fitting.

        This method scans the template and science images for suitable stars
        to use for constructing the convolution kernel. If a `star_catalog` was
        provided during initialization, this catalog is used directly,
        bypassing the automated search. Otherwise, a grid-based search is
        performed to find bright, isolated stars.

        All substamp coordinates are in science-image (LR) pixel space.

        It populates the `template_substamps` and `image_substamps` lists with
        `Substamp` objects, which initially contain only coordinate information.

        Returns:
            A tuple containing two lists: the `Substamp` objects found on the
            template and the `Substamp` objects found on the science image.
        """
        if self.use_c_extension:
            t_substamps_coords, i_substamps_coords = self.ext.find_stamps(
                self._c_state, self.template_data, self.image_data,
                self.config.fitthresh, self.star_catalog,
            )
            self.template_substamps = [Substamp(**coords) for coords in t_substamps_coords]
            self.image_substamps = [Substamp(**coords) for coords in i_substamps_coords]
        elif str(getattr(self.config, "stamp_mode", "grid")) == "connected_regions":
            # Connected irregular stamps from gated catalog (pure Python only).
            mask_lr = self.results.get("input_mask_lr", self.results["input_mask"])
            t_coords, i_coords, region_map = pure.regions.find_stamps_connected_regions(
                self.template_data,
                self.image_data,
                mask_lr,
                self.star_catalog,
                self.config,
                oversample=self.oversample,
                flux_image=self.image_data,
            )
            self.results["region_map"] = region_map
            self.template_substamps = [
                Substamp(
                    substamp_id=c["substamp_id"],
                    stamp_group_id=c["stamp_group_id"],
                    x=c["x"],
                    y=c["y"],
                    region_id=c.get("region_id"),
                )
                for c in t_coords
            ]
            self.image_substamps = [
                Substamp(
                    substamp_id=c["substamp_id"],
                    stamp_group_id=c["stamp_group_id"],
                    x=c["x"],
                    y=c["y"],
                    region_id=c.get("region_id"),
                )
                for c in i_coords
            ]
            if self.oversample > 1:
                self.image_substamps = []
            if self.config.verbose >= 1:
                print(
                    f"Connected regions: {len(region_map.regions)} regions from "
                    f"catalog ({len(self.star_catalog)} stars)."
                )
        elif self.star_catalog is not None:
            # Catalog path (C buildStamps parity). Coordinates are LR.
            t_coords, i_coords = pure.utils.find_stamps_from_catalog(
                self.template_data,
                self.image_data,
                self.results["input_mask"],
                self.star_catalog,
                self.config,
                oversample=self.oversample,
            )
            self.template_substamps = [Substamp(**coords) for coords in t_coords]
            self.image_substamps = [Substamp(**coords) for coords in i_coords]
            # Oversampled mode never convolves the science image.
            if self.oversample > 1:
                self.image_substamps = []
        else:
            mask_lr = self.results["input_mask_lr"]
            n_want = self.config.nstampx * self.config.nstampy
            box = self.config.rss * 2 + 1

            # Template search always in LR coords (downsample HR template when F>1).
            if self.oversample > 1:
                t_search = pure.utils.downsample_image(self.template_data, self.oversample)
                t_mask_for_search = mask_lr
            else:
                t_search = self.template_data
                t_mask_for_search = self.results["input_mask"]

            t_stamps_found = pure.utils.find_stamps(
                t_search, t_mask_for_search, n_want, box,
            )
            self.template_substamps = [
                Substamp(substamp_id=i, x=c["x"], y=c["y"], stamp_group_id=i)
                for i, c in enumerate(t_stamps_found)
            ]

            # Image-side stamps only needed when direction 'i'/'b' may be selected.
            if self.oversample > 1:
                self.image_substamps = []
            else:
                i_stamps_found = pure.utils.find_stamps(
                    self.image_data, mask_lr, n_want, box,
                )
                n_t = len(self.template_substamps)
                self.image_substamps = [
                    Substamp(
                        substamp_id=i + n_t,
                        x=c["x"], y=c["y"], stamp_group_id=i + n_t,
                    )
                    for i, c in enumerate(i_stamps_found)
                ]

        if self.config.verbose >= 1:
            print(
                f"Found {len(self.template_substamps)} potential template substamps and "
                f"{len(self.image_substamps)} potential image substamps."
            )

        if self.config.force_convolve == "t":
            if not self.template_substamps:
                raise HotpantsError("No valid template substamps found for kernel fitting.")
        elif self.config.force_convolve == "i":
            if not self.image_substamps:
                raise HotpantsError("No valid image substamps found for kernel fitting.")
        elif not self.template_substamps and not self.image_substamps:
            raise HotpantsError("No valid substamps found for kernel fitting.")

        return self.template_substamps, self.image_substamps

    def fit_and_select_direction(self) -> str:
        """
        Step 2: Performs initial fits and selects the best convolution direction.

        This method performs a localized fit for every potential substamp found
        in the previous step. A figure-of-merit (FOM) is calculated for each
        substamp to assess its quality. Stamps that are saturated, near bad
        pixels, or have a poor local fit (high chi-squared) are rejected.

        The aggregate FOM is then used to decide whether it is better to
        convolve the template to match the science image or vice-versa. The
        status of each `Substamp` is updated to either `PASSED_FOM_CHECK` or
        `REJECTED_FOM_CHECK`.

        Returns:
            The selected convolution direction, either 't' (template) or 'i' (image).
        """
        if not self.template_substamps and not self.image_substamps:
            self.find_stamps()
 
        t_fom, i_fom = float("inf"), float("inf")
        t_fit_results, i_fit_results = [], []
        conv_direction = self.config.force_convolve

        # Create maps for easy lookup of substamp objects by their unique ID
        # Calculate combined error map for weighting (mostly for C extension)
        # In Pure Python, fit_stamps_locally handles its own noise/weighting or ignores it for now.
        
        t_substamp_map = {s.id: s for s in self.template_substamps}
        if self.use_c_extension:
            combined_error_sq = self.results["t_noise_sq"] + self.results["i_noise_sq"]
            
            t_coords = [{"substamp_id": s.id, "stamp_group_id": s.stamp_group_id, "x": s.x, "y": s.y} for s in self.template_substamps]
            # Calls C extension's fit_stamps_and_get_fom - NO OVERSAMPLE ARG
            t_fom, t_fit_results = self.ext.fit_stamps_and_get_fom(self._c_state, self.template_data, self.image_data, combined_error_sq, "t", t_coords)

            # Populate substamp objects with the complete, isolated results from the C extension
            for result in t_fit_results:
                substamp = t_substamp_map.get(result["substamp_id"])
                if substamp:
                    substamp.image_cutout = result["image_cutout"]
                    substamp.template_cutout = result["template_cutout"]
                    substamp.noise_variance_cutout = result["noise_cutout"]
                    substamp.basis_vectors = result["basis_vectors"]
                    substamp.local_kernel_solution = result["local_solution"]
                    substamp.convolved_model_local = result["convolved_model_local"]
                    substamp.fit_results["t"] = {"fom": result["fom"], "chi2": result["chi2"]}
            
            # Direction 'i'
            i_substamp_map = {s.id: s for s in self.image_substamps}
            i_coords = [{"substamp_id": s.id, "stamp_group_id": s.stamp_group_id, "x": s.x, "y": s.y} for s in self.image_substamps]
            # Calls C extension's fit_stamps_and_get_fom - NO OVERSAMPLE ARG
            i_fom, i_fit_results = self.ext.fit_stamps_and_get_fom(self._c_state, self.image_data, self.template_data, combined_error_sq, "i", i_coords)

            # Populate substamp objects with the complete, isolated results from the C extension
            for result in i_fit_results:
                substamp = i_substamp_map.get(result["substamp_id"])
                if substamp:
                    substamp.image_cutout = result["image_cutout"]
                    substamp.template_cutout = result["template_cutout"]
                    substamp.noise_variance_cutout = result["noise_cutout"]
                    substamp.basis_vectors = result["basis_vectors"]
                    substamp.local_kernel_solution = result["local_solution"]
                    substamp.convolved_model_local = result["convolved_model_local"]
                    substamp.fit_results["i"] = {"fom": result["fom"], "chi2": result["chi2"]}
        else:
            # Pure Python Implementation
            # 1. Generate Basis Vectors (Global Config) — cached; OS>1 precomputes LR maps
            import time as _time

            t_fom0 = _time.perf_counter()
            scale = self.oversample
            basis_funcs = self._ensure_kernel_basis(scale)
            basis_lr = None
            if scale > 1:
                # Template→image direction uses HR template
                basis_lr = self._ensure_basis_lr_maps(self.template_data, scale)

            def _fom_by_stamp_group(substamps, conv_img, ref_img, direction, oversample_param):
                """
                Match C check_stamps: one local ksum test per stamp_group (sscnt=0),
                then apply survived_check to every substamp in that group.
                """
                groups = {}
                for s in substamps:
                    groups.setdefault(s.stamp_group_id, []).append(s)
                # Preserve discovery order within each group (first = sscnt 0)
                reps = [groups[gid][0] for gid in sorted(groups.keys())]
                region_map = self.results.get("region_map")
                blr = basis_lr if oversample_param > 1 else None
                fitted = pure.fitting.fit_stamps_locally(
                    reps,
                    conv_img,
                    ref_img,
                    self.config,
                    basis_funcs,
                    oversample=oversample_param,
                    region_map=region_map,
                    input_mask=self.results.get("input_mask_lr"),
                    noise_sq=self.results.get("combined_noise_sq_lr"),
                    basis_lr_maps=blr,
                )
                fit_by_coord = {(int(s.x), int(s.y)): s for s in fitted}
                # Also index by region_id for connected mode (centroid may shift)
                fit_by_region = {
                    int(s.region_id): s
                    for s in fitted
                    if getattr(s, "region_id", None) is not None
                }
                # Stash filled FOM stamps for iterative fit reuse (copy, not shared)
                if region_map is not None:
                    self._fom_stamps_by_region = {
                        rid: s for rid, s in fit_by_region.items() if not s.ignore
                    }
                survivors = []
                n_pass = 0
                for gid in sorted(groups.keys()):
                    rep = groups[gid][0]
                    # Prefer region_id: flux centroids can collide when rounded to int
                    stamp_res = None
                    if getattr(rep, "region_id", None) is not None:
                        stamp_res = fit_by_region.get(int(rep.region_id))
                    if stamp_res is None:
                        stamp_res = fit_by_coord.get((int(rep.x), int(rep.y)))
                    survived = bool(stamp_res is not None and not stamp_res.ignore)
                    if survived:
                        n_pass += 1
                        survivors.append(stamp_res)
                    for substamp in groups[gid]:
                        if stamp_res is not None:
                            substamp.chi2 = float(stamp_res.chi2)
                            substamp.image_cutout = stamp_res.image_cutout
                            substamp.template_cutout = stamp_res.template_cutout
                            substamp.basis_vectors = stamp_res.basis_vectors
                            substamp.local_kernel_solution = stamp_res.local_solution
                            substamp.convolved_model_local = stamp_res.convolved_model_local
                            substamp.fit_results[direction] = {
                                "fom": float(stamp_res.diff),
                                "chi2": float(stamp_res.chi2),
                                "survived_check": survived,
                            }
                        else:
                            substamp.fit_results[direction] = {
                                "fom": float("inf"),
                                "chi2": float("inf"),
                                "survived_check": False,
                            }
                        substamp.status = (
                            SubstampStatus.PASSED_FOM_CHECK
                            if survived
                            else SubstampStatus.REJECTED_FOM_CHECK
                        )
                if survivors:
                    fom = float(np.mean([s.chi2 for s in survivors]))
                else:
                    fom = float("inf")
                if self.config.verbose >= 1:
                    print(
                        f"DEBUG core.py: Direction '{direction}' FOM groups "
                        f"{n_pass}/{len(groups)} passed.",
                        flush=True,
                    )
                return fom

            if conv_direction == "t" or conv_direction == "b":
                t_fom = _fom_by_stamp_group(
                    self.template_substamps,
                    self.template_data,
                    self.image_data,
                    "t",
                    self.oversample,
                )

            # Image→template direction is unsupported when oversample>1 (forced to 't' at init).
            if self.oversample == 1 and (conv_direction == "i" or conv_direction == "b"):
                i_fom = _fom_by_stamp_group(
                    self.image_substamps,
                    self.image_data,
                    self.template_data,
                    "i",
                    1,
                )

            self.results.setdefault("stage_timings", {})["fom"] = _time.perf_counter() - t_fom0

        # Select best direction
        if conv_direction == "b":
            conv_direction = "t" if t_fom < i_fom else "i"
            if self.config.verbose >= 1:
                print(f"Template FOM: {t_fom:.3f}, Image FOM: {i_fom:.3f}. Selecting direction: '{conv_direction}'")

        # Update status based on the winning direction
        if conv_direction == "t":
            if self.use_c_extension:
                for result in t_fit_results:
                    substamp = t_substamp_map.get(result["substamp_id"])
                    if substamp:
                        substamp.status = SubstampStatus.PASSED_FOM_CHECK if result["survived_check"] else SubstampStatus.REJECTED_FOM_CHECK
            else:
                for s in self.template_substamps:
                    fr = s.fit_results.get("t")
                    if fr is not None and fr.get("survived_check", False):
                        s.status = SubstampStatus.PASSED_FOM_CHECK
                    else:
                        s.status = SubstampStatus.REJECTED_FOM_CHECK
        else:  # 'i'
            if self.use_c_extension:
                for result in i_fit_results:
                    substamp = i_substamp_map.get(result["substamp_id"])
                    if substamp:
                        substamp.status = SubstampStatus.PASSED_FOM_CHECK if result["survived_check"] else SubstampStatus.REJECTED_FOM_CHECK
            else:
                for s in self.image_substamps:
                    fr = s.fit_results.get("i")
                    if fr is not None and fr.get("survived_check", False):
                        s.status = SubstampStatus.PASSED_FOM_CHECK
                    else:
                        s.status = SubstampStatus.REJECTED_FOM_CHECK

        self.results["conv_direction"] = conv_direction
        return conv_direction

    def iterative_fit_and_clip(self) -> Tuple[np.ndarray, List[Substamp]]:
        """
        Step 3: Solves for the global kernel using iterative sigma-clipping.

        This method takes all substamps that passed the FOM check and uses them
        to derive a single, spatially varying kernel solution for the entire
        image. It iteratively rejects outlier substamps to achieve a robust fit.
        The status of the substamps is updated to either `USED_IN_FINAL_FIT` or
        `REJECTED_ITERATIVE_FIT`.

        Returns:
            A tuple containing:
            - The global kernel solution as a 1D NumPy array of coefficients.
            - A list of the `Substamp` objects that survived the clipping and
              were used in the final fit.
        """
        if "conv_direction" not in self.results:
            self.fit_and_select_direction()
        if self.config.verbose >= 1:
            print("Starting iterative kernel fit and solution...")

        conv_direction = self.results["conv_direction"]
        candidate_substamps = self.template_substamps if conv_direction == "t" else self.image_substamps
        
        passed_fom = [s for s in candidate_substamps if s.status == SubstampStatus.PASSED_FOM_CHECK]
        if self.config.verbose >= 1:
            print(f"DEBUG: iterative_fit. Direction={conv_direction}. Candidates passing FOM: {len(passed_fom)}/{len(candidate_substamps)}")
            if len(passed_fom) > 0:
                 print(f"DEBUG: Sample Chi2: {getattr(passed_fom[0], 'chi2', 'N/A')}")
        
        # Create a list of stamps for the C function, grouping substamps by group_id

        stamps_for_fit = []
        substamp_map = {}  # Maps group_id to a list of substamp objects
        for s in candidate_substamps:
            if s.status == SubstampStatus.PASSED_FOM_CHECK:
                if s.stamp_group_id not in substamp_map:
                    substamp_map[s.stamp_group_id] = []
                substamp_map[s.stamp_group_id].append(s)

        # This list preserves the order for matching survivor indices later
        candidate_stamps_in_order = []
        for group_id in sorted(substamp_map.keys()):
            substamps_in_group = substamp_map[group_id]
            stamps_for_fit.append({"substamps": [(s.x, s.y) for s in substamps_in_group]})
            candidate_stamps_in_order.append(substamps_in_group[0])  # Representative stamp

        if not stamps_for_fit:
            raise HotpantsError("No substamps passed the initial FOM check.")

        if self.use_c_extension:
            if conv_direction == "t":
                conv_img, ref_img = self.template_data, self.image_data
            else:
                conv_img, ref_img = self.image_data, self.template_data

            kernel_solution, stats, final_survivor_indices = self.ext.fit_kernel(self._c_state, stamps_for_fit, conv_img, ref_img, self.results["t_noise_sq"] + self.results["i_noise_sq"])

            final_fits_substamps = []
            survivor_group_ids = {candidate_stamps_in_order[i].stamp_group_id for i in final_survivor_indices}

            for s in candidate_substamps:
                if s.status == SubstampStatus.PASSED_FOM_CHECK:
                    if s.stamp_group_id in survivor_group_ids:
                        s.status = SubstampStatus.USED_IN_FINAL_FIT
                        final_fits_substamps.append(s)
                    else:
                        s.status = SubstampStatus.REJECTED_ITERATIVE_FIT

            if not final_fits_substamps:
                raise HotpantsError("All stamps were clipped during iterative fitting.")
            if self.config.verbose >= 1:
                print(f"Final fit uses {len(final_survivor_indices)} stamp groups. Fit stats: mean_sig={stats['meansig']:.3f}, scatter={stats['scatter']:.3f}")

            self.results["kernel_solution"] = kernel_solution
            self.results["final_fits"] = final_fits_substamps
            self.results["fit_stats"] = stats
            return kernel_solution, final_fits_substamps

        else:
            # Pure Python — mirror C: stamp groups with sscnt advancement (check_again).
            import time as _time

            t_fit0 = _time.perf_counter()
            oversample_param = self.oversample if conv_direction == 't' else 1

            basis_funcs = self._ensure_kernel_basis(oversample_param)
            basis_lr = None
            if oversample_param > 1 and conv_direction == "t":
                basis_lr = self._ensure_basis_lr_maps(self.template_data, oversample_param)

            # Groups of FOM survivors (ordered like C fit_kernel stamps_for_fit)
            stamp_groups = []
            for group_id in sorted(substamp_map.keys()):
                stamp_groups.append(substamp_map[group_id])

            if not stamp_groups:
                raise HotpantsError("No substamps passed the initial FOM check.")

            # Images to match C fit_kernel(conv, ref): template→image when 't'.
            if conv_direction == "t":
                conv_img, ref_img = self.template_data, self.image_data
            else:
                conv_img, ref_img = self.image_data, self.template_data

            kernel_sol, active_internal_stamps = pure.fitting.fit_kernel(
                stamp_groups,
                conv_img,
                ref_img,
                self.config,
                basis_funcs,
                oversample=oversample_param,
                verbose=self.config.verbose,
                skip_local_reject=True,
                noise_sq=self.results["combined_noise_sq_lr"],
                mask=self.results["input_mask_lr"],
                region_map=self.results.get("region_map"),
                basis_lr_maps=basis_lr,
                prefilled_by_region=self._fom_stamps_by_region
                if self.results.get("region_map") is not None
                else None,
            )
            self.results.setdefault("stage_timings", {})["iterative_fit"] = (
                _time.perf_counter() - t_fit0
            )

            if kernel_sol is None:
                raise HotpantsError("Kernel fit failed (singular matrix or other error).")

            self.results["kernel_solution"] = kernel_sol
            if active_internal_stamps:
                self.results["fit_stats"] = getattr(active_internal_stamps[0], "fit_stats", None)

            # Survivors: groups that still have a valid active substamp
            survivor_group_ids = set()
            active_by_orig = {}
            for s in active_internal_stamps:
                if not getattr(s, "ignore", False) and s.sscnt < s.nss:
                    survivor_group_ids.add(
                        stamp_groups[s.orig_idx][0].stamp_group_id
                        if s.orig_idx is not None and s.orig_idx < len(stamp_groups)
                        else None
                    )
                    active_by_orig[s.orig_idx] = s
            survivor_group_ids.discard(None)

            if self.config.verbose >= 1:
                print(
                    f"DEBUG: Mapping {len(active_by_orig)} internal stamps back to "
                    f"{len(candidate_substamps)} candidates; "
                    f"{len(survivor_group_ids)} groups survived."
                )

            valid_list = []
            for gidx, group in enumerate(stamp_groups):
                internal = active_by_orig.get(gidx)
                if internal is None or getattr(internal, "ignore", False):
                    for s in group:
                        if s.status == SubstampStatus.PASSED_FOM_CHECK:
                            s.status = SubstampStatus.REJECTED_ITERATIVE_FIT
                    continue
                # Mark the active substamp as used; others from the group stay PASSED or get rejected
                used = None
                if 0 <= internal.sscnt < len(group):
                    used = group[internal.sscnt]
                else:
                    # Fallback: match by coordinates
                    for s in group:
                        if int(s.x) == int(internal.x) and int(s.y) == int(internal.y):
                            used = s
                            break
                    if used is None and group:
                        used = group[0]
                for s in group:
                    if s is used:
                        s.status = SubstampStatus.USED_IN_FINAL_FIT
                        s.convolved_model_global = getattr(internal, "convolved_model_global", None)
                        s.convolved_model_local = getattr(internal, "convolved_model_local", None)
                        s.local_kernel_solution = getattr(internal, "local_solution", None)
                        valid_list.append(s)
                    elif s.status == SubstampStatus.PASSED_FOM_CHECK:
                        s.status = SubstampStatus.REJECTED_ITERATIVE_FIT

            if self.config.verbose >= 1:
                print(f"Iterative fit selected {len(valid_list)} substamps for final kernel solution.")

            return kernel_sol, valid_list

    def convolve_and_difference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 4: Applies the kernel, performs subtraction, and creates final images.

        This method uses the global kernel solution to convolve the appropriate
        image. It then subtracts the convolved image from the target image to
        produce the difference image and calculates the corresponding final
        noise image and output mask.

        Returns:
            A tuple containing:
            - diff_image (np.ndarray): The raw difference image (Target - Convolved Model).
            - convolved_image (np.ndarray): The image that was convolved to match the other.
            - noise_image (np.ndarray): The final 1-sigma noise map for the difference image.
            - output_mask (np.ndarray): The final integer mask indicating bad pixels.
        """
        if "kernel_solution" not in self.results:
            self.iterative_fit_and_clip()

        conv_direction = self.results["conv_direction"]

        t_noise_sq = self.results["t_noise_sq"]
        i_noise_sq = self.results["i_noise_sq"]

        if conv_direction == "t":
            image_to_convolve = self.template_data
            target_image = self.image_data
            noise_to_convolve_sq = t_noise_sq
            target_noise_sq = i_noise_sq
            # HR mask when convolving oversampled template; else native input_mask.
            mask = self.results["input_mask"]
        else:
            image_to_convolve = self.image_data
            target_image = self.template_data
            noise_to_convolve_sq = i_noise_sq
            target_noise_sq = t_noise_sq
            mask = self.results.get("input_mask_lr", self.results["input_mask"])

        if mask is None:
            mask = np.zeros(image_to_convolve.shape, dtype=np.int32)

        if self.use_c_extension:
            convolved_image, output_mask, conv_noise_sq = self.ext.apply_kernel(self._c_state, image_to_convolve, self.results["kernel_solution"], noise_to_convolve_sq)
            bkg = self.ext.get_background_image(self._c_state, self.results["kernel_solution"])
        else:
            # Pure Python Convolution — reuse cached basis
            import time as _time

            t_app0 = _time.perf_counter()
            oversample_param = self.oversample if self.results["conv_direction"] == 't' else 1
            basis_funcs = self._ensure_kernel_basis(oversample_param)
            basis_vecs = np.array(basis_funcs)

            convolved_image, bkg, conv_noise_sq, output_mask_conv = pure.convolution.apply_kernel(
                image_to_convolve, self.results["kernel_solution"],
                noise_to_convolve_sq, mask,
                self.config, basis_vecs,
                oversample=oversample_param
            )

            output_mask = output_mask_conv
            self.results.setdefault("stage_timings", {})["apply_kernel"] = (
                _time.perf_counter() - t_app0
            )

        convolved_image += bkg

        diff_image = target_image - convolved_image
        final_noise = np.sqrt(conv_noise_sq + target_noise_sq)

        if self.config.rescale_ok:
            if self.config.verbose >= 1:
                print("Rescaling noise for OK pixels...")
            if self.use_c_extension:
                final_noise = self.ext.rescale_noise_ok(self._c_state, diff_image, final_noise, output_mask)
            else:
                warnings.warn("rescale_ok is currently only implemented for the C extension path.")

        self.results.update({"convolved_image": convolved_image, "background": bkg, "output_mask": output_mask, "diff_image": diff_image, "noise_image": final_noise})

        self._populate_global_convolved_models()

        return diff_image, convolved_image, final_noise, output_mask

    def get_final_outputs(self) -> Dict[str, Any]:
        """
        Step 5: Applies final masks, calculates statistics, and returns all products.

        This is the final data-producing step. It applies fill values to masked
        pixels in the output images and calculates final image statistics.

        Returns:
            A dictionary containing all final data products, including:
            - 'diff_image': The final, masked difference image.
            - 'convolved_image': The final, masked convolved image.
            - 'noise_image': The final, masked noise image.
            - 'output_mask': The final integer mask.
            - 'stats': A dictionary of final image statistics.
            - 'conv_direction': The convolution direction ('t' or 'i').
            - 'kernel_solution': The global kernel solution coefficients.
        """
        if "diff_image" not in self.results:
            self.convolve_and_difference()
        if self.config.verbose >= 1:
            print("Applying final masks to outputs and calculating statistics...")

        final_diff, final_conv, final_bkg, final_noise, output_mask = (
            self.results["diff_image"].copy(),
            self.results["convolved_image"].copy(),
            self.results["background"].copy(),
            self.results["noise_image"].copy(),
            self.results["output_mask"].copy(),
        )
        # Match development C behavior: do not overwrite masked pixels with fill values
        # before computing/returning final products.

        if self.use_c_extension:
            self.results["stats"] = self.ext.calculate_final_stats(self._c_state, final_diff, final_noise, output_mask)
        else:
            # Pure Python Stats Calculation — match getNoiseStats3 / getStampStats3
            # good pixels: umask=0, smask=0xffff → skip any flagged mask bit; skip |diff|<=ZEROVAL
            ZEROVAL = 1e-20
            m = output_mask.astype(np.int32, copy=False)
            good = (m == 0) & np.isfinite(final_diff) & (np.abs(final_diff) > ZEROVAL) & (final_noise > 0)

            if np.any(good):
                diff_vals = final_diff[good]
                noise_vals = final_noise[good]

                stats = {}
                stats["diff_mean"] = float(np.mean(diff_vals))
                # Sample stdev like getStampStats3 / sigma_clip path
                if diff_vals.size > 1:
                    stats["diff_std"] = float(np.sqrt(np.sum((diff_vals - stats["diff_mean"]) ** 2) / (diff_vals.size - 1)))
                else:
                    stats["diff_std"] = 0.0
                stats["noise_mean"] = float(np.mean(noise_vals))
                stats["nx2norm"] = int(diff_vals.size)

                # getNoiseStats3: mean of (diff/noise)^2
                chi2 = np.sum((diff_vals / noise_vals) ** 2)
                stats["x2norm"] = float(chi2 / stats["nx2norm"])

                self.results["stats"] = stats
            else:
                self.results["stats"] = {
                    "diff_mean": 0.0, "diff_std": 0.0, "noise_mean": 0.0,
                    "nx2norm": 0, "x2norm": 0.0
                }

        return {
            "diff_image": final_diff,
            "convolved_image": final_conv,
            "background": final_bkg,
            "noise_image": final_noise,
            "output_mask": output_mask,
            "stats": self.results["stats"],
            "conv_direction": self.results["conv_direction"],
            "kernel_solution": self.results["kernel_solution"],
            "fit_stats": self.results.get("fit_stats"),
        }

    def save_outputs(self):
        """
        Saves all configured output files (FITS images and region files).

        This method checks the `HotpantsConfig` object for any specified output
        filenames and writes the corresponding files. This includes the main
        difference image, noise image, mask, and the diagnostic DS9 region file
        for the kernel fitting stamps.
        """
        if "diff_image" not in self.results:
            # This ensures all necessary data products are computed
            self.get_final_outputs()

        header_to_use = self.output_header
        # Check if any FITS saving is requested
        if any([self.config.output_file, self.config.noise_image_file, self.config.mask_image_file, self.config.convolved_image_file, self.config.sigma_image_file]):
            if header_to_use is None:
                warnings.warn("No FITS header available. Creating a minimal header. WCS and other metadata will be missing.")
                header_to_use = fits.Header()

        if self.config.stamp_region_file:
            self._save_stamp_region_file(self.config.stamp_region_file)

        # Save FITS files if configured
        if self.config.output_file:
            self._save_fits_image(self.config.output_file, self.results["diff_image"], header_to_use, "difference")
        if self.config.noise_image_file:
            self._save_fits_image(self.config.noise_image_file, self.results["noise_image"], header_to_use, "noise")
        if self.config.mask_image_file:
            self._save_fits_image(self.config.mask_image_file, self.results["output_mask"], header_to_use, "mask")
        if self.config.convolved_image_file:
            self._save_fits_image(self.config.convolved_image_file, self.results["convolved_image"], header_to_use, "convolved")
        if self.config.sigma_image_file:
            # Calculate sigma image on the fly
            sigma_image = np.divide(self.results["diff_image"], self.results["noise_image"], out=np.full_like(self.results["diff_image"], self.config.fillval), where=self.results["noise_image"] != 0)
            self._save_fits_image(self.config.sigma_image_file, sigma_image, header_to_use, "sigma")

    def _save_stamp_region_file(self, filename: str):
        """Saves a DS9 region file showing used and rejected stamps."""
        conv_direction = self.results.get("conv_direction")
        if not conv_direction:
            return

        stamps_to_plot = self.template_substamps if conv_direction == "t" else self.image_substamps
        box_size = self.config.fwksstamp

        with open(filename, "w") as f:
            f.write("# DS9 region file format\n")
            f.write("global color=green width=2\n")
            f.write("image\n")

            for s in stamps_to_plot:
                color = None
                if s.status == SubstampStatus.USED_IN_FINAL_FIT:
                    color = "green"
                elif s.status == SubstampStatus.REJECTED_ITERATIVE_FIT:
                    color = "red"
                elif s.status == SubstampStatus.REJECTED_FOM_CHECK:
                    color = "yellow"

                if color:
                    # DS9 uses 1-based coordinates
                    f.write(f"box({s.x + 1},{s.y + 1},{box_size},{box_size},0) # color={color}\n")
        if self.config.verbose >= 1:
            print(f"Saved stamp region file to {filename}")

    def _save_fits_image(self, filename: str, data: np.ndarray, header: fits.Header, image_type: str):
        """Internal helper to save a FITS image with replicated headers."""
        # Create a copy to avoid modifying the original header object in memory
        hdr = header.copy()

        # Add HOTPANTS specific headers, replicating main.c
        hdr.add_blank("", before=0)
        hdr.set("SOFTNAME", "HOTPanTS", "The software that differenced this image", after=0)
        hdr.set("SOFTVERS", __version__, "Version", after="SOFTNAME")
        hdr.set("SOFTAUTH", "A. Becker / A. Rest", "Author", after="SOFTVERS")
        try:
            hdr.set("AUTHOR", getpass.getuser(), "Who ran the software", after="SOFTAUTH")
        except Exception:
            hdr.set("AUTHOR", "unknown", "Who ran the software", after="SOFTAUTH")
        try:
            hdr.set("ORIGIN", socket.gethostname(), "Where it was done", after="AUTHOR")
        except Exception:
            hdr.set("ORIGIN", "unknown", "Where it was done", after="AUTHOR")
        hdr.set("DATE", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"), "When it was started (GMT)", after="ORIGIN")
        hdr.add_blank("", after="DATE")

        stats = self.results.get("stats", {})
        fit_stats = self.results.get("fit_stats", {})

        hdr.set("CONVOL00", self.results.get("conv_direction", "N/A").upper(), "Direction of convolution")
        # Calculate kernel sum at center of image
        kernel_center = self.visualize_kernel(at_coords=(self.nx_lr // 2, self.ny_lr // 2), size_factor=1.0)
        hdr.set("KSUM00", float(np.sum(kernel_center)), "Kernel Sum at image center")
        hdr.set("SSSIG00", fit_stats.get("meansig", -1.0), "Average Figure of Merit across Stamps")
        hdr.set("SSSCAT00", fit_stats.get("scatter", -1.0), "Stdev in Figure of Merit")
        hdr.set("X2NRM00", stats.get("x2norm", -1.0), "1/N * SUM (diff/noise)^2")
        hdr.set("NX2NRM00", stats.get("nx2norm", -1), "Number of pixels in X2NRM")
        hdr.set("DMEAN00", stats.get("diff_mean", -1.0), "Mean of diff image; good pixels")
        hdr.set("DSIGE00", stats.get("diff_std", -1.0), "Stdev of diff image; good pixels")
        hdr.set("DSIG00", stats.get("noise_mean", -1.0), "Mean of noise image; good pixels")

        if image_type == "mask":
            # Save mask as 16-bit integer with BZERO/BSCALE for compatibility with C output
            hdu = fits.PrimaryHDU(data=data.astype(np.int16), header=hdr)
            hdu.header["BITPIX"] = 16
            hdu.scale("int16", bzero=32768)
        else:
            hdu = fits.PrimaryHDU(data=data, header=hdr)

        hdu.writeto(filename, overwrite=True)
        if self.config.verbose >= 1:
            print(f"Saved {image_type} image to {filename}")

    def run_pipeline(self) -> Dict[str, Any]:
        """
        A convenience method to run the entire pipeline in a single call.

        This executes all steps from stamp finding to final output generation
        and saves any configured output files.

        Returns:
            A dictionary containing all final data products, as returned by
            `get_final_outputs`. This includes the final difference image,
            noise map, mask, and statistics.
        """
        import time as _time

        timings = self.results.setdefault("stage_timings", {})
        t0 = _time.perf_counter()
        self.find_stamps()
        timings["find_stamps"] = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        self.fit_and_select_direction()
        timings["fit_and_select_direction"] = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        self.iterative_fit_and_clip()
        timings["iterative_fit_and_clip"] = _time.perf_counter() - t0
        t0 = _time.perf_counter()
        self.convolve_and_difference()
        timings["convolve_and_difference"] = _time.perf_counter() - t0
        outputs = self.get_final_outputs()
        self.save_outputs()
        if self.config.verbose >= 1 and timings:
            parts = ", ".join(f"{k}={v:.2f}s" for k, v in sorted(timings.items()))
            print(f"Stage timings: {parts}", flush=True)
        return outputs

    def visualize_kernel(self, at_coords: Tuple[int, int], size_factor: float = 2.0) -> np.ndarray:
        """
        Generates an image of the convolution kernel at a specific coordinate.

        This method should be called *after* the pipeline has run and a
        kernel solution has been found. It uses the final kernel solution to
        reconstruct the kernel for the given (x, y) location.

        Args:
            at_coords: The (x, y) coordinates at which to visualize the kernel.
            size_factor: A multiplier for the kernel's width to
                determine the output image size. Defaults to 2.0.

        Returns:
            A 2D NumPy array containing the image of the kernel.

        Raises:
            HotpantsError: If the kernel fitting has not been run yet.
            TypeError: If at_coords is not a tuple of two integers.
            ValueError: If size_factor is not a positive number.
        """
        if "kernel_solution" not in self.results:
            raise HotpantsError("Kernel solution not found. The fitting pipeline must be run before a kernel can be visualized.")

        if not (isinstance(at_coords, tuple) and len(at_coords) == 2 and all(isinstance(i, int) for i in at_coords)):
            raise TypeError("at_coords must be a tuple of two integers (x, y).")

        if not isinstance(size_factor, (int, float)) or size_factor <= 0:
            raise ValueError("size_factor must be a positive number.")

        if self.use_c_extension:
            kernel_image = self.ext.visualize_kernel(self._c_state, at_coords, self.results["kernel_solution"], size_factor)
        else:
            # Pure Python: scale kernel to HR when oversample>1 (template convolution).
            scale = self.oversample if self.results.get("conv_direction", "t") == "t" else 1
            basis_funcs = self._ensure_kernel_basis(scale)
            hr_rkernel = self.config.rkernel * scale
            k_size = 2 * hr_rkernel + 1
            if not self.config.deg_fixe:
                self.config.deg_fixe = [self.config.ko] * self.config.ngauss
            kernel_vecs = np.array(basis_funcs)

            x, y = at_coords  # LR science coordinates
            kernel_sol = self.results["kernel_solution"]

            step = int(getattr(self.config, "kc_step", 2 * self.config.rkernel + 1) or (2 * self.config.rkernel + 1))
            ker_order = self.config.ko
            n_comp_ker = len(basis_funcs)

            # Spatial polys use LR frame (matches fit_kernel / apply_kernel).
            local_kernel = pure.convolution.jit_make_kernel(
                kernel_sol, step * scale, hr_rkernel,
                float(self.nx_lr), float(self.ny_lr),
                n_comp_ker, ker_order, kernel_vecs,
                float(x), float(y),
            )

            # Output image sizing
            out_size = int(k_size * size_factor)
            output = np.zeros((out_size, out_size))
            
            # Center the kernel
            y_off = (out_size - k_size) // 2
            x_off = (out_size - k_size) // 2
            
            # Handle clipping if output is smaller than kernel (unlikely but possible)
            y_start_k = 0
            y_end_k = k_size
            x_start_k = 0
            x_end_k = k_size
            
            if y_off < 0:
                y_start_k = -y_off
                y_off = 0
            if x_off < 0:
                x_start_k = -x_off
                x_off = 0
                
            h_paste = min(out_size - y_off, y_end_k - y_start_k)
            w_paste = min(out_size - x_off, x_end_k - x_start_k)
            
            if h_paste > 0 and w_paste > 0:
                output[y_off:y_off+h_paste, x_off:x_off+w_paste] = \
                    local_kernel[y_start_k:y_start_k+h_paste, x_start_k:x_start_k+w_paste]
            
            kernel_image = output
        return kernel_image

    def get_substamp_details(self) -> Dict[str, Any]:
        """
        Returns the complete, stateful master lists of all substamps.

        This is a diagnostic method to inspect the properties and final status
        of every substamp considered during the pipeline run.

        Returns:
            A dictionary containing the full lists of `template_substamps` and
            `image_substamps`.
        """
        if "conv_direction" not in self.results:
            raise HotpantsError("Pipeline must be run (at least to iterative_fit_and_clip) before getting substamp details.")

        final_fit_locations = [{"id": s.id, "x": s.x, "y": s.y} for s in self.template_substamps + self.image_substamps if s.status == SubstampStatus.USED_IN_FINAL_FIT]

        return {"template_substamps": self.template_substamps, "image_substamps": self.image_substamps, "final_fit_locations": {"convolution_direction": self.results["conv_direction"], "locations": final_fit_locations}}

    def _populate_global_convolved_models(self):
        """
        Internal helper to extract cutouts from the final globally convolved image
        and store them in the appropriate substamp objects. This is run after the
        main convolution step.
        """
        if "convolved_image" not in self.results:
            return

        convolved_image = self.results["convolved_image"]
        conv_dir = self.results["conv_direction"]

        substamps_to_process = []
        if conv_dir == "t":
            substamps_to_process = self.template_substamps
        elif conv_dir == "i":
            substamps_to_process = self.image_substamps

        if not substamps_to_process:
            return

        hw = self.config.hwksstamp
        fill_value = self.config.fillval

        for substamp in substamps_to_process:
            x_center = substamp.x
            y_center = substamp.y

            cutout = pyhotpants.cut_substamp_from_image(image=convolved_image, x_center=x_center, y_center=y_center, half_width=hw, fill_value=fill_value)
            substamp.convolved_model_global = cutout
