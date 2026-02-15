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


__version__ = "0.2.0"


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
        self.ny, self.nx = self.template_data.shape

        self.config = config if config is not None else HotpantsConfig(nx=self.nx, ny=self.ny)
        self._t_error_input = np.ascontiguousarray(t_error, dtype=np.float32) if t_error is not None else None
        self._i_error_input = np.ascontiguousarray(i_error, dtype=np.float32) if i_error is not None else None
        
        # Sanitize Inputs (NaN Handling)
        self._t_nan_mask = np.isnan(self.template_data)
        if np.any(self._t_nan_mask):
            fill = self.config.fillval if hasattr(self.config, 'fillval') else 1e-30
            self.template_data[self._t_nan_mask] = fill
            
        self._i_nan_mask = np.isnan(self.image_data)
        if np.any(self._i_nan_mask):
            fill = self.config.fillval if hasattr(self.config, 'fillval') else 1e-30
            self.image_data[self._i_nan_mask] = fill

        if star_catalog is not None:
            if not isinstance(star_catalog, np.ndarray) or star_catalog.ndim != 2 or star_catalog.shape[1] != 2:
                raise HotpantsError("star_catalog must be a 2D NumPy array with shape (N, 2).")
            self.star_catalog = np.ascontiguousarray(star_catalog, dtype=np.float32) - 1
        else:
            self.star_catalog = None

        self.results = {}
        # New master lists for substamp objects
        self.template_substamps: List[Substamp] = []
        self.image_substamps: List[Substamp] = []

        # Dynamically set thresholds if not provided
        if self.config.tuthresh is None:
            self.config.tuthresh = np.max(self.template_data)
        if self.config.tuktresh is None:
            self.config.tuktresh = self.config.tuthresh
        if self.config.tlthresh is None:
            self.config.tlthresh = np.min(self.template_data)
        if self.config.iuthresh is None:
            self.config.iuthresh = np.max(self.image_data)
        if self.config.iuktresh is None:
            self.config.iuktresh = self.config.iuthresh
        if self.config.ilthresh is None:
            self.config.ilthresh = np.min(self.image_data)

        # Initialize C state object and pre-compute masks and noise images
        if self.use_c_extension:
            self._c_state = self.ext.HotpantsState(self.nx, self.ny, self.config.to_dict())
            if self.config.verbose >= 1:
                print(f"Initialized HOTPANTS state: {self._c_state}")
        else:
            self._c_state = None
            if self.config.verbose >= 1:
                print(f"Initialized HOTPANTS (Pure Python Mode)")

        # 1. Create the initial input mask from the images and C extension.
        self._t_mask_input = np.ascontiguousarray(t_mask, dtype=np.int32) if t_mask is not None else None
        self._i_mask_input = np.ascontiguousarray(i_mask, dtype=np.int32) if i_mask is not None else None

        self._t_mask_input = np.ascontiguousarray(t_mask, dtype=np.int32) if t_mask is not None else None
        self._i_mask_input = np.ascontiguousarray(i_mask, dtype=np.int32) if i_mask is not None else None
        
        if self.use_c_extension:
            input_mask = self.ext.make_input_mask(self._c_state, self.template_data, self.image_data, self._t_mask_input, self._i_mask_input)
        else:
            # Pure Python masking
            t_bad = pure.utils.mask_pixels(self.template_data, self.config.tlthresh, self.config.tuthresh)
            i_bad = pure.utils.mask_pixels(self.image_data, self.config.ilthresh, self.config.iuthresh)
            
            if self.oversample > 1:
                # Upsample i_bad to match template resolution
                # i_bad is (ny, nx), t_bad is (ny*F, nx*F)
                # use np.repeat or Kronecker product
                # We want to flag all HR pixels corresponding to a bad LR pixel
                i_bad_up = i_bad.repeat(self.oversample, axis=0).repeat(self.oversample, axis=1)
                input_mask = t_bad | i_bad_up
            else:
                input_mask = t_bad | i_bad
                
            if self._t_mask_input is not None: 
                input_mask |= (self._t_mask_input > 0)
            if self._i_mask_input is not None: 
                if self.oversample > 1:
                     input_mask |= (self._i_mask_input.repeat(self.oversample, axis=0).repeat(self.oversample, axis=1) > 0)
                else:
                     input_mask |= (self._i_mask_input > 0)
            
            # Add NaN masks
            if hasattr(self, '_t_nan_mask') and np.any(self._t_nan_mask): input_mask |= self._t_nan_mask
            if hasattr(self, '_i_nan_mask') and np.any(self._i_nan_mask): 
                 if self.oversample > 1:
                     input_mask |= self._i_nan_mask.repeat(self.oversample, axis=0).repeat(self.oversample, axis=1)
                 else:
                     input_mask |= self._i_nan_mask 


        if self.config.verbose >= 1:
            print(f"Input mask created with shape: {input_mask.shape}, dtype: {input_mask.dtype}")
        self.results["input_mask"] = input_mask

        # 2. Generate noise images using C extension if not provided by the user.
        if self._t_error_input is not None:
            # Use user-provided noise image, squared.
            t_noise_sq = np.ascontiguousarray(self._t_error_input**2, dtype=np.float32)
        else:
            # Generate noise image from scratch and square it.
            if self.use_c_extension:
                t_noise_sq = self.ext.calculate_noise_image(self._c_state, self.template_data, True)
            else:
                sig, mode = pure.utils.calculate_noise(self.template_data, input_mask)
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

    def find_stamps(self) -> Tuple[List[Substamp], List[Substamp]]:
        """
        Step 1: Finds potential substamp coordinates for kernel fitting.

        This method scans the template and science images for suitable stars
        to use for constructing the convolution kernel. If a `star_catalog` was
        provided during initialization, this catalog is used directly,
        bypassing the automated search. Otherwise, a grid-based search is
        performed to find bright, isolated stars.

        It populates the `template_substamps` and `image_substamps` lists with
        `Substamp` objects, which initially contain only coordinate information.

        Returns:
            A tuple containing two lists: the `Substamp` objects found on the
            template and the `Substamp` objects found on the science image.
        """
        if self.use_c_extension:
            t_substamps_coords, i_substamps_coords = self.ext.find_stamps(self._c_state, self.template_data, self.image_data, self.config.fitthresh, self.star_catalog)

            self.template_substamps = [Substamp(**coords) for coords in t_substamps_coords]
            self.image_substamps = [Substamp(**coords) for coords in i_substamps_coords]
        else:
            # Pure Python implementation
            if self.config.verbose >= 1:
                print(f"DEBUG core.py: pure.utils={pure.utils}", flush=True)
                print(f"DEBUG core.py: pure.utils.find_stamps={pure.utils.find_stamps}", flush=True)
            
            # 1. Search in Template
            if self.star_catalog is not None:
                t_stamps_found = [{'x': float(x), 'y': float(y)} for x, y in self.star_catalog]
            else:
                if self.oversample > 1:
                     t_bad = pure.utils.mask_pixels(self.template_data, self.config.tlthresh, self.config.tuthresh)
                     t_mask_for_search = t_bad
                else:
                     t_mask_for_search = self.results["input_mask"]
    
                t_stamps_found = pure.utils.find_stamps(
                    self.template_data, t_mask_for_search, 
                    self.config.nstampx * self.config.nstampy, self.config.rss * 2 + 1
                )
            
            self.template_substamps = [
                Substamp(substamp_id=i, x=c['x'], y=c['y'], stamp_group_id=0) 
                for i, c in enumerate(t_stamps_found)
            ]
            
            # 2. Search in Image
            if self.star_catalog is not None:
                i_stamps_found = [{'x': float(x), 'y': float(y)} for x, y in self.star_catalog]
            else:
                if self.oversample > 1:
                     i_bad = pure.utils.mask_pixels(self.image_data, self.config.ilthresh, self.config.iuthresh)
                     i_mask_for_search = i_bad
                else:
                     i_mask_for_search = self.results["input_mask"]
    
                i_stamps_found = pure.utils.find_stamps(
                    self.image_data, i_mask_for_search, 
                    self.config.nstampx * self.config.nstampy, self.config.rss * 2 + 1
                )
            
            self.image_substamps = [
                Substamp(substamp_id=i+len(self.template_substamps), x=c['x'], y=c['y'], stamp_group_id=1) 
                for i, c in enumerate(i_stamps_found)
            ]

        if self.config.verbose >= 1:
            print(f"Found {len(self.template_substamps)} potential template substamps and {len(self.image_substamps)} potential image substamps.")

        if not self.template_substamps and not self.image_substamps:
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
            
        if self.config.verbose >= 1:
            print(f"DEBUG: fit_and_select_direction. Stamps: T={len(self.template_substamps)}, I={len(self.image_substamps)}")

        t_fom, i_fom = float("inf"), float("inf")
        t_fit_results, i_fit_results = [], []

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
            # 1. Generate Basis Vectors (Global Config)
            # Scale for HR Template if needed
            scale = self.oversample
            hr_rkernel = self.config.rkernel * scale
            shape_hr = (2 * hr_rkernel + 1, 2 * hr_rkernel + 1)
            scaled_sigmas = [s * scale for s in self.config.sigma_gauss]
            
            basis_funcs = pure.kernel.calculate_kernel_basis(
                shape_hr, 
                scaled_sigmas, 
                self.config.deg_fixe
            )
            
            # 2. Fit Template Stamps -> Match Template(Ref) to Image(Data)
            # Direction 't': We convolve Template. Template is Reference. Image is Data.
            t_valid = pure.fitting.fit_stamps_locally(self.template_substamps, self.template_data, self.image_data, self.config, basis_funcs, oversample=self.oversample)
            
            # Calculate FOM
            if t_valid:
                t_fom = np.mean([s.chi2 for s in t_valid])
                
                # Update status and map results back
                # Create lookup by (int(x), int(y)) to be safe? 
                # pure.fitting uses passed coordinates.
                t_map = {(int(s.x), int(s.y)): s for s in self.template_substamps}
                
                t_updated_count = 0
                for stamp_res in t_valid:
                    # Find matching substamp
                    key = (int(stamp_res.x), int(stamp_res.y))
                    if key in t_map:
                        substamp = t_map[key]
                        substamp.status = SubstampStatus.PASSED_FOM_CHECK
                        substamp.chi2 = float(stamp_res.chi2)
                        substamp.image_cutout = stamp_res.image_cutout
                        substamp.template_cutout = stamp_res.template_cutout
                        substamp.basis_vectors = stamp_res.basis_vectors
                        substamp.local_kernel_solution = stamp_res.local_solution
                        substamp.convolved_model_local = stamp_res.convolved_model_local
                        substamp.fit_results["t"] = {"fom": float(stamp_res.chi2), "chi2": float(stamp_res.chi2)}
                        t_updated_count += 1
                if self.config.verbose >= 1:
                    print(f"DEBUG core.py: Updated status for {t_updated_count}/{len(t_valid)} template stamps.", flush=True)
                    
            else:
                t_fom = float('inf')

            # 3. Fit Image Stamps -> Match Image(Ref) to Template(Data)
            # Direction 'i': We convolve Image. Image is Reference. Template is Data.
            i_valid = pure.fitting.fit_stamps_locally(self.image_substamps, self.image_data, self.template_data, self.config, basis_funcs)
            
            if i_valid:
                i_fom = np.mean([s.chi2 for s in i_valid])
                
                i_map = {(int(s.x), int(s.y)): s for s in self.image_substamps}
                
                i_updated_count = 0
                for stamp_res in i_valid:
                    key = (int(stamp_res.x), int(stamp_res.y))
                    if key in i_map:
                        substamp = i_map[key]
                        substamp.status = SubstampStatus.PASSED_FOM_CHECK
                        substamp.chi2 = float(stamp_res.chi2)
                        substamp.image_cutout = stamp_res.image_cutout
                        substamp.template_cutout = stamp_res.template_cutout
                        substamp.basis_vectors = stamp_res.basis_vectors
                        substamp.local_kernel_solution = stamp_res.local_solution
                        substamp.convolved_model_local = stamp_res.convolved_model_local
                        substamp.fit_results["i"] = {"fom": float(stamp_res.chi2), "chi2": float(stamp_res.chi2)}
                        i_updated_count += 1
                if self.config.verbose >= 1:
                    print(f"DEBUG core.py: Updated status for {i_updated_count}/{len(i_valid)} image stamps.", flush=True)
                        
            else:
                i_fom = float('inf')


        # Select best direction
        conv_direction = self.config.force_convolve
        if conv_direction == "b":
            conv_direction = "t" if t_fom < i_fom else "i"
            if self.config.verbose >= 1:
                print(f"Template FOM: {t_fom:.3f}, Image FOM: {i_fom:.3f}. Selecting direction: '{conv_direction}'")

        # Update status based on the winning direction
        # Update status based on the winning direction
        if conv_direction == "t":
            # For Pure Python, we need to mark them passed if they have fit results
            if self.use_c_extension:
                for result in t_fit_results:
                    substamp = t_substamp_map.get(result["substamp_id"])
                    if substamp:
                        substamp.status = SubstampStatus.PASSED_FOM_CHECK if result["survived_check"] else SubstampStatus.REJECTED_FOM_CHECK
            else:
                for s in self.template_substamps:
                    if "t" in s.fit_results and s.fit_results["t"]["fom"] < float('inf'):
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
                    if "i" in s.fit_results and s.fit_results["i"]["fom"] < float('inf'):
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
            # Pure Python
            # 1. Generate Basis
            # Scale if Convolution Direction is Template AND oversample > 1
            oversample_param = self.oversample if conv_direction == 't' else 1
            
            scale = oversample_param
            hr_rkernel = self.config.rkernel * scale
            k_size = 2 * hr_rkernel + 1
            scaled_sigmas = [s * scale for s in self.config.sigma_gauss]

            basis_funcs = pure.kernel.calculate_kernel_basis((k_size, k_size), scaled_sigmas, self.config.deg_fixe)
            
            # 2. Run Fit
            # fit_kernel handles downsampling internally if oversample > 1 is passed
            # AND if we pass the HR basis vectors.
            
            kernel_sol, active_internal_stamps = pure.fitting.fit_kernel(
                candidate_substamps, 
                self.template_data, 
                self.image_data, 
                self.config, 
                basis_funcs,
                oversample=oversample_param
            )
            
            if kernel_sol is None:
                raise HotpantsError("Kernel fit failed (singular matrix or other error).")

            self.results["kernel_solution"] = kernel_sol
            
            # Filter Valid/Rejected based on active_internal_stamps indices
            # And copy back local model data for visualization
            active_map = {s.orig_idx: s for s in active_internal_stamps}
            
            # Always print for debugging
            print(f"DEBUG: Mapping {len(active_map)} internal stamps back to {len(candidate_substamps)} candidates.")
            if len(active_map) > 0:
                 print(f"DEBUG: Sample active keys: {list(active_map.keys())[:10]}")
            
            valid_list = []
            for i, s in enumerate(candidate_substamps):
                if i in active_map:
                    internal_s = active_map[i]
                    s.status = SubstampStatus.USED_IN_FINAL_FIT
                    
                    # Copy attributes for plotting
                    s.convolved_model_global = getattr(internal_s, 'convolved_model_global', None)
                    s.convolved_model_local = getattr(internal_s, 'convolved_model_local', None)
                    s.local_kernel_solution = getattr(internal_s, 'local_solution', None)
                    # s.image_cutout = internal_s.image_cutout # Should verify if modified? Usually same source.
                    
                    valid_list.append(s)
                else:
                    s.status = SubstampStatus.REJECTED_ITERATIVE_FIT
            
            print(f"DEBUG: valid_list length: {len(valid_list)}")
            
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

        mask = self.results.get("input_mask", None) # Default to None or zeros if missing
        if mask is None:
             # Shape depends on which image is convolved, but typically masks are initialized.
             # If completely missing, we need shape. 
             # Let's derive shape from image_data/template_data later? 
             # But here mask is used for convolved image.
             # If convolving Template (HR), mask might need to be HR?
             # But generally input_mask matches Science Image.
             # If HR Template is convolved, we might need HR mask.
             # However, current logic just grabs 'input_mask' from results which is usually the science mask.
             # Let's initialize safely.
             pass 

        if conv_direction == "t":
            image_to_convolve = self.template_data
            target_image = self.image_data
            noise_to_convolve_sq = t_noise_sq
            target_noise_sq = i_noise_sq
        else:
            image_to_convolve = self.image_data
            target_image = self.template_data
            noise_to_convolve_sq = i_noise_sq
            target_noise_sq = t_noise_sq
        
        if mask is None:
             mask = np.zeros(image_to_convolve.shape, dtype=np.uint16)

        if self.use_c_extension:
            convolved_image, output_mask, conv_noise_sq = self.ext.apply_kernel(self._c_state, image_to_convolve, self.results["kernel_solution"], noise_to_convolve_sq)
        else:
            # Pure Python Convolution
            # 1. Calculate Basis Functions
            oversample_param = self.oversample if self.results["conv_direction"] == 't' else 1
            
            # Scale Basis if needed
            scale = oversample_param
            hr_rkernel = self.config.rkernel * scale
            k_size = 2 * hr_rkernel + 1
            scaled_sigmas = [s * scale for s in self.config.sigma_gauss]

            basis_funcs = pure.kernel.calculate_kernel_basis((k_size, k_size), scaled_sigmas, self.config.deg_fixe)
            
            # Convert basis_funcs to 3D array for Numba
            basis_vecs = np.array(basis_funcs)
            
            # 3. Apply Kernel
            convolved_image, conv_noise_sq, output_mask_conv = pure.convolution.apply_kernel(
                image_to_convolve, self.results["kernel_solution"], 
                noise_to_convolve_sq, mask, 
                self.config, basis_vecs,
                oversample=oversample_param
            )
            
            # Output mask from convolution needs to be merged with existing?
            # Hotpants C logic: convolve returns modified mask.
            output_mask = output_mask_conv

        if self.use_c_extension:
            bkg = self.ext.get_background_image(self._c_state, self.results["kernel_solution"])
            convolved_image += bkg

        diff_image = target_image - convolved_image
        final_noise = np.sqrt(conv_noise_sq + target_noise_sq)

        if self.config.rescale_ok:
            if self.config.verbose >= 1:
                print("Rescaling noise for OK pixels...")
            final_noise = self.ext.rescale_noise_ok(self._c_state, diff_image, final_noise, output_mask)

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

        final_diff, final_conv, final_noise, output_mask = (self.results["diff_image"].copy(), self.results["convolved_image"].copy(), self.results["noise_image"].copy(), self.results["output_mask"].copy())
        bad_pixels = output_mask != 0
        final_diff[bad_pixels] = self.config.fillval
        final_conv[bad_pixels] = self.config.fillval
        final_noise[bad_pixels] = self.config.fillval_noise

        if self.use_c_extension:
            self.results["stats"] = self.ext.calculate_final_stats(self._c_state, final_diff, final_noise, output_mask)
        else:
            # Pure Python Stats Calculation
            valid_pixels = (output_mask == 0) & np.isfinite(final_diff) & (final_noise > 0)
            
            if np.any(valid_pixels):
                diff_vals = final_diff[valid_pixels]
                noise_vals = final_noise[valid_pixels]
                
                stats = {}
                stats["diff_mean"] = float(np.mean(diff_vals))
                stats["diff_std"] = float(np.std(diff_vals))
                stats["noise_mean"] = float(np.mean(noise_vals))
                stats["nx2norm"] = int(np.sum(valid_pixels))
                
                # Chi2 Norm: sum((diff / noise)^2) / N
                chi2 = np.sum((diff_vals / noise_vals)**2)
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
        kernel_center = self.visualize_kernel(at_coords=(self.nx // 2, self.ny // 2), size_factor=1.0)
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
        self.find_stamps()
        self.fit_and_select_direction()
        self.iterative_fit_and_clip()
        self.convolve_and_difference()
        self.save_outputs()
        return self.get_final_outputs()

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
             # Pure Python Implementation
            k_size = 2 * self.config.rkernel + 1
            if not self.config.deg_fixe: # Handle potential missing config
                 self.config.deg_fixe = [self.config.ko] * self.config.ngauss # Default or error? Assuming config is valid.
                 
            basis_funcs = pure.kernel.calculate_kernel_basis((k_size, k_size), self.config.sigma_gauss, self.config.deg_fixe)
            kernel_vecs = np.array(basis_funcs)
            
            x, y = at_coords
            kernel_sol = self.results["kernel_solution"]
            
            step = getattr(self.config, 'kc_step', k_size)
            ker_order = self.config.ko
            n_comp_ker = len(basis_funcs)
            
            local_kernel = pure.convolution.jit_make_kernel(
                kernel_sol, step, self.config.rkernel, 
                self.nx, self.ny,
                n_comp_ker, ker_order, kernel_vecs,
                x, y
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
