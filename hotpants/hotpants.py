# hotpants.py
"""
HOTPANTS Python Wrapper - Modular Implementation

This module provides a complete Python interface to the HOTPANTS image differencing
algorithms. Each step of the pipeline is exposed as an individual method for
fine-grained control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

from . import functions as pyhotpants

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


__version__ = "0.1.0"


class HotpantsError(Exception):
    """Exception raised for HOTPANTS-specific errors."""

    pass


# New Data Models for Substamp Tracking
class SubstampStatus(Enum):
    """Enumeration for the status of a substamp during the fitting process."""

    FOUND = auto()
    PASSED_FOM_CHECK = auto()
    REJECTED_FOM_CHECK = auto()
    USED_IN_FINAL_FIT = auto()
    REJECTED_ITERATIVE_FIT = auto()


class Substamp:
    """Data class to hold all information about a single substamp."""

    def __init__(self, substamp_id: int, stamp_group_id: int, x: float, y: float):
        self.id: int = substamp_id
        self.stamp_group_id: int = stamp_group_id
        self.x: float = x
        self.y: float = y
        self.status: SubstampStatus = SubstampStatus.FOUND
        self.image_cutout: Optional[np.ndarray] = None
        self.template_cutout: Optional[np.ndarray] = None
        self.noise_variance_cutout: Optional[np.ndarray] = None
        self.fit_results: Dict[str, Dict[str, float]] = {}

    def __repr__(self) -> str:
        return f"Substamp(id={self.id}, group={self.stamp_group_id}, coords=({self.x:.2f}, {self.y:.2f}), status={self.status.name})"


class HotpantsConfig:
    """
    Configuration class for HOTPANTS parameters, mirroring the original C defaults.
    """

    def __init__(self, **kwargs):
        # Image-specific parameters
        # [-tu tuthresh]    : upper valid data count, template (max value in template)
        self.tuthresh = kwargs.get("tuthresh", None)
        # [-tuk tucthresh]  : upper valid data count for kernel, template (tuthresh)
        self.tuktresh = kwargs.get("tuktresh", None)
        # [-tl tlthresh]    : lower valid data count, template (min value in template)
        self.tlthresh = kwargs.get("tlthresh", 0.0)
        # [-tg tgain]       : gain in template (1)
        self.tgain = kwargs.get("tgain", 1.0)
        # [-tr trdnoise]    : e- readnoise in template (0)
        self.trdnoise = kwargs.get("trdnoise", 0.0)
        # [-tp tpedestal]   : ADU pedestal in template (0)
        self.tpedestal = kwargs.get("tpedestal", 0.0)
        # [-iu iuthresh]    : upper valid data count, image (max value in image)
        self.iuthresh = kwargs.get("iuthresh", None)
        # [-iuk iucthresh]  : upper valid data count for kernel, image (iuthresh)
        self.iuktresh = kwargs.get("iuktresh", None)
        # [-il ilthresh]    : lower valid data count, image (min value in image)
        self.ilthresh = kwargs.get("ilthresh", None)
        # [-ig igain]       : gain in image (1)
        self.igain = kwargs.get("igain", 1.0)
        # [-ir irdnoise]    : e- readnoise in image (0)
        self.irdnoise = kwargs.get("irdnoise", 0.0)
        # [-ip ipedestal]   : ADU pedestal in image (0)
        self.ipedestal = kwargs.get("ipedestal", 0.0)

        # Kernel fitting parameters
        # [-r rkernel]      : convolution kernel half width (10)
        self.rkernel = kwargs.get("rkernel", 10)
        # [-ko kernelorder] : spatial order of kernel variation within region (2)
        self.ko = kwargs.get("ko", 2)
        # [-bgo bgorder]    : spatial order of background variation within region (1)
        self.bgo = kwargs.get("bgo", 1)
        # [-ft fitthresh]   : RMS threshold for good centroid in kernel fit (20.0)
        self.fitthresh = kwargs.get("fitthresh", 20.0)
        # # [-sft scale]      : scale fitthresh by this fraction if... (0.5)
        self.scale_fitthresh = kwargs.get("scale_fitthresh", 0.5)
        # [-nft fraction]   : this fraction of stamps are not filled (0.1)
        self.min_frac_stamps = kwargs.get("min_frac_stamps", 0.1)
        # [-nss substamps]  : number of centroids to use for each stamp (3)
        self.nss = kwargs.get("nss", 3)
        # [-rss radius]     : half width substamp to extract around each centroid (15)
        self.rss = kwargs.get("rss", 15)
        # [-ks badkernelsig]: high sigma rejection for bad stamps in kernel fit (2.0)
        self.ks = kwargs.get("ks", 2.0)
        # [-kfm kerfracmask]: fraction of abs(kernel) sum for ok pixel (0.990)
        self.kfm = kwargs.get("kfm", 0.99)
        # [-ssig statsig]   : threshold for sigma clipping statistics  (3.0)
        self.stat_sig = kwargs.get("stat_sig", 3.0)
        # [-mins spread]    : Fraction of kernel half width to spread input mask (1.0)
        self.kf_spread_mask1 = kwargs.get("kf_spread_mask1", 1.0)

        # General and miscellaneous
        # [-v] verbosity    : level of verbosity, 0-2 (1)
        self.verbose = kwargs.get("verbose", 1)
        # [-c  toconvolve]  : force convolution on (t)emplate or (i)mage (undef)
        self.force_convolve = kwargs.get("force_convolve", "b")
        # [-n  normalize]   : normalize to (t)emplate, (i)mage, or (u)nconvolved (t)
        self.normalize = kwargs.get("normalize", "t")
        # [-fom figmerit]   : (v)ariance, (s)igma or (h)istogram convolution merit (v)
        self.fom = kwargs.get("fom", "v")
        # [-fi fill]        : value for invalid (bad) pixels (1.0e-30)
        self.fillval = kwargs.get("fillval", 1e-30)
        # [-fin fill]       : noise image only fillvalue (0.0e+00)
        self.fillval_noise = kwargs.get("fillval_noise", 0.0)
        # [-okn]            : rescale noise for 'ok' pixels (0)
        self.rescale_ok = kwargs.get("rescale_ok", False)
        # [-convvar]        : convolve variance not noise (0)
        self.conv_var = kwargs.get("conv_var", False)
        self.use_pca = kwargs.get("use_pca", False)

        # Assumed single region for this wrapper
        # [-nrx xregion]    : number of image regions in x dimension (1)
        self.nregx = 1
        # [-nry yregion]    : number of image regions in y dimension (1)
        self.nregy = 1
        # [-nsx xstamp]     : number of each region's stamps in x dimension (10)
        self.nstampx = kwargs.get("nstampx", 10)
        # [-nsy ystamp]     : number of each region's stamps in y dimension (10)
        self.nstampy = kwargs.get("nstampy", 10)

        # Derived values for C code
        self.hwkernel = self.rkernel
        self.hwksstamp = self.rss
        self.fwkernel = 2 * self.hwkernel + 1
        self.fwksstamp = 2 * self.hwksstamp + 1

        _nx = kwargs.get("nx", 2048)
        _ny = kwargs.get("ny", 2048)

        # This calculation mirrors the logic in main.c to estimate stamp width
        fwstamp_est = min(_nx / self.nregx / self.nstampx, _ny / self.nregy / self.nstampy)
        fwstamp_est -= self.fwkernel
        fwstamp_est -= 1 if int(fwstamp_est) % 2 == 0 else 0
        self.fwstamp = int(max(fwstamp_est, self.fwksstamp + self.fwkernel))

        # [-ng ngauss]      : number of gaussians which compose kernel (3)
        self.ngauss = 3
        # [-ng ... degree0 .. degreeN] : degree of polynomial associated with gaussian #
        self.deg_fixe = [6, 4, 2]
        # [-ng ... sigma0 .. sigmaN] : width of gaussian #
        self.sigma_gauss = [0.7, 1.5, 3.0]
        self.ncomp = ((self.ko + 1) * (self.ko + 2)) // 2
        self.n_bg_vectors = ((self.bgo + 1) * (self.bgo + 2)) // 2
        self.ncomp_ker = sum(((d + 1) * (d + 2)) // 2 for d in self.deg_fixe)
        self.n_comp_total = self.ncomp_ker * self.ncomp + self.n_bg_vectors

        if self.tuktresh is None:
            self.tuktresh = self.tuthresh
        if self.iuktresh is None:
            self.iuktresh = self.iuthresh

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for passing to the C extension."""
        d = self.__dict__.copy()
        # These are Python-only and not needed by the C extension
        d.pop("_nx", None)
        d.pop("_ny", None)
        # Ensure lists are passed correctly
        d["deg_fixe"] = list(d["deg_fixe"])
        d["sigma_gauss"] = list(d["sigma_gauss"])
        return d


class Hotpants:
    """
    The central HOTPANTS object for stateful image differencing.
    The public methods correspond to the major steps of the pipeline.
    """

    def __init__(
        self,
        template_data: np.ndarray,
        image_data: np.ndarray,
        t_mask: Optional[np.ndarray] = None,
        i_mask: Optional[np.ndarray] = None,
        t_error: Optional[np.ndarray] = None,
        i_error: Optional[np.ndarray] = None,
        star_catalog: Optional[np.ndarray] = None,
        config: Optional[HotpantsConfig] = None,
    ):
        """
        Initializes the Hotpants object with template and image data.
        This now also creates the initial masks and noise images.

        Args:
            template_data (np.ndarray): The template image data.
            image_data (np.ndarray): The image data to be differenced.
            t_mask (np.ndarray, optional): An optional mask for the template.
            i_mask (np.ndarray, optional): An optional mask for the image.
            t_error (np.ndarray, optional): An optional error/noise image for the template.
            i_error (np.ndarray, optional): An optional error/noise image for the image.
            star_catalog (np.ndarray, optional): A pre-existing array of
                star positions to use for kernel fitting, bypassing the stamp search.
                Must be in 1-based system coordinates.
            config (HotpantsConfig, optional): A custom configuration object.
        """
        self.ext = _get_ext()
        self._validate_images(template_data, image_data, "template and image")

        self.template_data = np.ascontiguousarray(template_data, dtype=np.float32)
        self.image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        self.ny, self.nx = self.template_data.shape

        self.config = config if config is not None else HotpantsConfig(nx=self.nx, ny=self.ny)
        self._t_error_input = np.ascontiguousarray(t_error, dtype=np.float32) if t_error is not None else None
        self._i_error_input = np.ascontiguousarray(i_error, dtype=np.float32) if i_error is not None else None

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
        self._c_state = self.ext.HotpantsState(self.nx, self.ny, self.config.to_dict())
        print(f"Initialized HOTPANTS state: {self._c_state}")

        # 1. Create the initial input mask from the images and C extension.
        self._t_mask_input = np.ascontiguousarray(t_mask, dtype=np.int32) if t_mask is not None else None
        self._i_mask_input = np.ascontiguousarray(i_mask, dtype=np.int32) if i_mask is not None else None

        input_mask = self.ext.make_input_mask(self._c_state, self.template_data, self.image_data, self._t_mask_input, self._i_mask_input)
        print(f"Input mask created with shape: {input_mask.shape}, dtype: {input_mask.dtype}")
        self.results["input_mask"] = input_mask

        # 2. Generate noise images using C extension if not provided by the user.
        if self._t_error_input is not None:
            # Use user-provided noise image, squared.
            t_noise_sq = np.ascontiguousarray(self._t_error_input**2, dtype=np.float32)
        else:
            # Generate noise image from scratch and square it.
            t_noise_sq = self.ext.calculate_noise_image(self._c_state, self.template_data, True)

        if self._i_error_input is not None:
            # Use user-provided noise image, squared.
            i_noise_sq = np.ascontiguousarray(self._i_error_input**2, dtype=np.float32)
        else:
            # Generate noise image from scratch and square it.
            i_noise_sq = self.ext.calculate_noise_image(self._c_state, self.image_data, False)

        # 3. Store the squared noise images for later use.
        self.results["t_noise_sq"] = t_noise_sq
        self.results["i_noise_sq"] = i_noise_sq

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
        Step 1: Finds potential substamp coordinates.
        This populates the master lists with Substamp objects containing only coordinate data.
        """
        t_substamps_coords, i_substamps_coords = self.ext.find_stamps(self._c_state, self.template_data, self.image_data, self.config.fitthresh, self.star_catalog)

        self.template_substamps = [Substamp(**coords) for coords in t_substamps_coords]
        self.image_substamps = [Substamp(**coords) for coords in i_substamps_coords]

        if self.config.verbose >= 1:
            print(f"Found {len(self.template_substamps)} potential template substamps and {len(self.image_substamps)} potential image substamps.")

        if not self.template_substamps and not self.image_substamps:
            raise HotpantsError("No valid substamps found for kernel fitting.")

        return self.template_substamps, self.image_substamps

    def fit_and_select_direction(self) -> str:
        """
        Step 2: Performs initial fits, populates substamp data, and selects the best
        convolution direction. Updates the status of all tested substamps.
        """
        if not self.template_substamps and not self.image_substamps:
            self.find_stamps()

        combined_error_sq = self.results["t_noise_sq"] + self.results["i_noise_sq"]
        t_fom, i_fom = float("inf"), float("inf")
        t_fit_results, i_fit_results = None, None

        # Fit template-derived substamps
        if self.template_substamps:
            t_coords = [{"substamp_id": s.id, "stamp_group_id": s.stamp_group_id, "x": s.x, "y": s.y} for s in self.template_substamps]
            t_fom, t_fit_results = self.ext.fit_stamps_and_get_fom(self._c_state, self.template_data, self.image_data, combined_error_sq, "t", t_coords)
            for substamp, result in zip(self.template_substamps, t_fit_results):
                substamp.image_cutout = result["image_cutout"]
                substamp.template_cutout = result["template_cutout"]
                substamp.noise_variance_cutout = result["noise_cutout"]
                substamp.fit_results["t"] = {"fom": result["fom"], "chi2": result["chi2"]}

        # Fit image-derived substamps
        if self.image_substamps:
            i_coords = [{"substamp_id": s.id, "stamp_group_id": s.stamp_group_id, "x": s.x, "y": s.y} for s in self.image_substamps]
            i_fom, i_fit_results = self.ext.fit_stamps_and_get_fom(self._c_state, self.image_data, self.template_data, combined_error_sq, "i", i_coords)
            for substamp, result in zip(self.image_substamps, i_fit_results):
                substamp.image_cutout = result["image_cutout"]
                substamp.template_cutout = result["template_cutout"]
                substamp.noise_variance_cutout = result["noise_cutout"]
                substamp.fit_results["i"] = {"fom": result["fom"], "chi2": result["chi2"]}

        # Select best direction
        conv_direction = self.config.force_convolve
        if conv_direction == "b":
            conv_direction = "t" if t_fom < i_fom else "i"
            if self.config.verbose >= 1:
                print(f"Template FOM: {t_fom:.3f}, Image FOM: {i_fom:.3f}. Selecting direction: '{conv_direction}'")

        # Update status based on the winning direction
        if conv_direction == "t":
            for substamp, result in zip(self.template_substamps, t_fit_results):
                substamp.status = SubstampStatus.PASSED_FOM_CHECK if result["survived_check"] else SubstampStatus.REJECTED_FOM_CHECK
        else:  # 'i'
            for substamp, result in zip(self.image_substamps, i_fit_results):
                substamp.status = SubstampStatus.PASSED_FOM_CHECK if result["survived_check"] else SubstampStatus.REJECTED_FOM_CHECK

        self.results["conv_direction"] = conv_direction
        return conv_direction

    def iterative_fit_and_clip(self) -> Tuple[np.ndarray, List[Substamp]]:
        """
        Step 3 & 4: Performs iterative clipping and computes the global kernel solution.
        Updates the status of candidate substamps to reflect the outcome.
        """
        if "conv_direction" not in self.results:
            self.fit_and_select_direction()
        if self.config.verbose >= 1:
            print("Starting iterative kernel fit and solution...")

        conv_direction = self.results["conv_direction"]
        candidate_substamps = self.template_substamps if conv_direction == "t" else self.image_substamps

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

    def convolve_and_difference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 5: Applies the final kernel to the appropriate image, calculates the difference,
        and computes the final noise image.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - The difference image
                - The convolved image
                - The final noise image
                - The output mask
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
        else:
            image_to_convolve = self.image_data
            target_image = self.template_data
            noise_to_convolve_sq = i_noise_sq
            target_noise_sq = t_noise_sq

        convolved_image, output_mask, conv_noise_sq = self.ext.apply_kernel(self._c_state, image_to_convolve, self.results["kernel_solution"], noise_to_convolve_sq)

        bkg = self.ext.get_background_image(self._c_state, self.results["kernel_solution"])
        convolved_image += bkg

        diff_image = target_image - convolved_image
        final_noise = np.sqrt(conv_noise_sq + target_noise_sq)

        if self.config.rescale_ok:
            if self.config.verbose >= 1:
                print("Rescaling noise for OK pixels...")
            final_noise = self.ext.rescale_noise_ok(self._c_state, diff_image, final_noise, output_mask)

        self.results.update({"convolved_image": convolved_image, "output_mask": output_mask, "diff_image": diff_image, "noise_image": final_noise})

        self._populate_global_convolved_models()

        return diff_image, convolved_image, final_noise, output_mask

    def get_final_outputs(self) -> Dict[str, Any]:
        """
        Step 6: Applies final masks and calculates statistics.
        Returns a dictionary of all final output data.
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

        self.results["stats"] = self.ext.calculate_final_stats(self._c_state, final_diff, final_noise, output_mask)

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

    def run_pipeline(self) -> Dict[str, Any]:
        """A convenience method to run the entire pipeline in a single call."""
        self.find_stamps()
        self.fit_and_select_direction()
        self.iterative_fit_and_clip()
        self.convolve_and_difference()
        return self.get_final_outputs()

    def visualize_kernel(self, at_coords: Tuple[int, int], size_factor: float = 2.0) -> np.ndarray:
        """
        Generates an image of the convolution kernel at a specific coordinate.

        This method should be called *after* the pipeline has been run and a
        kernel solution has been found. It uses the final kernel solution to
        reconstruct the kernel for the given (x, y) location.

        Args:
            at_coords (Tuple[int, int]): The (x, y) coordinates at which to visualize the kernel.
            size_factor (float, optional): A multiplier for the kernel's width to
                determine the output image size. Defaults to 2.0.

        Returns:
            np.ndarray: A 2D NumPy array containing the image of the kernel.

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

        kernel_image = self.ext.visualize_kernel(self._c_state, at_coords, self.results["kernel_solution"], size_factor)
        return kernel_image

    def get_substamp_details(self) -> Dict[str, Any]:
        """
        Returns the complete, stateful master lists of all substamps and a
        summary of the locations used in the final fit.
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
            x_center = int(round(substamp.x))
            y_center = int(round(substamp.y))

            cutout = pyhotpants.cut_substamp_from_image(image=convolved_image, x_center=x_center, y_center=y_center, half_width=hw, fill_value=fill_value)
            substamp.convolved_model_global = cutout
