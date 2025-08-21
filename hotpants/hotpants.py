# hotpants.py
"""
HOTPANTS Python Wrapper - Modular Implementation

This module provides a complete Python interface to the HOTPANTS image differencing
algorithms. Each step of the pipeline is exposed as an individual method for
fine-grained control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

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


__version__ = "1.0.0"


class HotpantsError(Exception):
    """Exception raised for HOTPANTS-specific errors."""

    pass


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
            config (HotpantsConfig, optional): A custom configuration object.
        """
        self.ext = _get_ext()
        self._validate_images(template_data, image_data, "template and image")

        self.template_data = np.ascontiguousarray(template_data, dtype=np.float32)
        self.image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        self.ny, self.nx = self.template_data.shape

        self.config = config if config is not None else HotpantsConfig(nx=self.nx, ny=self.ny)
        self._t_error_input = t_error
        self._i_error_input = i_error

        if star_catalog is not None:
            if not isinstance(star_catalog, np.ndarray) or star_catalog.ndim != 2 or star_catalog.shape[1] != 2:
                raise HotpantsError("star_catalog must be a 2D NumPy array with shape (N, 2).")
            self.star_catalog = np.ascontiguousarray(star_catalog, dtype=np.float32)
        else:
            self.star_catalog = None

        self.results = {}

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

        # 1. Create the initial input mask from the images and C extension.
        input_mask = self.ext.make_input_mask(self._c_state, self.template_data, self.image_data, t_mask, i_mask)
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

    def find_stamps(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Step 1: Finds potential stamps for kernel fitting.
        This method now dispatches to the C extension function with the appropriate
        parameters based on whether a star catalog is provided.

        Returns:
            List[Tuple[float, float]]: A list of (x, y) coordinates for the found stamps.
        """
        t_stamps, i_stamps = self.ext.find_stamps(self._c_state, self.template_data, self.image_data, self.config.fitthresh, self.star_catalog)

        num_t_stamps = len(t_stamps)
        num_i_stamps = len(i_stamps)

        if self.config.verbose >= 1:
            print(f"Found {num_t_stamps} template stamps and {num_i_stamps} image stamps.")

        if num_t_stamps == 0 and num_i_stamps == 0:
            raise HotpantsError("No valid stamps found for kernel fitting.")

        self.results["t_stamps_data"] = t_stamps
        self.results["i_stamps_data"] = i_stamps
        return t_stamps, i_stamps

    def fit_and_select_direction(self) -> Tuple[str, List[Dict]]:
        """
        Step 2: Performs initial kernel fits for both convolution directions and selects
        the best one based on the figure of merit.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: The chosen convolution direction ('t' or 'i')
                and the list of best-fit stamps with their figure of merit.
        """
        if "t_stamps_data" not in self.results:
            self.find_stamps()

        combined_error_sq = self.results["t_noise_sq"] + self.results["i_noise_sq"]

        conv_direction = self.config.force_convolve
        if conv_direction == "b":
            t_fits, t_fom = self.ext.fit_stamps_and_get_fom(self._c_state, self.template_data, self.image_data, combined_error_sq, "t", self.results["t_stamps_data"])
            i_fits, i_fom = self.ext.fit_stamps_and_get_fom(self._c_state, self.image_data, self.template_data, combined_error_sq, "i", self.results["i_stamps_data"])
            conv_direction = "t" if t_fom < i_fom else "i"
            best_fits = t_fits if t_fom < i_fom else i_fits
            if self.config.verbose >= 1:
                print(f"Template FOM: {t_fom:.3f}, Image FOM: {i_fom:.3f}. Convolving: {conv_direction}")
        else:
            stamps_to_fit = self.results["t_stamps_data"] if conv_direction == "t" else self.results["i_stamps_data"]
            conv_img, ref_img = (self.template_data, self.image_data) if conv_direction == "t" else (self.image_data, self.template_data)
            best_fits, _ = self.ext.fit_stamps_and_get_fom(self._c_state, conv_img, ref_img, combined_error_sq, conv_direction, stamps_to_fit)

        self.results["conv_direction"] = conv_direction
        self.results["best_fits"] = best_fits
        return conv_direction, best_fits

    def iterative_fit_and_clip(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Step 3 & 4: Performs iterative clipping of stamps and computes the global kernel solution.

        Returns:
            Tuple containing:
                - np.ndarray: The array of global kernel and background coefficients.
                - List[Dict[str, Any]]: The final, clipped list of best-fit stamps.
        """
        if "best_fits" not in self.results:
            self.fit_and_select_direction()
        if self.config.verbose >= 1:
            print("Starting iterative kernel fit and solution...")

        conv_direction = self.results["conv_direction"]
        if conv_direction == "t":
            conv_img, ref_img = self.template_data, self.image_data
        else:
            conv_img, ref_img = self.image_data, self.template_data

        kernel_solution, final_fits, stats = self.ext.fit_kernel(self._c_state, self.results["best_fits"], conv_img, ref_img, self.results["t_noise_sq"] + self.results["i_noise_sq"])

        if len(final_fits) == 0:
            raise HotpantsError("All stamps were clipped during iterative fitting.")
        if self.config.verbose >= 1:
            print(f"Final fit uses {len(final_fits)} stamps. Fit stats: mean_sig={stats['meansig']:.3f}, scatter={stats['scatter']:.3f}")

        self.results["kernel_solution"] = kernel_solution
        self.results["final_fits"] = final_fits
        self.results["fit_stats"] = stats
        return kernel_solution, final_fits

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
