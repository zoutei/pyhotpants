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
            from . import hotpants_ext as ext_module

            hotpants_ext = ext_module
        except ImportError as e:
            raise RuntimeError(f"Could not import C extension: {e}")
    return hotpants_ext


__version__ = "1.1.0"


class HotpantsError(Exception):
    """Exception raised for HOTPANTS-specific errors."""

    pass


class HotpantsConfig:
    """
    Configuration class for HOTPANTS parameters, mirroring the original C defaults.
    """

    def __init__(self, **kwargs):
        # Image-specific parameters
        # [-tu tuthresh]    : upper valid data count, template (25000)
        self.tuthresh = kwargs.get("tuthresh", 25000.0)
        # [-tuk tucthresh]  : upper valid data count for kernel, template (tuthresh)
        self.tuktresh = kwargs.get("tuktresh", None)
        # [-tl tlthresh]    : lower valid data count, template (0)
        self.tlthresh = kwargs.get("tlthresh", 0.0)
        # [-tg tgain]       : gain in template (1)
        self.tgain = kwargs.get("tgain", 1.0)
        # [-tr trdnoise]    : e- readnoise in template (0)
        self.trdnoise = kwargs.get("trdnoise", 0.0)
        # [-tp tpedestal]   : ADU pedestal in template (0)
        self.tpedestal = kwargs.get("tpedestal", 0.0)
        # [-iu iuthresh]    : upper valid data count, image (25000)
        self.iuthresh = kwargs.get("iuthresh", 25000.0)
        # [-iuk iucthresh]  : upper valid data count for kernel, image (iuthresh)
        self.iuktresh = kwargs.get("iuktresh", None)
        # [-il ilthresh]    : lower valid data count, image (0)
        self.ilthresh = kwargs.get("ilthresh", 0.0)
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
        # [-sft scale]      : scale fitthresh by this fraction if... (0.5)
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

        # Derived values for C code (for convenience)
        self.hwkernel = self.rkernel
        self.hwksstamp = self.rss
        self.fwkernel = 2 * self.hwkernel + 1
        self.fwksstamp = 2 * self.hwksstamp + 1
        self.fwstamp = self.hwksstamp * 2 + 1 + self.hwkernel * 2 + 1
        self.ncomp_ker = 0
        # [-ng ngauss]      : number of gaussians which compose kernel (3)
        self.ngauss = 3
        # [-ng ... degree0 .. degreeN] : degree of polynomial associated with gaussian #
        self.deg_fixe = [6, 4, 2]
        # [-ng ... sigma0 .. sigmaN] : width of gaussian #
        self.sigma_gauss = [0.7, 1.5, 3.0]
        self.ncomp = ((self.ko + 1) * (self.ko + 2)) // 2
        self.n_bg_vectors = ((self.bgo + 1) * (self.bgo + 2)) // 2

        for i in range(self.ngauss):
            self.ncomp_ker += ((self.deg_fixe[i] + 1) * (self.deg_fixe[i] + 2)) // 2
        self.n_comp_total = self.ncomp_ker * self.ncomp + self.n_bg_vectors

        if self.tuktresh is None:
            self.tuktresh = self.tuthresh
        if self.iuktresh is None:
            self.iuktresh = self.iuthresh

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert config to a dictionary for passing to the C extension.
        This also handles the case where `deg_fixe` or `sigma_gauss` might
        not be set, though they should be.
        """
        d = self.__dict__.copy()
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
        star_catalog: Optional[List[Tuple[float, float]]] = None,
        config: Optional[HotpantsConfig] = None,
    ):
        """
        Initializes the Hotpants object with template and image data.

        Args:
            template_data (np.ndarray): The template image data.
            image_data (np.ndarray): The image data to be differenced.
            t_mask (np.ndarray, optional): An optional mask for the template.
            i_mask (np.ndarray, optional): An optional mask for the image.
            t_error (np.ndarray, optional): An optional error/noise image for the template.
            i_error (np.ndarray, optional): An optional error/noise image for the image.
            star_catalog (List[Tuple[float, float]], optional): A pre-existing list of
                star positions to use for kernel fitting, bypassing the stamp search.
            config (HotpantsConfig, optional): A custom configuration object.
        """
        self.ext = _get_ext()

        self._validate_images(template_data, image_data, "template and image")

        self.template_data = np.ascontiguousarray(template_data, dtype=np.float32)
        self.image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        self.ny, self.nx = self.template_data.shape

        if t_mask is not None:
            self._validate_images(template_data, t_mask, "template and template mask")
        if i_mask is not None:
            self._validate_images(image_data, i_mask, "image and image mask")

        self.config = config if config is not None else HotpantsConfig()

        # Create a single C state object to hold all C-level state
        self._c_state = self.ext.HotpantsState(self.nx, self.ny, self.config.to_dict())

        self._t_mask_input = t_mask
        self._i_mask_input = i_mask
        self._t_error_input = t_error
        self._i_error_input = i_error
        self._star_catalog_input = star_catalog

        self.results = {}

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

    def find_stamps(self) -> List[Tuple[float, float]]:
        """
        Step 1: Finds potential stamps for kernel fitting.
        Uses a stamp finding algorithm or a pre-defined star catalog.

        Returns:
            List[Tuple[float, float]]: A list of (x, y) coordinates for the found stamps.
        """
        config_dict = self.config.to_dict()
        fit_thresh = self.config.fitthresh
        attempts = 0
        all_stamps = []

        # Create initial mask
        input_mask = self.ext.make_input_mask(self._c_state, self.template_data, self.image_data)
        # Incorporate provided external masks if they exist
        if self._t_mask_input is not None:
            input_mask |= np.ascontiguousarray(self._t_mask_input, dtype=np.int32)
        if self._i_mask_input is not None:
            input_mask |= np.ascontiguousarray(self._i_mask_input, dtype=np.int32)
        self.results["input_mask"] = input_mask

        while attempts < 2:
            if attempts > 0 and self.config.verbose >= 1:
                print(f"Attempt {attempts + 1}: Too few stamps, scaling down threshold to {fit_thresh}")

            if self._star_catalog_input is not None:
                # Use provided star catalog, but filter them with the mask
                all_stamps = [(x, y) for x, y in self._star_catalog_input if input_mask[int(y), int(x)] == 0]
                if self.config.verbose >= 1:
                    print(f"Using {len(all_stamps)} stamps from provided catalog.")
                break
            else:
                all_stamps = self.ext.find_stamps(
                    self._c_state,
                    self.template_data,
                    self.image_data,
                    input_mask,
                    fit_thresh,
                )

            if len(all_stamps) / (self.config.nstampx * self.config.nstampy) >= self.config.min_frac_stamps:
                if self.config.verbose >= 1:
                    print(f"Found {len(all_stamps)} stamps, enough to proceed.")
                break

            fit_thresh *= self.config.scale_fitthresh
            attempts += 1

        if len(all_stamps) == 0:
            raise HotpantsError("No valid stamps found for kernel fitting.")

        self.results["all_stamps"] = all_stamps
        return all_stamps

    def fit_and_select_direction(self) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Step 2: Performs initial kernel fits for both convolution directions and selects
        the best one based on the figure of merit.

        Returns:
            Tuple[str, List[Dict[str, Any]]]: The chosen convolution direction ('t' or 'i')
                and the list of best-fit stamps with their figure of merit.
        """
        if "all_stamps" not in self.results:
            self.find_stamps()

        # Generate noise images if not provided by the user
        if self._t_error_input is not None:
            t_error_sq = np.ascontiguousarray(self._t_error_input**2, dtype=np.float32)
        else:
            t_error_sq = self.ext.calculate_noise_image(self._c_state, self.template_data, is_template=True)

        if self._i_error_input is not None:
            i_error_sq = np.ascontiguousarray(self._i_error_input**2, dtype=np.float32)
        else:
            i_error_sq = self.ext.calculate_noise_image(self._c_state, self.image_data, is_template=False)

        combined_error_sq = t_error_sq + i_error_sq
        self.results["combined_error_sq"] = combined_error_sq

        conv_direction = self.config.force_convolve
        if self.config.force_convolve == "b":
            t_fits, t_fom = self.ext.fit_stamps_and_get_fom(self._c_state, self.template_data, self.image_data, combined_error_sq, "t", self.results["all_stamps"])
            i_fits, i_fom = self.ext.fit_stamps_and_get_fom(self._c_state, self.image_data, self.template_data, combined_error_sq, "i", self.results["all_stamps"])

            conv_direction = "t" if t_fom < i_fom else "i"
            best_fits = t_fits if t_fom < i_fom else i_fits

            if self.config.verbose >= 1:
                print(f"Template FOM: {t_fom:.3f}, Image FOM: {i_fom:.3f}. Convolving: {conv_direction}")
        else:
            if conv_direction == "t":
                conv_img, ref_img = self.template_data, self.image_data
            else:
                conv_img, ref_img = self.image_data, self.template_data
            best_fits, _ = self.ext.fit_stamps_and_get_fom(self._c_state, conv_img, ref_img, combined_error_sq, conv_direction, self.results["all_stamps"])

        self.results["conv_direction"] = conv_direction
        self.results["best_fits"] = best_fits

        return conv_direction, best_fits

    def iterative_fit_and_clip(self) -> List[Dict[str, Any]]:
        """
        Step 3: Performs the iterative sigma clipping of stamps based on the final
        global solution. This process continues until the stamp list converges.

        Returns:
            List[Dict[str, Any]]: The final, clipped list of best-fit stamps.
        """
        if "best_fits" not in self.results or "combined_error_sq" not in self.results:
            self.fit_and_select_direction()

        iter_count = 0
        max_iter = 10
        converged = False

        current_fits = self.results["best_fits"]

        # Use the correct images based on the convolution direction
        if self.results["conv_direction"] == "t":
            conv_img, ref_img = self.template_data, self.image_data
        else:
            conv_img, ref_img = self.image_data, self.template_data

        while not converged and iter_count < max_iter:
            if self.config.verbose >= 2:
                print(f"Starting iterative fit loop, iteration {iter_count + 1}")

            current_fits, skip_count, check_again_needed = self.ext.check_and_refit_stamps(
                self._c_state,
                current_fits,
                conv_img,
                ref_img,
                self.results["combined_error_sq"],
                self.results["conv_direction"],
            )

            if not check_again_needed:
                converged = True

            iter_count += 1
            if len(current_fits) == 0:
                raise HotpantsError("All stamps were clipped during iterative fitting.")

        if not converged and self.config.verbose >= 1:
            print("Warning: Stamp filtering did not converge within max iterations.")

        if self.config.verbose >= 1:
            print(f"Final fit uses {len(current_fits)} stamps.")

        self.results["best_fits"] = current_fits
        return current_fits

    def get_global_solution(self) -> np.ndarray:
        """
        Step 4: Computes the global kernel solution from the final set of stamps.

        Returns:
            np.ndarray: The array of global kernel and background coefficients.
        """
        if "best_fits" not in self.results:
            self.iterative_fit_and_clip()

        if self.config.verbose >= 1:
            print("Computing global kernel solution...")

        kernel_solution = self.ext.get_global_solution(self._c_state, self.results["best_fits"])
        self.results["kernel_solution"] = kernel_solution
        return kernel_solution

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
            self.get_global_solution()

        conv_direction = self.results["conv_direction"]

        # Get noise images from inputs or generate them
        if self._t_error_input is not None:
            t_noise = np.ascontiguousarray(self._t_error_input, dtype=np.float32)
        else:
            t_noise = self.ext.calculate_noise_image(self._c_state, self.template_data, is_template=True)

        if self._i_error_input is not None:
            i_noise = np.ascontiguousarray(self._i_error_input, dtype=np.float32)
        else:
            i_noise = self.ext.calculate_noise_image(self._c_state, self.image_data, is_template=False)

        if conv_direction == "t":
            image_to_convolve = self.template_data
            target_image = self.image_data
            noise_to_convolve_sq = t_noise**2
            target_noise_sq = i_noise**2
        else:
            image_to_convolve = self.image_data
            target_image = self.template_data
            noise_to_convolve_sq = i_noise**2
            target_noise_sq = t_noise**2

        convolved_image, output_mask = self.ext.apply_kernel(self._c_state, image_to_convolve, self.results["kernel_solution"])

        bkg = self.ext.get_background_image(self._c_state, convolved_image.shape, self.results["kernel_solution"])
        convolved_image += bkg

        diff_image = target_image - convolved_image
        final_noise = np.sqrt(noise_to_convolve_sq + target_noise_sq)

        if self.config.rescale_ok:
            if self.config.verbose >= 1:
                print("Rescaling noise for OK pixels...")
            final_noise = self.ext.rescale_noise_ok(self._c_state, diff_image, final_noise, output_mask)

        self.results["convolved_image"] = convolved_image
        self.results["output_mask"] = output_mask
        self.results["diff_image"] = diff_image
        self.results["noise_image"] = final_noise

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

        # Fill masked pixels with `fillval`
        final_diff = self.results["diff_image"].copy()
        final_conv = self.results["convolved_image"].copy()
        final_noise = self.results["noise_image"].copy()
        output_mask = self.results["output_mask"].copy()

        final_diff[output_mask != 0] = self.config.fillval
        final_conv[output_mask != 0] = self.config.fillval
        final_noise[output_mask != 0] = self.config.fillval_noise

        stats = self.ext.calculate_final_stats(self._c_state, final_diff, final_noise, output_mask)
        self.results["stats"] = stats

        return {
            "diff_image": final_diff,
            "convolved_image": final_conv,
            "noise_image": final_noise,
            "output_mask": output_mask,
            "stats": stats,
            "conv_direction": self.results["conv_direction"],
            "kernel_solution": self.results["kernel_solution"],
        }

    def run_pipeline(self) -> Dict[str, Any]:
        """
        A convenience method to run the entire pipeline in a single call.
        """
        self.find_stamps()
        self.fit_and_select_direction()
        self.iterative_fit_and_clip()
        self.get_global_solution()
        self.convolve_and_difference()
        return self.get_final_outputs()
