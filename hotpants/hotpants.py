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
        self.tuthresh = kwargs.get("tuthresh", 25000.0)
        self.tuktresh = kwargs.get("tuktresh", None)
        self.tlthresh = kwargs.get("tlthresh", 0.0)
        self.tgain = kwargs.get("tgain", 1.0)
        self.trdnoise = kwargs.get("trdnoise", 0.0)
        self.tpedestal = kwargs.get("tpedestal", 0.0)
        self.iuthresh = kwargs.get("iuthresh", 25000.0)
        self.iuktresh = kwargs.get("iuktresh", None)
        self.ilthresh = kwargs.get("ilthresh", 0.0)
        self.igain = kwargs.get("igain", 1.0)
        self.irdnoise = kwargs.get("irdnoise", 0.0)
        self.ipedestal = kwargs.get("ipedestal", 0.0)

        # Kernel fitting parameters
        self.rkernel = kwargs.get("rkernel", 10)
        self.ko = kwargs.get("ko", 2)
        self.bgo = kwargs.get("bgo", 1)
        self.fitthresh = kwargs.get("fitthresh", 20.0)
        self.scale_fitthresh = kwargs.get("scale_fitthresh", 0.5)
        self.min_frac_stamps = kwargs.get("min_frac_stamps", 0.1)
        self.nss = kwargs.get("nss", 3)
        self.rss = kwargs.get("rss", 15)
        self.ks = kwargs.get("ks", 2.0)
        self.kfm = kwargs.get("kfm", 0.99)

        # General and miscellaneous
        self.verbose = kwargs.get("verbose", 1)
        self.force_convolve = kwargs.get("force_convolve", "b")
        self.normalize = kwargs.get("normalize", "t")
        self.fom = kwargs.get("fom", "v")  # figure of merit: 'v', 's', 'h'
        self.fillval = kwargs.get("fillval", 1e-30)
        self.fillval_noise = kwargs.get("fillval_noise", 0.0)
        self.rescale_ok = kwargs.get("rescale_ok", False)
        self.conv_var = kwargs.get("conv_var", False)
        self.use_pca = kwargs.get("use_pca", False)

        # Assumed single region for this wrapper
        self.nregx = 1
        self.nregy = 1
        self.nstampx = kwargs.get("nstampx", 10)
        self.nstampy = kwargs.get("nstampy", 10)

        # Derived values for C code (for convenience)
        self.hwkernel = self.rkernel
        self.hwksstamp = self.rss
        self.fwkernel = 2 * self.hwkernel + 1
        self.fwksstamp = 2 * self.hwksstamp + 1
        self.fwstamp = self.hwksstamp * 2 + 1 + self.hwkernel * 2 + 1
        self.ncomp_ker = 0
        self.ngauss = 3
        self.deg_fixe = [6, 4, 2]
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
        """Convert config to dictionary for passing to C extension."""
        d = self.__dict__.copy()
        d["fwkernel"] = self.fwkernel
        d["fwksstamp"] = self.fwksstamp
        d["ncomp_ker"] = self.ncomp_ker
        d["ncomp"] = self.ncomp
        d["n_bg_vectors"] = self.n_bg_vectors
        d["n_comp_total"] = self.n_comp_total
        d["deg_fixe"] = d["deg_fixe"]
        d["sigma_gauss"] = d["sigma_gauss"]
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
        t_noise: Optional[np.ndarray] = None,
        i_noise: Optional[np.ndarray] = None,
        star_catalog: Optional[List[Tuple[float, float]]] = None,
        config: Optional[HotpantsConfig] = None,
    ):
        self._validate_images(template_data, image_data)
        self.template_data = np.ascontiguousarray(template_data, dtype=np.float32)
        self.image_data = np.ascontiguousarray(image_data, dtype=np.float32)
        self.ny, self.nx = self.template_data.shape
        self.config = config if config is not None else HotpantsConfig()

        # Store optional inputs
        self._t_mask_input = t_mask
        self._i_mask_input = i_mask
        self._t_noise_input = t_noise
        self._i_noise_input = i_noise
        self._star_catalog_input = star_catalog

        self.results = {}
        self.ext = _get_ext()

        if self.config.verbose >= 1:
            print(f"Initialized Hotpants object: {self.nx}x{self.ny} images")

    def _validate_images(self, template: np.ndarray, image: np.ndarray):
        if template.ndim != 2 or image.ndim != 2:
            raise HotpantsError("Input images must be 2D arrays")
        if template.shape != image.shape:
            raise HotpantsError("Template and image must have the same dimensions")

    def _prepare_inputs(self, fit_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares mask and noise images based on whether they were provided or need to be generated.
        Returns the combined input mask and combined initial noise squared.
        """
        config_dict = self.config.to_dict()

        # Use provided mask if available, otherwise generate it
        if self._t_mask_input is not None and self._i_mask_input is not None:
            combined_mask = self._t_mask_input | self._i_mask_input
            input_mask = np.ascontiguousarray(combined_mask, dtype=np.int32)
        else:
            input_mask = self.ext.make_input_mask(self.template_data, self.image_data, config_dict)

        # Use provided noise if available, otherwise generate it
        if self._t_noise_input is not None and self._i_noise_input is not None:
            temp_noise_sq = np.ascontiguousarray(self._t_noise_input**2, dtype=np.float32)
            image_noise_sq = np.ascontiguousarray(self._i_noise_input**2, dtype=np.float32)
        else:
            temp_noise_sq = self.ext.calculate_noise_image(self.template_data, config_dict, is_template=True)
            image_noise_sq = self.ext.calculate_noise_image(self.image_data, config_dict, is_template=False)

        combined_noise_sq = temp_noise_sq + image_noise_sq
        self.results["combined_noise_sq"] = combined_noise_sq

        return input_mask, combined_noise_sq

    def find_stamps(self) -> List[Tuple[float, float]]:
        """
        Step 1: Finds potential stamps for kernel fitting.
        Replicates the C code's iterative stamp finding loop.
        """
        config_dict = self.config.to_dict()
        fit_thresh = self.config.fitthresh
        attempts = 0
        all_stamps = []

        while attempts < 2:
            if attempts > 0 and self.config.verbose >= 1:
                print(f"Attempt {attempts + 1}: Too few stamps, scaling down threshold to {fit_thresh}")

            input_mask, _ = self._prepare_inputs(fit_thresh)

            if self._star_catalog_input is not None:
                # Use provided star catalog, but filter them with the mask
                all_stamps = [(x, y) for x, y in self._star_catalog_input if input_mask[int(y), int(x)] == 0]
                if self.config.verbose >= 1:
                    print(f"Using {len(all_stamps)} stamps from provided catalog.")
                break
            else:
                all_stamps = self.ext.find_stamps(self.template_data, self.image_data, input_mask, fit_thresh, config_dict)

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
        """
        if "all_stamps" not in self.results:
            self.find_stamps()

        config_dict = self.config.to_dict()

        if self.config.verbose >= 1:
            print(f"Fitting initial kernels on {len(self.results['all_stamps'])} stamps...")

        if "combined_noise_sq" not in self.results:
            _, combined_noise_sq = self._prepare_inputs(self.config.fitthresh)
        else:
            combined_noise_sq = self.results["combined_noise_sq"]

        conv_direction = self.config.force_convolve
        if self.config.force_convolve == "b":
            t_fits, t_fom = self.ext.fit_stamps_and_get_fom(self.template_data, self.image_data, combined_noise_sq, self.results["all_stamps"], "t", config_dict)
            i_fits, i_fom = self.ext.fit_stamps_and_get_fom(self.image_data, self.template_data, combined_noise_sq, self.results["all_stamps"], "i", config_dict)

            conv_direction = "t" if t_fom < i_fom else "i"
            best_fits = t_fits if t_fom < i_fom else i_fits

            if self.config.verbose >= 1:
                print(f"Template FOM: {t_fom:.3f}, Image FOM: {i_fom:.3f}. Convolving: {conv_direction}")
        else:
            best_fits, _ = self.ext.fit_stamps_and_get_fom(self.template_data if conv_direction == "t" else self.image_data, self.image_data if conv_direction == "t" else self.template_data, combined_noise_sq, self.results["all_stamps"], conv_direction, config_dict)

        self.results["conv_direction"] = conv_direction
        self.results["best_fits"] = best_fits

        return conv_direction, best_fits

    def iterative_fit_and_clip(self) -> List[Dict[str, Any]]:
        """
        Step 3: Performs the iterative sigma clipping of stamps based on the final
        global solution. This process continues until the stamp list converges.
        """
        if "best_fits" not in self.results or "combined_noise_sq" not in self.results:
            self.fit_and_select_direction()

        config_dict = self.config.to_dict()
        iter_count = 0
        max_iter = 10
        converged = False

        current_fits = self.results["best_fits"]

        while not converged and iter_count < max_iter:
            if self.config.verbose >= 2:
                print(f"Starting iterative fit loop, iteration {iter_count + 1}")

            current_fits, skip_count, check_again_needed = self.ext.check_and_refit_stamps(current_fits, self.template_data, self.image_data, self.results["combined_noise_sq"], self.results["conv_direction"], config_dict)

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
        """
        if "best_fits" not in self.results:
            self.iterative_fit_and_clip()

        config_dict = self.config.to_dict()
        if self.config.verbose >= 1:
            print("Computing global kernel solution...")

        kernel_solution = self.ext.get_global_solution(self.results["best_fits"], config_dict)
        self.results["kernel_solution"] = kernel_solution
        return kernel_solution

    def convolve_and_difference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 5: Applies the final kernel to the appropriate image, calculates the difference,
        and computes the final noise image.
        """
        if "kernel_solution" not in self.results:
            self.get_global_solution()

        config_dict = self.config.to_dict()
        if self.config.verbose >= 1:
            print("Convolving image and creating difference image...")

        conv_direction = self.results["conv_direction"]

        # Get noise images from inputs or generate them
        if self._t_noise_input is not None:
            t_noise = np.ascontiguousarray(self._t_noise_input, dtype=np.float32)
        else:
            t_noise = self.ext.calculate_noise_image(self.template_data, config_dict, is_template=True)

        if self._i_noise_input is not None:
            i_noise = np.ascontiguousarray(self._i_noise_input, dtype=np.float32)
        else:
            i_noise = self.ext.calculate_noise_image(self.image_data, config_dict, is_template=False)

        if conv_direction == "t":
            image_to_convolve = self.template_data
            target_image = self.image_data
            noise_to_convolve = t_noise**2
            target_noise = i_noise**2
        else:
            image_to_convolve = self.image_data
            target_image = self.template_data
            noise_to_convolve = i_noise**2
            target_noise = t_noise**2

        convolved_image, output_mask = self.ext.apply_kernel(image_to_convolve, self.results["kernel_solution"], config_dict)

        bkg = self.ext.get_background_image(convolved_image.shape, self.results["kernel_solution"], config_dict)
        convolved_image += bkg

        diff_image = target_image - convolved_image
        final_noise = np.sqrt(noise_to_convolve + target_noise)

        if self.config.rescale_ok:
            if self.config.verbose >= 1:
                print("Rescaling noise for OK pixels...")
            final_noise = self.ext.rescale_noise_ok(diff_image, final_noise, output_mask, config_dict)

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

        config_dict = self.config.to_dict()
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

        stats = self.ext.calculate_final_stats(final_diff, final_noise, output_mask, config_dict)
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
