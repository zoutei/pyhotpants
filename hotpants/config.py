# hotpants/config.py
"""
Configuration module for the HOTPANTS wrapper.
"""

from typing import Dict, Any
import math
import numbers


class HotpantsConfig:
    """
    Configuration class for HOTPANTS parameters.

    This class holds all the tunable parameters for the HOTPANTS algorithm,
    mirroring the command-line flags of the original C implementation. Default
    values are set to match the original code.

    An instance of this class is created by the Hotpants object, but a custom,
    pre-configured instance can also be passed during initialization.

    Attributes:
        tuthresh (float): Upper valid data count for the template.
            (Default: dynamically set to max value).
        tuktresh (float): Upper valid data count for kernel fitting on the template.
            (Default: Same as tuthresh).
        tlthresh (float): Lower valid data count for the template.
            (Default: dynamically set to min value).
        tgain (float): Gain in template (e-/ADU). (Default: 1.0).
        trdnoise (float): Read noise in template (e-). (Default: 0.0).
        tpedestal (float): ADU pedestal in template. (Default: 0.0).
        iuthresh (float): Upper valid data count for the image.
            (Default: dynamically set to max value).
        iuktresh (float): Upper valid data count for kernel fitting on the image.
            (Default: Same as iuthresh).
        ilthresh (float): Lower valid data count for the image.
            (Default: dynamically set to min value).
        igain (float): Gain in image (e-/ADU). (Default: 1.0).
        irdnoise (float): Read noise in image (e-). (Default: 0.0).
        ipedestal (float): ADU pedestal in image. (Default: 0.0).
        rkernel (int): Convolution kernel half-width in pixels. (Default: 10).
        ko (int): Spatial order of kernel variation within a region. (Default: 2).
        bgo (int): Spatial order of background variation within a region. (Default: 1).
        fitthresh (float): RMS threshold for good centroids in kernel fit. (Default: 20.0).
        nss (int): Number of centroids (sub-stamps) to use for each stamp. (Default: 3).
        rss (int): Half-width of sub-stamps to extract around each centroid. (Default: 15).
        ks (float): High sigma rejection for bad stamps in the kernel fit. (Default: 2.0).
        kfm (float): Fraction of absolute kernel sum for a pixel to be considered 'OK'. (Default: 0.990).
        stat_sig (float): Threshold for sigma clipping statistics. (Default: 3.0).
        kf_spread_mask1 (float): Fraction of kernel half-width to spread input masks by. (Default: 1.0).
        verbose (int): Level of verbosity, 0-2. (Default: 1).
        force_convolve (str): Force convolution on 't' (template) or 'i' (image). 'b' for best. (Default: 'b').
        normalize (str): Normalize to 't' (template), 'i' (image), or 'u' (unconvolved). (Default: 't').
        fom (str): Figure-of-merit for choosing convolution direction: 'v' (variance), 's' (sigma), or 'h' (histogram). (Default: 'v').
        fillval (float): Value for invalid (bad) pixels in the difference image. (Default: 1.0e-30).
        fillval_noise (float): Value for invalid pixels in the noise image. (Default: 0.0).
        rescale_ok (bool): If True, rescale noise for 'OK' pixels. (Default: False).
        conv_var (bool): If True, convolve variance instead of noise. (Default: False).
        use_pca (bool): If True, use PCA to model the kernel basis. (Default: False).
        nstampx (int): Number of stamps to place in the x-direction per region. (Default: 10).
        nstampy (int): Number of stamps to place in the y-direction per region. (Default: 10).
        output_file (str): Path to save the difference FITS image. (Default: None).
        noise_image_file (str): Path to save the noise FITS image. (Default: None).
        mask_image_file (str): Path to save the mask FITS image. (Default: None).
        convolved_image_file (str): Path to save the convolved FITS image. (Default: None).
        sigma_image_file (str): Path to save the sigma (difference/noise) FITS image. (Default: None).
        stamp_region_file (str): Path to save the DS9 region file of stamps. (Default: None).

        Connected-region stamps (pure Python only)
        ------------------------------------------
        stamp_mode (str): ``"grid"`` (classic) or ``"connected_regions"``.
        region_max_diameter (float): Split islands larger than this bbox span (px).
        region_weight (str): Equation weight ``uniform`` / ``npix`` / ``flux`` / ``invvar``.
        region_min_npix (int|None): Floor on good pixels after exclusion; ``None`` → one paint box.
        region_bisect_on_reject (bool): Split region if star exclusion leaves it too small.

        Kernel composition parameters
        -----------------------------
        The following parameters control how the convolution kernel is constructed from a sum of Gaussians. They mirror the `-ng` style flags in the original HOTPANTS tool and are passed through to the C layer where the kernel basis is built.

        ngauss (int): Number of Gaussian components which compose the kernel.
            Defaults to 3. Each Gaussian has an associated polynomial degree and
            a sigma (width) value.
        deg_fixe (List[int]): Polynomial degrees associated with each Gaussian
            component. Length should equal `ngauss`. Default: [6, 4, 2]. Each
            entry is the maximum polynomial degree for that Gaussian basis.
        sigma_gauss (List[float]): Gaussian exponent coefficients in inverse pixel squared
            units: exp(-coefficient * x**2). For width sigma in pixels,
            supply coefficient = 1/(2*sigma**2). Length should equal `ngauss`. Default: [0.7, 1.5, 3.0].

        Note:
            The following parameters from the original HOTPANTS are not implemented in this wrapper:
            - `scale_fitthresh` (-sft): Logic to scale `fitthresh` is not used.
            - `min_frac_stamps` (-nft): Check for minimum fraction of filled stamps is not used.
            - `nregx` and `nregy`: The wrapper treats the entire image as a single region, so these are fixed to 1.
    """

    def __init__(self, **kwargs):
        """
        Initializes the configuration object.

        Args:
            **kwargs: Keyword arguments corresponding to HOTPANTS parameters.
                For example, `rkernel=15` to set the kernel half-width. Any
                parameter not provided will use the default value.
        """
        # Image-specific parameters
        # [-tu tuthresh]    : upper valid data count, template (max value in template)
        self.tuthresh = kwargs.get("tuthresh", None)
        # [-tuk tucthresh]  : upper valid data count for kernel, template (tuthresh)
        self.tuktresh = kwargs.get("tuktresh", None)
        # [-tl tlthresh]    : lower valid data count, template (min value in template)
        self.tlthresh = kwargs.get("tlthresh", None)
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
        # [-v] verbosity    : level of verbosity, 0-2 (0)
        self.verbose = kwargs.get("verbose", 0)
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
        # Pure-path Tikhonov regularization (relative to mean matrix diagonal; 0 = off)
        self.lambda_reg = kwargs.get("lambda_reg", 0.0)

        # Connected-region stamps (pure Python only; stamp_mode="grid" is classic)
        self.stamp_mode = kwargs.get("stamp_mode", "grid")
        self.region_max_diameter = kwargs.get("region_max_diameter", 40.0)
        self.region_max_area = kwargs.get("region_max_area", 0)
        self.region_connectivity = kwargs.get("region_connectivity", 8)
        self.region_rss = kwargs.get("region_rss", None)  # None → use rss
        self.region_weight = kwargs.get("region_weight", "npix")
        self.region_min_npix = kwargs.get("region_min_npix", None)  # None → (2*rss+1)^2
        self.region_bisect_on_reject = kwargs.get("region_bisect_on_reject", False)
        self.region_max_bisects = kwargs.get("region_max_bisects", 100)
        self.region_weight_cap = kwargs.get("region_weight_cap", (0.25, 4.0))

        # Assumed single region for this wrapper
        # [-nrx xregion]    : number of image regions in x dimension (1)
        self.nregx = 1
        # [-nry yregion]    : number of image regions in y dimension (1)
        self.nregy = 1
        # [-nsx xstamp]     : number of each region's stamps in x dimension (10)
        self.nstampx = kwargs.get("nstampx", 10)
        if (
            not isinstance(self.nstampx, numbers.Integral)
            or isinstance(self.nstampx, bool)
            or self.nstampx < 1
        ):
            raise ValueError("nstampx must be a positive integer")
        # [-nsy ystamp]     : number of each region's stamps in y dimension (10)
        self.nstampy = kwargs.get("nstampy", 10)
        if (
            not isinstance(self.nstampy, numbers.Integral)
            or isinstance(self.nstampy, bool)
            or self.nstampy < 1
        ):
            raise ValueError("nstampy must be a positive integer")

        # Output file paths
        self.output_file = kwargs.get("output_file", None)
        self.noise_image_file = kwargs.get("noise_image_file", None)
        self.mask_image_file = kwargs.get("mask_image_file", None)
        self.convolved_image_file = kwargs.get("convolved_image_file", None)
        self.sigma_image_file = kwargs.get("sigma_image_file", None)
        self.stamp_region_file = kwargs.get("stamp_region_file", None)

        unknown = (
            set(kwargs)
            - set(self.__dict__)
            - {
                "nx",
                "ny",
                "ngauss",
                "deg_fixe",
                "sigma_gauss",
                "hwkernel",
                "hwksstamp",
                "fwkernel",
                "fwksstamp",
                "fwstamp",
                "ncomp",
                "n_bg_vectors",
                "ncomp_ker",
                "n_comp_total",
            }
        )
        if unknown:
            raise TypeError(
                f"Unknown HotpantsConfig parameters: {', '.join(sorted(unknown))}"
            )
        for name, minimum, maximum in (
            ("rkernel", 0, 16383),
            ("rss", 1, 16383),
            ("ko", 0, 64),
            ("bgo", -1, 64),
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
        for name in ("nx", "ny"):
            value = kwargs.get(name, 2048)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")

        # Derived values for C code
        self.hwkernel = self.rkernel
        self.hwksstamp = self.rss
        self.fwkernel = 2 * self.hwkernel + 1
        self.fwksstamp = 2 * self.hwksstamp + 1

        _nx = kwargs.get("nx", 2048)
        _ny = kwargs.get("ny", 2048)

        # This calculation mirrors the logic in main.c to estimate stamp width
        fwstamp_est = min(
            _nx / self.nregx / self.nstampx, _ny / self.nregy / self.nstampy
        )
        fwstamp_est -= self.fwkernel
        fwstamp_est -= 1 if int(fwstamp_est) % 2 == 0 else 0
        self.fwstamp = int(max(fwstamp_est, self.fwksstamp + self.fwkernel))

        # [-ng ngauss]      : number of gaussians which compose kernel (3)
        self.ngauss = kwargs.get("ngauss", len(kwargs.get("deg_fixe", [6, 4, 2])))
        # [-ng ... degree0 .. degreeN] : degree of polynomial associated with gaussian #
        self.deg_fixe = list(kwargs.get("deg_fixe", [6, 4, 2]))
        # [-ng ... sigma0 .. sigmaN] : width of gaussian #
        self.sigma_gauss = list(kwargs.get("sigma_gauss", [0.7, 1.5, 3.0]))
        self.validate()
        self.ncomp = ((self.ko + 1) * (self.ko + 2)) // 2
        self.n_bg_vectors = ((self.bgo + 1) * (self.bgo + 2)) // 2
        self.ncomp_ker = sum(((d + 1) * (d + 2)) // 2 for d in self.deg_fixe)
        self.n_comp_total = self.ncomp_ker * self.ncomp + self.n_bg_vectors

        if self.tuktresh is None:
            self.tuktresh = self.tuthresh
        if self.iuktresh is None:
            self.iuktresh = self.iuthresh

    def validate(self):
        """Reject invalid numeric and algorithm settings before either backend runs."""
        for name in ("nss", "nstampx", "nstampy", "ngauss"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if len(self.deg_fixe) != self.ngauss or len(self.sigma_gauss) != self.ngauss:
            raise ValueError("ngauss must match deg_fixe and sigma_gauss lengths")
        if any(
            isinstance(d, bool)
            or not isinstance(d, numbers.Integral)
            or not 0 <= d <= 64
            for d in self.deg_fixe
        ):
            raise ValueError("Gaussian degrees must be integers in [0, 64]")
        if any(not math.isfinite(s) or s <= 0 for s in self.sigma_gauss):
            raise ValueError("Gaussian coefficients must be finite and positive")
        choices = {
            "force_convolve": "tib",
            "normalize": "tiu",
            "fom": "vsh",
            "stamp_mode": ("grid", "connected_regions"),
            "region_weight": ("uniform", "npix", "flux", "invvar"),
            "region_connectivity": (4, 8),
        }
        for name, allowed in choices.items():
            value = getattr(self, name)
            if value not in allowed or (isinstance(allowed, str) and len(value) != 1):
                raise ValueError(f"invalid {name}: {value!r}")
        for name in ("tgain", "igain", "ks", "stat_sig", "region_max_diameter"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "trdnoise",
            "irdnoise",
            "fitthresh",
            "kf_spread_mask1",
            "lambda_reg",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "tuthresh",
            "tuktresh",
            "tlthresh",
            "iuthresh",
            "iuktresh",
            "ilthresh",
            "tpedestal",
            "ipedestal",
            "fillval",
            "fillval_noise",
        ):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or abs(value) > 3.402823466e38
            ):
                raise ValueError(f"{name} must be representable as a finite float32")
        if not math.isfinite(self.kfm) or not 0 <= self.kfm <= 1:
            raise ValueError("kfm must be in [0, 1]")
        if self.use_pca:
            raise ValueError(
                "PCA kernels are not supported: no PCA basis input is implemented"
            )
        for name in ("region_rss", "region_min_npix"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer or None")
        for name in ("region_max_area", "region_max_bisects", "verbose"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Integral)
                or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            len(self.region_weight_cap) != 2
            or not all(math.isfinite(v) for v in self.region_weight_cap)
            or not 0 < self.region_weight_cap[0] <= self.region_weight_cap[1]
        ):
            raise ValueError("region_weight_cap must be a positive ordered pair")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary for passing to the C extension."""
        d = self.__dict__.copy()
        # These are Python-only and not needed by the C extension
        d.pop("_nx", None)
        d.pop("_ny", None)
        # Ensure lists are passed correctly
        d["deg_fixe"] = list(d["deg_fixe"])
        d["sigma_gauss"] = list(d["sigma_gauss"])
        # Convert NumPy scalar configuration values before entering the C API.
        for key, value in d.items():
            if isinstance(value, numbers.Integral):
                d[key] = int(value)
            elif isinstance(value, numbers.Real):
                d[key] = float(value)
        d["deg_fixe"] = [int(v) for v in d["deg_fixe"]]
        d["sigma_gauss"] = [float(v) for v in d["sigma_gauss"]]
        return d
