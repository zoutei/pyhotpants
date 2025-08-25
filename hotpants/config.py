# hotpants/config.py
"""
Configuration module for the HOTPANTS wrapper.
"""

from typing import Dict, Any

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
        use_pca (bool): If True, use PCA to model the kernel basis (not fully supported). (Default: False).
        nstampx (int): Number of stamps to place in the x-direction per region. (Default: 10).
        nstampy (int): Number of stamps to place in the y-direction per region. (Default: 10).
        output_file (str): Path to save the difference FITS image. (Default: None).
        noise_image_file (str): Path to save the noise FITS image. (Default: None).
        mask_image_file (str): Path to save the mask FITS image. (Default: None).
        convolved_image_file (str): Path to save the convolved FITS image. (Default: None).
        sigma_image_file (str): Path to save the sigma (difference/noise) FITS image. (Default: None).
        stamp_region_file (str): Path to save the DS9 region file of stamps. (Default: None).

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

        # Output file paths
        self.output_file = kwargs.get("output_file", None)
        self.noise_image_file = kwargs.get("noise_image_file", None)
        self.mask_image_file = kwargs.get("mask_image_file", None)
        self.convolved_image_file = kwargs.get("convolved_image_file", None)
        self.sigma_image_file = kwargs.get("sigma_image_file", None)
        self.stamp_region_file = kwargs.get("stamp_region_file", None)

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
