.. _compatibility:

Compatibility and migration
===========================

0.3.0rc1 is an unreleased audit candidate based on v0.2.0. The C backend remains
the default; oversampling and connected-region stamps use Python/Numba.
Python 3.10 or newer is required.

Numerical changes
-----------------

``normalize`` previously had no effect. It now selects template (``t``, default),
image (``i``), or unconvolved/reference (``u``) units. Use ``normalize="u"`` to
retain the previous wrapper's unscaled output convention. Nonzero pedestals are
now applied. Caller-mask propagation, border flags and stamp-selection indexing
are corrected; these can change which pixels and stars contribute to fits.
Singular or nonfinite solutions are rejected. Validate old science recipes
before upgrading; fixed coefficient layout does not imply identical refits.

``sigma_gauss`` keeps its historical wrapper meaning: exponent coefficients,
not widths in pixels. A coefficient for Gaussian width sigma is
``1/(2*sigma**2)``. No automatic reinterpretation of saved configurations occurs.

Ownership and validation
------------------------

Objects snapshot caller images and configuration. Create a new object to change
numerical parameters. Output filenames remain configurable. Native methods
validate dtype, native byte order, contiguity, alignment, shape and capacities.
High-level methods convert ordinary array inputs. State reinitialization and
unsupported PCA kernels are rejected. Regularization is Python-backend only.

Outputs preserve the established keys. Both backends expose ``final_fits``.
Raw image values are returned together with masks; invalid pixels are not
silently replaced with fill values. Saved kernel models copy their state and
support native-resolution standalone convolution only.
