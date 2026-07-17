.. _compatibility:

Compatibility and migration
===========================

Supported compatibility contract (0.2.0 staging)
------------------------------------------------

* Scripts written against **0.1.x** that construct ``Hotpants`` / ``HotpantsConfig``
  without new keyword arguments continue to work.
* Default backend remains the **C extension** (``use_c_extension=True``).
* Default stamp selection remains classic ``stamp_mode="grid"``.
* Public exports from 0.1.2 (``KernelModel``, ``convolve_template``) remain available.
* Return products from ``run_pipeline()`` / ``get_final_outputs()`` keep the
  established keys (difference image, convolved image, background, noise,
  mask, stats, kernel solution, convolution direction).

What is new in 0.2.0 (staging on ``development``)
-------------------------------------------------

* Pure Python backend via ``use_c_extension=False``
* Oversampled templates via ``oversample > 1``
* Connected-region stamps via ``stamp_mode="connected_regions"``
* Expanded documentation for backends, parity, and connected stamps

Migration tips
--------------

1. Keep production pipelines on the C default until you have validated pure
   Python on your data (see :ref:`parity_policy`).
2. Opt into new modes explicitly; do not rely on changed defaults.
3. When comparing C vs pure, pin the same ``HotpantsConfig`` and inputs.
