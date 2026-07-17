.. _parity_policy:

C vs pure Python parity policy
==============================

For classic configurations the pure Python backend aims to reproduce the C
extension results on the same inputs.

In scope
--------

* ``use_c_extension=False``
* ``oversample=1``
* ``stamp_mode="grid"`` (classic substamps)

Compared products (minimum)
---------------------------

* Difference image
* Convolved image / background (as returned by the public API)
* Kernel solution vector
* Output mask (exact or documented bit policy)
* Noise map when part of the returned products

Tolerances should start strict (float32 ``atol`` / ``rtol`` as appropriate) and
only be widened with a written justification per product.

Out of scope (no C twin)
------------------------

* ``oversample > 1``
* ``stamp_mode="connected_regions"``

These modes are validated with their own unit tests and real-data soak runs,
not by bit-matching the C grid path.

How we test
-----------

* Synthetic / unit tests under ``tests/`` (e.g. connected-region builders)
* Real-data comparisons (JWST JADES example data under ``example/data/`` and
  developer notebooks such as ``dev/compare_c_vs_pure_jwst.ipynb``)
* A curated fixture pack with C golden outputs is planned before promoting
  0.2.0 from ``development`` to ``main``

Until that fixture harness lands, treat 0.2.0 on ``development`` as **staging**:
suitable for further testing, not yet the default install target.
