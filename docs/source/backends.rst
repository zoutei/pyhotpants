.. _backends:

Backends (C extension vs pure Python)
=====================================

``pyhotpants`` supports two execution backends. **Backward compatibility is
a hard requirement:** existing scripts that omit backend flags continue to
use the C extension path.

Default: C extension
--------------------

::

   from hotpants import Hotpants, HotpantsConfig

   hp = Hotpants(template, science, config=HotpantsConfig(...))
   # equivalent to use_c_extension=True
   results = hp.run_pipeline()

This is the stable path shipped on ``main`` (0.1.x). Prefer it for production
workflows unless you need a pure-Python-only feature.

Opt-in: pure Python
-------------------

::

   hp = Hotpants(
       template,
       science,
       config=HotpantsConfig(...),
       use_c_extension=False,
   )

The pure backend reimplements the Alard & Lupton pipeline in
``hotpants.pure`` (NumPy / Numba). For classic grid stamps with
``oversample=1``, it is intended to recover the **same scientific answer** as
the C path (see :ref:`parity_policy`).

Pure-Python-only features
-------------------------

These require ``use_c_extension=False``:

* ``oversample > 1`` — high-resolution template / low-resolution science
* ``stamp_mode="connected_regions"`` — irregular connected stamps from a
  star catalog (see :ref:`connected_stamps`)

Attempting oversampled templates or connected-region stamps with the C
backend raises an error.
