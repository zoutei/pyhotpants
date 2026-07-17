.. pyhotpants documentation master file

pyhotpants documentation
========================

**Version 0.2.0** (staging on the ``development`` branch)

``pyhotpants`` is a Python wrapper around A. Becker's HOTPANTS
(High Order Transform of PSF And Template Subtraction) C code. It exposes the
Alard & Lupton image-subtraction algorithm through a NumPy/Astropy-friendly API.

Stable releases on ``main`` remain **0.1.x** (currently prepared as 0.1.2) with
the C extension as the default backend. This **0.2.0** documentation describes
the staging line on ``development``, which adds an opt-in pure Python backend,
oversampled templates, and connected-region stamps. Old call patterns continue
to work; see :ref:`compatibility`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   compatibility
   backends
   parity
   connected_stamps
   api
   config
   models
   convolve
