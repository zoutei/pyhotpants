.. pyhotpants documentation master file

pyhotpants documentation
========================

**Version 0.1.2**

``pyhotpants`` is a Python wrapper around A. Becker's HOTPANTS
(High Order Transform of PSF And Template Subtraction) C code. It exposes the
Alard & Lupton image-subtraction algorithm through a NumPy/Astropy-friendly API.

This release adds standalone template convolution via ``KernelModel`` and
``convolve_template`` (apply a saved kernel without re-running the full
pipeline). The default ``Hotpants`` pipeline behavior is unchanged.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   config
   models
   convolve
