.. _convolve_reference:

Standalone template convolution
===============================

After a successful HOTPANTS fit you can reuse the saved kernel on a template
image without re-running stamp finding or kernel fitting. This is useful for
applying a previously derived kernel to another (same-shape) template or for
offline inspection of the spatial convolution alone.

.. important::

   ``convolve_template`` returns the raw spatial convolution only. It does
   **not** add the spatial background polynomial that
   ``Hotpants.convolve_and_difference`` includes in the full pipeline.

Quick start
-----------

::

   from hotpants import Hotpants, HotpantsConfig, KernelModel, convolve_template

   hp = Hotpants(template, science, config=HotpantsConfig(...))
   hp.run_pipeline()

   kernel = KernelModel.from_hotpants(hp)
   convolved = convolve_template(template, kernel)

The template passed to ``convolve_template`` must have the same ``(ny, nx)``
shape as the image the kernel was fit on. Spatial kernel variation is tied to
those dimensions.

API
---

.. automodule:: hotpants.convolve
   :members:
   :undoc-members:
   :show-inheritance:
