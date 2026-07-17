.. _connected_stamps:

Connected-region stamps
=======================

Connected-region stamp mode builds irregular fitting regions from a star
catalog instead of the classic rectangular grid search. It is available only
in the pure Python backend.

Requirements
------------

* ``use_c_extension=False``
* ``config.stamp_mode = "connected_regions"``
* A ``star_catalog`` array of shape ``(N, 2)`` (FITS 1-based coordinates, same
  convention as elsewhere in ``Hotpants``)

Example
-------

::

   from hotpants import Hotpants, HotpantsConfig

   config = HotpantsConfig(
       stamp_mode="connected_regions",
       region_max_diameter=40.0,
       # other kernel / noise parameters as usual...
   )

   hp = Hotpants(
       template,
       science,
       star_catalog=catalog_xy,  # shape (N, 2), FITS 1-based
       config=config,
       use_c_extension=False,
   )
   results = hp.run_pipeline()

Related ``HotpantsConfig`` fields
---------------------------------

See :ref:`config_reference` for defaults and meanings, including:

* ``region_max_diameter``, ``region_max_area``
* ``region_connectivity``, ``region_rss``
* ``region_weight``, ``region_min_npix``
* ``region_bisect_on_reject``, ``region_max_bisects``, ``region_weight_cap``

Classic ``stamp_mode="grid"`` remains the default and is unchanged for C and
pure backends.
