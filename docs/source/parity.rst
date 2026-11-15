.. _parity_policy:

C and Python validation
=======================

The tests compare fixed kernel operators, background, variance propagation,
mask policy and local kernel evaluation between backends. Synthetic end-to-end
workflows exercise both directions, automatic selection and background orders,
including disabled background. Connected-region and oversampled FFT tests
cover modes with no C counterpart.

End-to-end refits are not promised to be bit-identical: stamp selection,
statistics and numerical solvers differ. Fixed-operator agreement does not
establish real-data photometric equivalence. Validate each science recipe and
retain its version and configuration.

Native memory checks
--------------------

``tools/memory_probe.py`` exercises repeated initialization, searches, local and
global fits, convolution, visualization, statistics and failure cleanup.
Valgrind checks invalid accesses and lost allocations. The Linux/glibc preload
helper ``tools/native_alloc_fail.c`` enumerates native allocation failures;
it does not replace failure testing of Python or NumPy's own allocators.

See ``docs/AUDIT.md`` and ``CHANGELOG.md`` for coverage and migration details.
