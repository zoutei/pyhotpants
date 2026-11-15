# Changelog

## 0.3.0rc1 — unreleased audit candidate

### Native safety

- Validate all NumPy buffers, kernel sizes, catalog coordinates, substamp group
  capacities and configuration dimensions at the extension boundary.
- Fix repeated stamp-search leaks, destruction after automatic direction
  selection, forced-direction scratch leaks, late configuration-parser leaks,
  and singular-solver workspace leaks. Reject state reinitialization.
- Unwind partial allocations and propagate native allocation failures to
  `MemoryError`; remove process termination from the compiled library helpers.
- Support local and global fitting with `bgo=-1`, bound small-image masking,
  fix noise-rescaling null pointers, and reject failed numerical solutions.
- Python owns returned arrays or an explicit capsule owns their C buffer;
  outputs remain valid after the originating state is destroyed.

### Numerical changes requiring migration checks

- Honor `normalize="t"` (the documented default), `"i"`, and `"u"`. Previously
  the option was ignored. Set `normalize="u"` to retain the old unscaled output
  convention. Images, background and noise are scaled consistently.
- Honor input pedestals and adjust explicit thresholds accordingly.
- Propagate caller masks as bad input, flag uncomputed convolution borders,
  correct transposed stamp-selection masking, and mask nonfinite data.
- Kernel visualization now evaluates the requested position. Singular fits
  are rejected instead of continuing with invalid/uninitialized results.
- Report the actual surviving substamp within each C stamp group.
- Both backends expose `final_fits`; the Python backend supports noise rescaling.
- Keep `sigma_gauss`'s historical exponent-coefficient meaning; correct its
  documentation rather than silently reinterpret saved configurations.

### API and Python backend

- Snapshot caller images/configuration; reject numerical configuration changes
  on an existing object. Validate unsupported options rather than ignore them.
- Handle singular SciPy LU warnings, reject nonfinite solves, enable Numba
  bounds checks, validate FFT/support shapes and localize nonfinite FFT inputs.
- Preserve cutout fill values, validate coordinate lengths, make catalog noise
  sampling reproducible, and preserve unsigned mask bits in FITS round trips.
- Saved kernel models copy their coefficients/configuration; reject unsupported
  oversampled standalone models and malformed kernel vectors.

### Distribution

- Python >=3.10; declare SciPy, correct license distribution, and separate wrapper
  author metadata from upstream credits. Add software citation metadata without
  inventing a DOI or release date.
- Build NumPy extension wheels with NumPy >=2.0 for 1.x/2.x compatibility.
- Test installed wheels on Linux/macOS/Windows in CI; add Valgrind and native
  allocation-failure jobs. Platform CI is a gate, not evidence until it runs.

This candidate has not been deployed to the live SynDiff environment or
published to Zenodo/ASCL. Revalidate real-data recipes before adopting numerical
changes. Existing saved kernel coefficients retain their layout.

Additional audit findings: connected-region FOM caches are now selected by
convolution direction; producing C diagnostic cutouts no longer changes the
representative substamp used for group scoring. Nonfinite/degenerate Gaussian
bases are rejected, and polynomial basis coordinates use floating arithmetic
instead of overflowing integer powers at high degree.

Automatic stamp search now terminates on invalid/empty statistics, handles zero
background scatter safely, and bounds degenerate histogram interpolation. Native
container parsing rejects user conversion callbacks that could mutate borrowed
inputs during validation.
