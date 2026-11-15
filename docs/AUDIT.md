# Audit scope and validation

This document describes the 0.3.0rc1 candidate based on v0.2.0 (`faa9e4d`). It is
an implementation audit, not a guarantee that arbitrary scientific recipes or
all platforms produce equivalent photometry. See CHANGELOG.md for migrations.

## Reviewed code

| Area | Review and improvements | Regression coverage |
| --- | --- | --- |
| `hotpants_ext.c` | Every exported method, NumPy layout/size validation, ownership, construction/destruction, partial allocations, configuration parsing, local/global fit records and statistics | `test_native_safety.py`, memory probe |
| `alard.c`, `functions.c`, headers | Allocation/free paths, singular LU propagation, background-disabled fitting, bounds, mask indexing, finite statistics, file helper errors | Native suite, fixed-operator parity, Valgrind, injected allocation failures |
| `core.py`, `config.py`, models and saved convolution | Configuration/image snapshots, argument validation, stage invalidation, actual surviving substamps, normalization/pedestals, caches, FITS masks and kernel serialization | `test_pipeline_contracts.py` |
| Python/Numba kernels, fitting, convolution, utilities | Bounds checks, finite normal equations, singular SciPy warnings, support geometry, deterministic noise samples, cutout padding | Fixed-operator comparisons and public workflow tests |
| Connected regions | Input masks, geometry/weights, rejection and per-direction FOM cache ownership | Connected-region suite and automatic-versus-forced regression |
| Oversampled FFT path | Cached transform, thread controls, shapes, block alignment and localization of nonfinite input support | FFT reference tests plus complete grid/connected workflows |
| Packaging, docs, examples and workflows | Missing SciPy dependency/license files, NumPy ABI build policy, supported Python floor, installed-wheel tests, release/citation preparation | Isolated sdist-to-wheel build, clean environment, docs build, CI matrix |

The standalone legacy programs `src/main.c`, `vargs.c`, `maskim.c`, and
`extractkern*.c` are retained as historical upstream reference, not compiled,
installed or supported by this Python package. Their CLI is not an alternative
entry point to this stateful library. Obtain the upstream HOTPANTS distribution
for its standalone CLI. The source archive contains the C files actually used by
the extension and their headers/license. No stable external C ABI is promised.

## Checks to reproduce

Set temporary/build/cache locations appropriate to your machine before running.
Use a separate environment/checkout; do not rebuild an editable installation
that a production pipeline is importing.

```sh
python -m pip install '.[test]'
python -m pytest tests
python -m build
# Install the wheel into a clean environment, then run from outside the checkout:
python -m pytest /absolute/path/to/tests --import-mode=importlib
```

Native lifetimes and C allocation failures (Linux/glibc):

```sh
valgrind --error-exitcode=86 --leak-check=full \
  --errors-for-leak-kinds=definite,indirect \
  --show-leak-kinds=definite,indirect \
  python tools/memory_probe.py --package /absolute/path/to/hotpants --repeat 15

gcc -shared -fPIC -O0 tools/native_alloc_fail.c -ldl -o /safe/scratch/alloc_fail.so
LD_PRELOAD=/safe/scratch/alloc_fail.so \
  python tools/memory_probe.py --package /absolute/path/to/hotpants --oom
```

The fault injector fails each allocation position made directly by the native
extension during representative operations. It does not inject failures into
Python/NumPy allocators. Run it separately from Valgrind; its glibc-specific
allocator interposition is not intended for production use.

The native memory probe covers successful and singular fits, configuration
parse failures, repeated searches/FOM evaluation, destruction between stages,
convolution, visualization, statistics and noise rescaling. Resident-memory
trends complement these traces; RSS alone cannot prove absence of leaks.

## Numerical interpretation

The wrapper preserves its historical Gaussian coefficient convention:

$$G(x)=\exp(-a x^2),\qquad a=\frac{1}{2\sigma^2}.$$

Here $G$ is the dimensionless one-dimensional Gaussian factor; $x$ is the pixel
offset; $a$ is the configured `sigma_gauss` coefficient in inverse pixel squared;
and $\sigma$ is the corresponding Gaussian width in pixels. This is different
from the upstream CLI's width input conversion. Changing documentation does not
change existing configured kernels.

Normalization uses the fitted constant kernel sum:

$$D_{\rm normalized}=D_{\rm reference}/s.$$

Here $D_{\rm reference}$ is the raw reference-minus-model difference in the
unconvolved image's units, $s$ is the fitted dimensionless kernel-sum coefficient
`kernel_solution[1]`, and $D_{\rm normalized}$ is the difference in the convolved
input image's units. This division is applied only when `normalize` selects the
convolved input. Background, model and noise products are scaled consistently.
`normalize="u"` retains reference units, matching the old wrapper's behavior.

Fixed-operator C/Python agreement is tested separately from end-to-end refits.
Selection, clipping and solver choices may differ; numerical safety does not
establish sub-percent photometric equivalence. The repository's JWST example is
an additional integration smoke test, not a held-out scientific accuracy study.

## Release gates

- Regression suite and installed-wheel tests pass.
- Native memory and allocation-failure probes pass.
- Both licenses and correct dependency metadata are present in distributions.
- Review numerical migrations on the intended SynDiff recipes before adoption.
- Platform CI must actually pass before claiming macOS/Windows support verified.
- Confirm final author metadata/version before creating the Zenodo/ASCL records.

The audit does not modify the live editable installation, create a DOI, publish
an ASCL record, or merge/deploy this candidate. Those are separate actions.
