# pyhotpants — A Python Wrapper for HOTPANTS (High Order Transform of PSF And Template Subtraction) 

![CI](https://github.com/zoutei/pyhotpants/actions/workflows/build.yml/badge.svg)

## **NOT AFFILIATED | UNDER DEVELOPMENT | USE AT YOUR OWN RISK**

This project is **not** affiliated with the original HOTPANTS project. It is currently under active development and may contain bugs or incomplete features. By using this software you acknowledge that the maintainers are not responsible for any damage, incorrect scientific results, or other issues that may arise from its use.


## pyhotpants
This project provides a modern, object-oriented Python wrapper for the
robust and widely-used HOTPANTS (High Order Transform of PSF And Template
Subtraction) C code, originally developed by A. Becker. It exposes the
Alard & Lupton image-subtraction algorithm through a clean, Pythonic API and
integrates with the NumPy/Astropy ecosystem.

The wrapper keeps the performant C core (via a small C extension) while
providing modular, inspectable, step-by-step execution so you can examine
intermediate products (kernel basis functions, substamp fits, global kernel
solution) or run the whole pipeline with a single call.

## Key features

- Full pipeline and step-by-step execution. Run everything at once or run
    individual phases for detailed analysis and debugging.
- Detailed inspection and control. Access intermediate products such as
    substamp cutouts, local kernel fits, and the final convolved models.
- Pythonic configuration. Command-line flags from the original HOTPANTS are
    mapped to a `HotpantsConfig` class.
- Seamless NumPy / Astropy integration. Works with NumPy arrays and Astropy
    FITS headers.
- Stateful, object-oriented design. The `Hotpants` class manages data,
    configuration, and results.
- High performance. The core algorithms are executed by the original C code
    via a lightweight extension.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/zoutei/pyhotpants
```

## Requirements

- Python: as declared in `pyproject.toml` (>=3.6). We recommend Python 3.8+.
- Core dependencies (installed automatically):
    - numpy >= 1.15.0
    - astropy >= 3.0.0
    - numba >= 0.51.2

To install the package for development:

```bash
pip install -e .
```

## Documentation

The project's full documentation is published on GitHub Pages: [Online documentation](https://zoutei.github.io/pyhotpants).

## Usage

You can initialize the wrapper with either FITS file paths or in-memory
NumPy arrays. See the example notebook for a detailed walkthrough:

- `example/JWST_JADES_example.ipynb` (example using JWST/NIRCam data)

### Running the full pipeline (FITS files)

```python
from hotpants import Hotpants, HotpantsConfig

# Configure HOTPANTS and specify the output file name.
config = HotpantsConfig(
        rkernel=15,
        normalize='t',
        output_file='diff.fits'
)

# Initialize with FITS file paths. A star_catalog is optional.
hp = Hotpants(
        template_data='template.fits',
        image_data='science.fits',
        config=config
)

# Run the pipeline. The result is saved to 'diff.fits'.
results = hp.run_pipeline()
print(f"Image subtraction complete! Final statistics: {results['stats']}")
```

### Running the full pipeline (NumPy arrays)

```python
import numpy as np
from hotpants import Hotpants, HotpantsConfig

template_image = np.load('template.npy')
science_image = np.load('science.npy')

hp = Hotpants(template_data=template_image, image_data=science_image)
results = hp.run_pipeline()

# Access the final difference image from the results dictionary
diff_image = results['diff_image']
```

## Configuration

The recommended starting configuration (example values) is shown below —
these are concrete numbers used in the example notebook rather than
computed on-the-fly. Adjust them for your data.

```python
from hotpants import HotpantsConfig

config = HotpantsConfig(
        # Kernel parameters
        rkernel=5,  # convolution kernel half-width (pixels)
        ko=2,       # spatial order of kernel variation
        bgo=2,      # background order

        # Stamp grid and substamp parameters
        nstampx=20,
        nstampy=20,
        nss=10,
        rss=5,

        # Gaussian basis and polynomial degrees
        ngauss=3,
        deg_fixe=[6, 4, 2],
        sigma_gauss=[0.9, 1.8, 3.6],

        # Robust-fitting controls
        ks=2.0,
        kfm=0.9,
        fitthresh=10.0,
        stat_sig=3.0,

        # Algorithm control and outputs
        force_convolve="t",
        normalize="i",
        verbose=0,

        output_file="diff.fits",
        noise_image_file="diff_noise.fits",
        mask_image_file="diff_mask.fits",
        convolved_image_file="diff_convolved.fits",
        sigma_image_file="diff_sigma.fits",
        stamp_region_file="stamp_regions.fits",
)
```

All tunable parameters from the original HOTPANTS C program are accessible
through the `HotpantsConfig` class; the wrapper aims to present a familiar
mapping to the original command-line flags in a Pythonic form.

Not implemented / unsupported parameters

The following parameters from the original HOTPANTS distribution are not
implemented or are fixed in this wrapper (see `hotpants/config.py`):

- `nregx`, `nregy` — the wrapper treats the entire image as a single region
- `scale_fitthresh` (`-sft`) — logic to dynamically scale `fitthresh` is not used.
- `min_frac_stamps` (`-nft`) — minimum-fraction-of-filled-stamps check is not used.
    (these are fixed to 1).

For the authoritative list and exact behavior, consult the `HotpantsConfig`
docstring in `hotpants/config.py`.

## Substamp inspection & diagnostics

The pipeline returns a set of substamp objects you can inspect to debug or
investigate local fits. Use `hp.find_stamps()` to retrieve candidate
substamps; each entry is a `Substamp` instance exposing useful attributes
documented in `hotpants/models.py` — for example:

- `id`, `stamp_group_id`, `x`, `y` — identifiers and coordinates
- `status` — the substamp processing status (e.g. FOUND, USED_IN_FINAL_FIT)
- `image_cutout`, `template_cutout`, `noise_variance_cutout` — pixel data
- `basis_vectors` — the design-matrix slices (basis functions convolved with
    the local image)
- `fit_results`, `local_kernel_solution` — local fit outputs
- `convolved_model_local`, `convolved_model_global` — models using local or
    final global solutions

You can iterate these substamps to visualize cutouts, inspect fit residuals,
or plot the local kernel basis. This is useful for diagnosing poor fits or
identifying problematic regions that should be masked or rejected before the
final global fit.

Example — show the six image-like cutouts available on each `Substamp`:

```python
import matplotlib.pyplot as plt

# get candidate substamps
template_substamps, image_substamps = hp.find_stamps()
# choose template substamps by default (change if you convolved the other way)
substamps = template_substamps

for s in substamps[:4]:  # show first 4 substamps as an example
    panels = [
        ("Science", getattr(s, "image_cutout", None)),
        ("Template", getattr(s, "template_cutout", None)),
        ("Noise variance", getattr(s, "noise_variance_cutout", None)),
        ("Local conv model", getattr(s, "convolved_model_local", None)),
        ("Global conv model", getattr(s, "convolved_model_global", None)),
        ("(unused)", None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    axes = axes.ravel()

    for ax, (title, img) in zip(axes, panels):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is None:
            ax.text(0.5, 0.5, "(missing)", ha="center", va="center")
        else:
            cmap = "RdBu_r" if "Noise" in title or "basis" in title else "viridis"
            ax.imshow(img, origin="lower", cmap=cmap)

    plt.suptitle(f"Substamp {s.id} — coords=({s.x}, {s.y})")
    plt.tight_layout()
    plt.show()
```

## Step-by-step execution

The wrapper shines when you want to run and inspect each stage separately:

```python
import numpy as np
import matplotlib.pyplot as plt
from hotpants import Hotpants, HotpantsConfig

# 1. Initialization (example using NumPy arrays)
template_image = np.load('template.npy')
science_image = np.load('science.npy')
hp = Hotpants(template_data=template_image, image_data=science_image)

# 2. Find substamps
template_substamps, image_substamps = hp.find_stamps()
print(f"Found {len(template_substamps)} potential template substamps.")

# 3. Select convolution direction
conv_direction = hp.fit_and_select_direction()
print(f"Best convolution direction: '{conv_direction}'")

# 4. Iterative kernel fit
kernel_solution, final_substamps = hp.iterative_fit_and_clip()
print(f"Global kernel solution found using {len(final_substamps)} substamps.")

# 5. Convolve and difference
diff, convolved, noise, mask = hp.convolve_and_difference()

# 6. Inspect the kernel at the image center
kernel_image = hp.visualize_kernel(at_coords=(1024, 1024))
plt.imshow(kernel_image, origin='lower', cmap='gray')
plt.title("Convolution Kernel at Image Center")
plt.show()
```



## Results

The `run_pipeline()` and `get_final_outputs()` methods return a dictionary
containing the following data products:

- `diff_image`: The final, masked difference image (float32).
- `convolved_image`: The convolved version of either the template or
    science image (float32).
- `noise_image`: The corresponding 1-sigma noise map for the difference
    image (float32).
- `output_mask`: The final integer bitmask indicating bad pixels (int32).
- `stats`: A dictionary of statistics calculated from the final difference
    and noise images.
- `conv_direction`: The chosen convolution direction (`'t'` or `'i'`). If not
    specified, the algorithm automatically determines the best direction.
- `kernel_solution`: The raw 1D array of coefficients for the global kernel
    solution.

## Contributing

Contributions are welcome. If you find a bug or have a suggestion, please
open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License — see the `LICENSE` file for
details.