# HOTPANTS Pipeline Documentation

This document provides a comprehensive, step-by-step analysis of the HOTPANTS (High Order Transform of PSF ANd Template Subtraction) pipeline. It details the logical flow, mathematical basis, and implementation details for both the original C extension ([hotpants_ext](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c#1216-1248) / `src`) and the Pure Python implementation (`pure`).

---

## 1. Pipeline Overview

The HOTPANTS pipeline performs Difference Image Analysis (DIA) using the Alard & Lupton (1998) algorithm. The core idea is to spatially match the Point Spread Function (PSF) and photometric scale of a Reference image (Template) to a Science image (Image) before subtraction.

**Equation:**
$$ \text{Image}(x,y) \approx [\text{Kernel}(u,v) \otimes \text{Template}(x-u, y-v)] + \text{Background}(x,y) $$

The pipeline determines the optimal **Kernel** and **Background** coefficients by minimizing the squared difference between the Image and the Convolved Template.

---

## 2. Initialization & Configuration

Before processing, the system initializes configuration parameters that define the complexity of the kernel and spatial variations.

*   **Key Parameters:**
    *   `rkernel`: Kernel half-width (radius).
    *   `ngauss`, `sigma_gauss`, `deg_fixe`: Definition of Gaussian basis functions.
        *   `ngauss`: Number of Gaussian components (e.g., 3: defined by sigma values like 0.7, 1.5, 3.0).
        *   `deg_fixe`: Polynomial degree modification for each Gaussian (e.g., 6, 4, 2).
    *   `ko` (Kernel Order): Degree of spatial polynomial variation for kernel coefficients.
    *   `bgo` (Background Order): Degree of spatial polynomial for the differential background.
    *   `nsx`, `nsy`: Number of regions for stamp selection.

---

## 3. Detailed Pipeline Steps

### Step 1: Pre-Processing & Masking

**Goal:** Identify bad pixels (saturated, low value, or user-masked) to exclude them from calculations.

**Process:**
1.  **Load Images**: Read Template and Image FITS data.
2.  **Generate Masks**:
    *   Mask pixels $< \text{tlthresh}$ or $> \text{tuthresh}$ (Template).
    *   Mask pixels $< \text{ilthresh}$ or $> \text{iuthresh}$ (Image).
    *   Merge with user-supplied masks ([t_mask](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c#71-172), `i_mask`).
    *   Propagate masks: Often broadened by the kernel radius to avoid contamination from bad neighbors.

**Implementation Mapping:**
| Action | Python ([core.py](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py) / `pure`) | C ([hotpants_ext.c](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c) / [main.c](file:///Users/kshukawa/Documents/pyhotpants/src/main.c)) |
| :--- | :--- | :--- |
| Entry | `Hotpants.__init__` | [main()](file:///Users/kshukawa/Documents/pyhotpants/src/maskim.c#16-138) |
| Logic | `pure.utils.mask_pixels` | [makeInputMask](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#1215-1233) (in `hotpants.c`) called via [py_make_input_mask](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c#71-172) |
| Details | Uses vectorized `numpy` comparisons. | Loops over pixels, sets bit flags (`FLAG_T_BAD`, `FLAG_I_BAD`). |

---

### Step 2: Stamp Selection ([find_stamps](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py#282-376))

**Goal:** Select high-quality, isolated stars (substamps) to maximize the signal-to-noise ratio for kernel fitting.

**Process:**
1.  **Grid Partitioning**: The image is divided into a grid of `nsx` $\times$ `nsy` regions.
2.  **Candidate Search**: In each region, find the brightest pixel (`dmax`) that:
    *   Is not masked.
    *   Is not saturated.
    *   Has sufficient S/N (Peak - Sky) / Noise > `fitthresh`.
    *   Is not too close to the edge.
3.  **Centroid Verification**: Fits a localized gaussian or performs moment analysis to ensure the peak is "star-like" and not a cosmic ray or artifact.
4.  **Selection**: Use the top `nKSStamps` candidates per region.

**Implementation Mapping:**
| Action | Python ([core.py](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py) / `pure`) | C ([hotpants_ext.c](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c) / `src`) |
| :--- | :--- | :--- |
| Entry | `Hotpants.find_stamps()` | [py_find_stamps](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c#210-374) $\to$ [buildStamps](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#178-353) (in [functions.c](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c)) |
| Logic | `pure.utils.find_stamps` | [buildStamps](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#178-353) iterates regions, calls [getPsfCenters](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#495-708). |
| Algorithm | `scikit-image` peak finding or custom windowed max search. | [getPsfCenters](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#495-708): Iterative search masking out previous peaks to find next brightest. |
| Differences | Python implementation may use `scipy.ndimage` for efficiency. | C implementation calculates centroids precisely using [checkPsfCenter](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#436-494) to validate shape. |

---

### Step 3: Initial Local Fitting & Direction Selection ([fit_and_select_direction](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py#377-529))

**Goal:** Determine which image has the better "PSF" (usually the better seeing) to serve as the reference for convolution. Also rejects bad stamps early.

**Process:**
1.  **Basis Construction**: Generate kernel basis functions $K_n(u,v)$.
    *   Equation: $K_n(u,v) = e^{-(u^2+v^2)/2\sigma^2} u^i v^j$.
2.  **Extraction**: For each stamp:
    *   Extract [Image](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#1163-1179) patch (Data $D$).
    *   Extract `Template` patch (Reference $R$).
3.  **Convolution**: Convolve $R$ with every basis function $K_n$.
    *   $V_n = R \otimes K_n$.
4.  **Local Solve**: Solve $D \approx \sum a_n V_n + \text{Bg}$.
    *   Construct Matrix $M_{nm} = \sum_{pix} V_n V_m$.
    *   Construct Vector $b_n = \sum_{pix} V_n D$.
    *   Solve $Ma = b$ for coefficients $a$.
5.  **Figure of Merit (FOM)**: Calculate $\chi^2$ of the residuals ($D - Model$).
6.  **Direction Choice**: Compare total FOM for "Convolve Template" vs "Convolve Image". Choose the one with the *lower* residual $\chi^2$ (implies better fit is possible, theoretically convolving the better-seeing image to match the worse-seeing one is mathematically more stable).

**Implementation Mapping:**
| Action | Python ([core.py](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py) / `pure`) | C ([hotpants_ext.c](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c) / `src`) |
| :--- | :--- | :--- |
| Entry | `Hotpants.fit_and_select_direction()` | [py_fit_stamps_and_get_fom](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c#375-588) |
| Local Setup | `pure.fitting.fit_stamps_locally` | [fillStamp](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#41-124) (in [alard.c](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c)) |
| Convolution | `pure.convolution.jit_xy_conv_stamp` | [xy_conv_stamp](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#221-271) (in [alard.c](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c)) |
| Solver | `np.linalg.solve` | `ludcmp` / [lubksb](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1407-1427) (Numerical Recipes in C) |
| Rejection | `pure.fitting` 3-sigma clip on kernel sums. | [check_stamps](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#474-716) (in [alard.c](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c)) calculates kernel sums and performs sigma clipping. |

---

### Step 4: Global Kernel Fitting ([iterative_fit_and_clip](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py#530-656))

**Goal:** Derive a single, spatially varying kernel solution using all valid stamps.

**Process:**
1.  **Global Model**: The kernel coefficient $a_n$ for basis $n$ varies spatially:
    $$ a_n(x,y) = \sum_{k} c_{nk} x^{p_k} y^{q_k} $$
2.  **System Construction**:
    *   The unknowns are now $c_{nk}$ (Global Coefficients).
    *   The basis vectors effectively become $V_{nk}(u,v,x,y) = (R \otimes K_n) \times x^p y^q$.
    *   Matrix size is roughly $(N_{gauss} \times N_{deg} \times N_{spatial\_deg})^2$.
3.  **Iteration Loop (Sigma Clipping)**:
    1.  Build global matrix $A$ and vector $b$ summing over *all* active stamps.
    2.  Solve $Ax = b$ for global coeffs $c$.
    3.  **Check Residuals**: For every stamp, re-calculate the model using the new global solution.
    4.  Compute residual $\sigma$.
    5.  **Reject**: If a stamp's residual sigma deviates $>$ [ks](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1407-1427) (sigma threshold) from the population mean, flag it as BAD.
    6.  Repeat until convergence (no stamps rejected) or max iterations.

**Implementation Mapping:**
| Action | Python ([core.py](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py) / `pure`) | C ([hotpants_ext.c](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c) / `src`) |
| :--- | :--- | :--- |
| Entry | `Hotpants.iterative_fit_and_clip()` | [py_fit_kernel](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c#589-682) $\to$ [fitKernel](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#307-373) (in [alard.c](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c)) |
| Matrix Build| `pure.fitting.build_matrix_numba` | [build_matrix](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#717-840) (in [alard.c](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c)) |
| Logic | Loop in `pure.fitting.fit_kernel`. | Loop in [fitKernel](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#307-373) calling [check_again](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1017-1122) for rejection. |
| Difference | Python re-builds the matrix using Numba-optimized loops. | C uses pointers and pre-allocated `wxy` spatial weight arrays in [build_matrix](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#717-840). |

---

### Step 5: Convolution & Subtraction ([convolve_and_difference](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py#658-737))

**Goal:** Apply the derived spatially-varying kernel to the *entire* reference image and subtract it from the science image.

**Process:**
1.  **Block-Based Approximation**:
    *   The kernel $K(u,v,x,y)$ varies continuously across the image. Computing it for every pixel is computationally prohibitive.
    *   **Optimization**: The image is divided into small blocks of size `kc_step` $\times$ `kc_step` (default: usually related to kernel size).
    *   The kernel is computed *once* at the center of each block using the global coefficients $c_{nk}$.
    *   **Assumption**: The kernel is treated as *constant* within this block. There is no bilinear interpolation of the kernel coefficients per pixel within the block; the spatial variation is step-wise.

2.  **Pixel Convolution**:
    *   For a pixel $(x,y)$ within a block centered at $(x_c, y_c)$, the convolved value is:
        $$ I_{conv}(x,y) = \sum_{u,v} \text{Template}(x+u, y+v) \times K(u,v, x_c, y_c) $$
    *   **Mask Propagation**:
        *   The input mask is checked for every pixel in the kernel footprint.
        *   `mbit` accumulates the OR-ed mask values of all pixels under the kernel.
        *   **Kernel Fraction Check**: The algorithm calculates the sum of absolute kernel weights falling on *good* pixels (`uks`) vs *all* pixels (`aks`).
        *   If `uks / aks < kerFracMask` (default 0.99), the output pixel is flagged as `BAD_CONV`. This ensures that pixels near bad regions (but not strictly masked) don't get garbage values if they rely heavily on masked data.

3.  **Variance Propagation**:
    *   The variance of the convolved image is calculated simultaneously.
    *   **Standard Mode**: If `convolve_variance` is False (default for basic noise propagation):
        $$ \text{Var}_{conv}(x,y) = \sum_{u,v} \text{Var}_{template}(x+u, y+v) \times [K(u,v)]^2 $$
    *   **Convolve Variance Mode**: If `convolve_variance` is True:
        $$ \text{Var}_{conv}(x,y) = \sum_{u,v} \text{Var}_{template}(x+u, y+v) \times K(u,v) $$
        *(Note: This mode is mathematically unusual for independent pixel noise but might be intended for specific covariance handling).*

4.  **Background Addition**:
    *   The background polynomial $B(x,y)$ is evaluated for every pixel and added to the convolved template.
    *   $$ \text{Model}(x,y) = I_{conv}(x,y) + \sum_{k} c_{bg\_k} x^{p_k} y^{q_k} $$

5.  **Difference Calculation**:
    *   $$ \text{Diff}(x,y) = \text{Image}(x,y) - \text{Model}(x,y) $$
    *   Final Noise: $\sigma_{diff}^2 = \sigma_{image}^2 + \text{Var}_{conv}$.

**Implementation Mapping:**
| Action | Python ([core.py](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py) / `pure`) | C ([hotpants_ext.c](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c) / `src`) |
| :--- | :--- | :--- |
| Entry | `Hotpants.convolve_and_difference()` | [py_apply_kernel](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c#683-780) $\to$ [spatial_convolve](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1123-1219) |
| Algorithm | `pure.convolution.jit_spatial_convolve` (Numba Parallel). | [spatial_convolve](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1123-1219) (in [alard.c](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c)). |
| Optimization| Uses `prange` for parallel block processing. | Uses nested loops over blocks `j1, i1` then pixels `j2, i2`. |
| Variance | Implements both `K^2` and `K` variance modes. | Implements both modes in [spatial_convolve](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1123-1219). |
| Masking | Implements Logic: if `uks/aks < kfm` flag BAD. | Identical logic in C [spatial_convolve](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1123-1219). |

---

### Step 6: Final Statistics & Rejection Details

**Detailed Rejection Logic ([check_again](file:///Users/kshukawa/Documents/pyhotpants/src/alard.c#1017-1122) in C):**
*   After the global fit, the pipeline performs a robust check to ensure the kernel solution is valid across the image.
*   **Process**:
    1.  Iterates through all valid stamps.
    2.  For each stamp, it computes the **Residual Sigma**: $\sigma_{resid} = \text{std}(\text{Data} - \text{Model})$.
    3.  It calculates the global mean and standard deviation of these sigma values (`mean`, `stdev`).
    4.  **Sigma Clipping**: Any stamp with $\sigma_{resid} > \text{mean} + (\text{kerSigReject} \times \text{stdev})$ is flagged.
    5.  **Substamp Substitution**:
        *   Instead of just dropping the region, HOTPANTS attempts to "recover" the stamp.
        *   If a stamp is rejected, it increments `sscnt` (Substamp Count).
        *   This switches the stamp to the *next* brightest star identified in that grid region during the [find_stamps](file:///Users/kshukawa/Documents/pyhotpants/hotpants/core.py#282-376) step (where `niS` stamps were originally saved).
        *   It then *refits* (or marks for refit) using this new candidate star.
*   This feature ensures that a transient (like a supernova) or a cosmic ray landing on the primary stamp candidate doesn't leave a "hole" in the spatial constraints; the algorithm simply moves to the next best star in that patch.

**Other "Left Out" Details:**
*   **[maskim.c](file:///Users/kshukawa/Documents/pyhotpants/src/maskim.c)**: This is a standalone C utility included in the source but acts as a CLI wrapper around the [makeInputMask](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#1215-1233) logic. It allows users to pre-process masks from the command line outside the main pipeline.
*   **Pixel Value thresholds**: The `ilthresh`, `iuthresh`, etc., are strict cutoffs. Any pixel exceeding these in the input is flagged `BAD`. The mask is then "spread" (dilated) by `rKernel` to ensure convolution doesn't pull in these edge values.

---

## Technical Appendix: C vs. Python Nuances

*   **Memory Management**:
    *   **C**: Heavily relies on pre-allocated structures (`stamp_struct`, `hotpants_state_t`) and manual `malloc`/[free](file:///Users/kshukawa/Documents/pyhotpants/src/functions.c#1140-1162). [hotpants_ext.c](file:///Users/kshukawa/Documents/pyhotpants/hotpants/hotpants_ext.c) manages the lifespan of these structures within a Python Object wrapper.
    *   **Python**: Uses `numpy` arrays. The `pure` implementation uses [numba](file:///Users/kshukawa/Documents/pyhotpants/hotpants/pure/fitting.py#342-375) to achieve C-like performance by JIT-compiling the critical loops (matrix building, convolution).
*   **Linear Algebra**:
    *   **C**: Uses `ludcmp` (LU Decomposition) from Numerical Recipes. This is a robust but older standard.
    *   **Python**: Uses `numpy.linalg.solve` (LAPACK). This is generally faster and more stable, but differences in floating-point precision or solver strategy (e.g., pivoting) can lead to *tiny* (1e-10) differences in coefficients.
*   **Masking**:
    *   **C**: Uses bitmasks (int). Checks `bit & FLAG`.
    *   **Python**: Can use boolean masks or integer bitmasks. `pyhotpants` maintains compatibility by using the same integer bitmask definitions.

