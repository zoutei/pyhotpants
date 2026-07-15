
import numpy as np
from numba import njit, prange
from .utils import downsample_image

# =========================================================
# Local Separable Convolution (For Fitting)
# Matches alard.c: xy_conv_stamp
# =========================================================

@njit(cache=True)
def jit_xy_conv_stamp(image_patch, filter_x, filter_y, ren_vector=None):
    """
    Perform separable convolution on a small patch.
    Matches logic of xy_conv_stamp in alard.c (lines 221-270).
    """
    ny, nx = image_patch.shape
    fw_kernel = len(filter_x)
    hw_kernel = fw_kernel // 2
    
    res_ny = ny - 2 * hw_kernel
    res_nx = nx - 2 * hw_kernel
    
    conv_patch = np.zeros((res_ny, res_nx), dtype=np.float64)
    temp_patch = np.zeros((res_ny, nx), dtype=np.float64) 
    
    # Y-Pass
    for i in range(nx):
        for j in range(res_ny):
            val = 0.0
            for yc in range(fw_kernel):
                val += image_patch[j + yc, i] * filter_y[yc]
            temp_patch[j, i] = val
            
    # X-Pass
    for j in range(res_ny):
        for i in range(res_nx):
            val = 0.0
            for xc in range(fw_kernel):
                val += temp_patch[j, i + xc] * filter_x[xc]
            conv_patch[j, i] = val
            
    # Subtraction (ren logic)
    if ren_vector is not None:
        for j in range(res_ny):
            for i in range(res_nx):
                conv_patch[j, i] -= ren_vector[j, i]
                
    return conv_patch

# =========================================================
# Global Spatial Convolution (Block-Based)
# Matches alard.c: spatial_convolve / make_kernel
# =========================================================

@njit(cache=True)
def jit_make_kernel(kernel_sol, kc_step, hw_kernel, r_pix_x, r_pix_y, 
                    n_comp_ker, ker_order, kernel_vecs,
                    block_center_x, block_center_y):
    """
    Construct the local kernel at the block center.
    Matches alard.c make_kernel (C kernelSol layout with leading slot).
    """
    fw_kernel = 2 * hw_kernel + 1
    local_kernel = np.zeros((fw_kernel, fw_kernel), dtype=np.float64)
    kernel_coeffs = np.zeros(n_comp_ker, dtype=np.float64)

    xf = (block_center_x - 0.5 * r_pix_x) / (0.5 * r_pix_x)
    yf = (block_center_y - 0.5 * r_pix_y) / (0.5 * r_pix_y)

    # C: kernel_coeffs[0] = kernelSol[1]; k starts at 2 for bases 1..
    kernel_coeffs[0] = kernel_sol[1]
    k = 2
    for i1 in range(1, n_comp_ker):
        coeff = 0.0
        ax = 1.0
        for ix in range(ker_order + 1):
            ay = 1.0
            for iy in range(ker_order - ix + 1):
                coeff += kernel_sol[k] * ax * ay
                k += 1
                ay *= yf
            ax *= xf
        kernel_coeffs[i1] = coeff

    for i1 in range(n_comp_ker):
        c = kernel_coeffs[i1]
        kv = kernel_vecs[i1]
        for y in range(fw_kernel):
            for x in range(fw_kernel):
                local_kernel[y, x] += c * kv[y, x]

    return local_kernel


@njit(cache=True, parallel=True)
def jit_spatial_convolve(image, kernel_sol, variance, mask,
                         kc_step, hw_kernel,
                         n_comp_ker, ker_order, kernel_vecs,
                         convolve_variance=False, ker_frac_mask=0.0,
                         oversample=1):
    """
    Block-based spatial convolution matching spatial_convolve in alard.c.

    Critical C behaviors replicated here:
    - Output mask starts at 0 (C allocate fresh buffer); only interior pixels written
    - Pixel loops start at hwKernel (border left zeroed)
    - Kernel is applied with jk = j - jc + hw (180° flip)
    - Unmasked kernel mass uses FLAG_INPUT_ISBAD (0x80), not mask==0
    - Bad convolution sets FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV
    - Acceptable masked neighborhood sets FLAG_OK_CONV
    """
    FLAG_INPUT_ISBAD = 0x80
    FLAG_BAD_CONV = 0x10
    FLAG_OK_CONV = 0x40
    FLAG_OUTPUT_ISBAD = 0x8000

    ny, nx = image.shape

    conv_image = np.zeros((ny, nx), dtype=np.float32)
    out_variance = np.zeros((ny, nx), dtype=np.float32)
    # Match C py_apply_kernel: fresh zero output mask (not a copy of input)
    out_mask = np.zeros((ny, nx), dtype=np.int32)

    # C: nsteps = ceil(size / kcStep); j0 = step*kcStep + hwKernel
    n_blocks_y = (ny + kc_step - 1) // kc_step
    n_blocks_x = (nx + kc_step - 1) // kc_step

    for by in prange(n_blocks_y):
        j0 = by * kc_step + hw_kernel
        if j0 >= ny - hw_kernel:
            continue

        for bx in range(n_blocks_x):
            i0 = bx * kc_step + hw_kernel
            if i0 >= nx - hw_kernel:
                continue

            r_pix_x = float(nx) / oversample
            r_pix_y = float(ny) / oversample
            cx = (i0 + hw_kernel) / float(oversample)
            cy = (j0 + hw_kernel) / float(oversample)
            local_kernel = jit_make_kernel(
                kernel_sol, kc_step, hw_kernel, r_pix_x, r_pix_y,
                n_comp_ker, ker_order, kernel_vecs,
                cx, cy,
            )

            for j2 in range(kc_step):
                j = j0 + j2
                if j >= ny - hw_kernel:
                    break

                for i2 in range(kc_step):
                    i = i0 + i2
                    if i >= nx - hw_kernel:
                        break

                    q = 0.0
                    qv = 0.0
                    aks = 0.0
                    uks = 0.0
                    mbit = 0

                    for jc in range(j - hw_kernel, j + hw_kernel + 1):
                        # C: jk = j - jc + hwKernel  (flipped)
                        jk = j - jc + hw_kernel
                        for ic in range(i - hw_kernel, i + hw_kernel + 1):
                            ik = i - ic + hw_kernel
                            kk = local_kernel[jk, ik]

                            q += image[jc, ic] * kk
                            if convolve_variance:
                                qv += variance[jc, ic] * kk
                            else:
                                qv += variance[jc, ic] * kk * kk

                            pix_mask = mask[jc, ic]
                            mbit |= pix_mask
                            aks += abs(kk)
                            if (pix_mask & FLAG_INPUT_ISBAD) == 0:
                                uks += abs(kk)

                    conv_image[j, i] = q
                    out_variance[j, i] = qv

                    # Match C: mRData[ni] |= cMask[ni]; then OUTPUT_ISBAD / OK_CONV / BAD_CONV
                    current_mask = mask[j, i]
                    if (mask[j, i] & FLAG_INPUT_ISBAD) != 0:
                        current_mask |= FLAG_OUTPUT_ISBAD
                    if mbit != 0:
                        if aks > 0.0 and (uks / aks) < ker_frac_mask:
                            current_mask |= (FLAG_OUTPUT_ISBAD | FLAG_BAD_CONV)
                        else:
                            current_mask |= FLAG_OK_CONV
                    out_mask[j, i] = current_mask

    return conv_image, out_variance, out_mask


@njit(cache=True, parallel=True)
def jit_get_background(kernel_sol, bg_order, n_comp_ker, ker_order, r_pix_x, r_pix_y, ny, nx):
    """
    Calculate background image from solution.
    Matches get_background (C kernelSol layout with leading slot).

    Requires kernel_sol length >= ncompBG + 1 + n_bg_vectors.
    """
    background = np.zeros((ny, nx), dtype=np.float64)
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    n_bg = (bg_order + 1) * (bg_order + 2) // 2
    # C: ncompBG = (nCompKer-1)*nComp + 1; uses kernelSol[ncompBG + k] for k=1..n_bg
    ncomp_bg = (n_comp_ker - 1) * n_spatial + 1
    n_needed = ncomp_bg + 1 + n_bg
    n_sol = kernel_sol.shape[0]
    if n_sol < n_needed:
        # Leave zeros rather than indexing out of range (Numba has no exceptions here).
        return background

    for y in prange(ny):
        yf = (y - 0.5 * r_pix_y) / (0.5 * r_pix_y)
        for x in range(nx):
            xf = (x - 0.5 * r_pix_x) / (0.5 * r_pix_x)
            val = 0.0
            k = 1
            ax = 1.0
            for i in range(bg_order + 1):
                ay = 1.0
                for j in range(bg_order - i + 1):
                    val += kernel_sol[ncomp_bg + k] * ax * ay
                    k += 1
                    ay *= yf
                ax *= xf
            background[y, x] = val

    return background

@njit(cache=True)
def jit_convolve_patch(image, kernel):
    """
    2D patch convolution matching alard.c xy_conv_stamp filter orientation.

    C indexes filters as filter[hwKernel - offset], i.e. a 180° flip / true
    convolution. Even kernels are unchanged; odd kernels pick up the correct sign.
    """
    input_h, input_w = image.shape
    kh, kw = kernel.shape
    out_h = input_h - kh + 1
    out_w = input_w - kw + 1

    output = np.zeros((out_h, out_w))

    for y in range(out_h):
        for x in range(out_w):
            val = 0.0
            for ky in range(kh):
                for kx in range(kw):
                    val += image[y + ky, x + kx] * kernel[kh - 1 - ky, kw - 1 - kx]
            output[y, x] = val
    return output

def apply_kernel(image, kernel_sol, variance, mask, config, kernel_vecs, oversample=1):
    """
    High-level wrapper for spatial_convolve + background.
    Returns: Convolved, Background, Output Variance, Output Mask

    If oversample > 1, image is High-Res Template.
    Convolution is done in HR, then downsampled.
    Background is calculated in LR.
    """
    image = np.ascontiguousarray(image, dtype=np.float32)
    variance = np.ascontiguousarray(variance, dtype=np.float32)
    kernel_sol = np.ascontiguousarray(kernel_sol, dtype=np.float64)
    kernel_vecs = np.ascontiguousarray(kernel_vecs, dtype=np.float64)

    # Mask must be int bitflags like C mRData. Bool masks are coerced to 0/1.
    if mask is None:
        mask = np.zeros(image.shape, dtype=np.int32)
    else:
        mask = np.ascontiguousarray(mask)
        if mask.dtype == np.bool_ or mask.dtype == bool:
            mask = mask.astype(np.int32)
        else:
            mask = mask.astype(np.int32, copy=False)

    if image.shape != variance.shape or image.shape != mask.shape:
        raise ValueError(
            f"apply_kernel shape mismatch: image={image.shape}, "
            f"variance={variance.shape}, mask={mask.shape}"
        )

    h_b, w_b = kernel_vecs[0].shape
    hw_kernel_hr = w_b // 2

    # C default: kcStep = fwKernel
    fw_kernel_lr = 2 * config.rkernel + 1
    kc_step_lr = int(getattr(config, "kc_step", fw_kernel_lr) or fw_kernel_lr)
    kc_step_hr = kc_step_lr * oversample

    ker_order = config.ko
    n_basis = kernel_vecs.shape[0]
    ncomp_ker_layout = int(getattr(config, "ncomp_ker", n_basis))
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    n_bg = (config.bgo + 1) * (config.bgo + 2) // 2
    if config.bgo < 0:
        n_bg = 0
    # C layout: background index uses config.ncomp_ker, not len(kernel_vecs)
    n_needed = (ncomp_ker_layout - 1) * n_spatial + 1 + 1 + n_bg
    if kernel_sol.shape[0] < n_needed:
        raise ValueError(
            f"kernel_sol length {kernel_sol.shape[0]} too short for background "
            f"indexing (need >= {n_needed}). Ensure C-layout packing "
            f"(n_comp_total+1) is used."
        )

    convolve_variance = bool(getattr(config, "conv_var", False))
    ker_frac_mask = float(getattr(config, "kfm", 0.99))

    conv_hr, var_hr, mask_out_hr = jit_spatial_convolve(
        image, kernel_sol, variance, mask,
        kc_step_hr, hw_kernel_hr,
        n_basis, ker_order, kernel_vecs,
        convolve_variance, ker_frac_mask,
        oversample,
    )

    if oversample > 1:
        conv_lr = downsample_image(conv_hr, oversample)
        var_lr = downsample_image(var_hr, oversample)
        # OR-reduce mask blocks
        ny_hr, nx_hr = mask_out_hr.shape
        ny_lr, nx_lr = ny_hr // oversample, nx_hr // oversample
        mask_out_lr = np.zeros((ny_lr, nx_lr), dtype=np.int32)
        for oy in range(oversample):
            for ox in range(oversample):
                mask_out_lr |= mask_out_hr[oy::oversample, ox::oversample][:ny_lr, :nx_lr]
    else:
        conv_lr = conv_hr
        var_lr = var_hr
        mask_out_lr = mask_out_hr

    ny_lr, nx_lr = conv_lr.shape
    bg_lr = jit_get_background(
        kernel_sol, config.bgo, ncomp_ker_layout, ker_order,
        float(nx_lr), float(ny_lr),
        ny_lr, nx_lr,
    ).astype(np.float32)

    # Match C py_apply_kernel: do not force FLAG_OUTPUT_ISBAD on the hwKernel
    # border here — C leaves unvisited border pixels as 0 in the fresh mask.

    return conv_lr, bg_lr, var_lr, mask_out_lr
