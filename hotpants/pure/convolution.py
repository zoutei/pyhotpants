
import numpy as np
from numba import njit, prange

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
    Matches alard.c: make_kernel
    """
    fw_kernel = 2 * hw_kernel + 1
    local_kernel = np.zeros((fw_kernel, fw_kernel), dtype=np.float64)
    kernel_coeffs = np.zeros(n_comp_ker, dtype=np.float64)
    
    xf = (block_center_x - 0.5 * r_pix_x) / (0.5 * r_pix_x)
    yf = (block_center_y - 0.5 * r_pix_y) / (0.5 * r_pix_y)
    
    # Basis 0
    kernel_coeffs[0] = kernel_sol[0]
    
    # Basis > 0
    k = 1 # Index into kernel_sol
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
                         convolve_variance=False, ker_frac_mask=0.0):
    """
    Block-based spatial convolution matching spatial_convolve in alard.c.
    """
    ny, nx = image.shape
    
    conv_image = np.zeros_like(image)
    out_variance = np.zeros_like(variance)
    out_mask = np.zeros_like(mask)
    
    # Parallelize over ROWS (y)
    # Numba prange requires constant step, so we iterate over block index
    n_blocks_y = (ny - hw_kernel + kc_step - 1) // kc_step
    
    for by in prange(n_blocks_y):
        j0 = by * kc_step
        # Check bound explicitly in case ceiling division overshoots
        if j0 >= ny - hw_kernel:
             continue
             
        for i0 in range(0, nx - hw_kernel, kc_step):
            
            # 1. Update Kernel for this block
            local_kernel = jit_make_kernel(kernel_sol, kc_step, hw_kernel, nx, ny,
                                           n_comp_ker, ker_order, kernel_vecs,
                                           i0 + hw_kernel, j0 + hw_kernel)
            
            # 2. Convolve pixels in this block
            for j2 in range(kc_step):
                j = j0 + j2
                if j >= ny - hw_kernel: break
                    
                for i2 in range(kc_step):
                    i = i0 + i2
                    if i >= nx - hw_kernel: break
                    
                    q = 0.0
                    qv = 0.0
                    aks = 0.0 
                    uks = 0.0 
                    mbit = 0
                    
                    for jc in range(j - hw_kernel, j + hw_kernel + 1):
                        jk = j - jc + hw_kernel 
                        for ic in range(i - hw_kernel, i + hw_kernel + 1):
                            ik = i - ic + hw_kernel 
                            kk = local_kernel[jk, ik]
                            
                            val = image[jc, ic]
                            q += val * kk
                            
                            if convolve_variance:
                                qv += variance[jc, ic] * kk
                            else:
                                qv += variance[jc, ic] * kk * kk
                                
                            pix_mask = mask[jc, ic]
                            mbit |= pix_mask
                            
                            aks += abs(kk)
                            if pix_mask == 0:
                                uks += abs(kk)
                                
                    conv_image[j, i] = q
                    out_variance[j, i] = qv
                    
                    current_mask = mask[j, i] | mbit
                    if mbit: 
                         if aks > 0 and (uks / aks) < ker_frac_mask:
                             current_mask |= 2 # FLAG_BAD_CONV
                    out_mask[j, i] = current_mask
                    
    return conv_image, out_variance, out_mask

@njit(cache=True, parallel=True)
def jit_get_background(kernel_sol, bg_order, n_comp_ker, ker_order, r_pix_x, r_pix_y, ny, nx):
    """
    Calculate background image from solution.
    Matches get_background logic loop over image.
    """
    background = np.zeros((ny, nx), dtype=np.float64)
    
    # Index of background coeffs start
    # 1 (const) + (n_comp - 1) * n_spatial
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    bg_start = 1 + (n_comp_ker - 1) * n_spatial
    
    # Pre-calculate BG coeffs in a structure or just direct loop
    # Optimization: The Background is a global polynomial. 
    # B(x,y) = sum C_ij * x^i * y^j
    # We can compute this efficiently.
    
    # Loop pixels (parallel)
    for y in prange(ny):
        yf = (y - 0.5 * r_pix_y) / (0.5 * r_pix_y)
        
        for x in range(nx):
            xf = (x - 0.5 * r_pix_x) / (0.5 * r_pix_x)
            
            val = 0.0
            k = 0 # index into BG part of kernel_sol
            
            ax = 1.0
            for i in range(bg_order + 1):
                ay = 1.0
                for j in range(bg_order - i + 1):
                    coeff = kernel_sol[bg_start + k]
                    val += coeff * ax * ay
                    k += 1
                    ay *= yf
                ax *= xf
            background[y, x] = val
            
    return background

@njit(cache=True)
def jit_convolve_patch(image, kernel):
    """
    Simple 2D convolution for patches.
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
                    val += image[y+ky, x+kx] * kernel[ky, kx]
            output[y, x] = val
    return output

def apply_kernel(image, kernel_sol, variance, mask, config, kernel_vecs):
    """
    High-level wrapper for spatial_convolve + background.
    Returns: Convolved + Background, Output Variance, Output Mask
    """
    hw_kernel = config.rkernel
    # Calculate kc_step if not present or default
    kc_step = getattr(config, 'kc_step', 2 * hw_kernel + 1)
    
    ker_order = config.ko
    n_comp_ker = kernel_vecs.shape[0]
    
    # Check for correct attribute names
    convolve_variance = getattr(config, 'conv_var', False)
    ker_frac_mask = getattr(config, 'kfm', 0.99)
    
    # 1. Convolve
    conv, var, mask_out = jit_spatial_convolve(
        image, kernel_sol, variance, mask,
        kc_step, hw_kernel,
        n_comp_ker, ker_order, kernel_vecs,
        convolve_variance, ker_frac_mask
    )
    
    # 2. Add Background
    bg = jit_get_background(
        kernel_sol, config.bgo, n_comp_ker, ker_order,
        float(image.shape[1]), float(image.shape[0]),
        image.shape[0], image.shape[1]
    )
    
    total_model = conv + bg
    
    return total_model, var, mask_out
