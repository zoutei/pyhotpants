
import numpy as np
from numba import njit, prange
import scipy.linalg
from .convolution import jit_xy_conv_stamp, jit_make_kernel, jit_convolve_patch
from .kernel import get_spatial_polynomials
from .utils import downsample_image


def _tikhonov_solve(M, b, lambda_reg=0.0):
    """
    Solve (M + λ·scale·I) x = b.

    λ is relative to the mean positive diagonal of M so the same default
    works across oversample factors. Falls back to lstsq if LU fails.
    """
    M = np.asarray(M, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = M.shape[0]
    M_reg = M.copy()

    if lambda_reg and lambda_reg > 0:
        diag = np.diag(M_reg).copy()
        pos = diag[diag > 0]
        scale = float(np.mean(pos)) if pos.size else 1.0
        M_reg += lambda_reg * scale * np.eye(n)

    try:
        lu, piv = scipy.linalg.lu_factor(M_reg)
        return scipy.linalg.lu_solve((lu, piv), b)
    except Exception:
        diag_A = np.diag(M_reg).copy()
        diag_A[diag_A <= 1e-20] = 1.0
        scale_vec = 1.0 / np.sqrt(diag_A)
        A_scaled = M_reg * scale_vec[:, None] * scale_vec[None, :]
        b_scaled = b * scale_vec
        x_scaled, _, _, _ = np.linalg.lstsq(A_scaled, b_scaled, rcond=None)
        return x_scaled * scale_vec


class Stamp:
    def __init__(self, x, y, data=None, orig_idx=None):
        self.x = int(x)
        self.y = int(y)
        self.orig_idx = orig_idx # Track original index from input list
        self.vectors = None # (n_vecs, n_pix)
        self.basis_vectors = None  # (n_basis, h, w)
        self.substamp = None  # The DATA (I) stamp (n_pix,)
        self.image_cutout = None
        self.template_cutout = None
        self.weights = None # (n_spatial,)
        self.norm = 0.0 # Kernel Sum
        self.chi2 = 0.0
        self.diff = 0.0 # Rejection metric
        self.local_solution = None
        self.convolved_model_local = None
        self.ignore = False
        self.residuals = None
        
        # Iteration state (C stamp_struct sscnt / nss / xss / yss)
        self.sscnt = 0
        self.nss = 1
        self.substamp_coords = [(int(x), int(y))]

        # Connected-region stamp fields (optional)
        self.spatial_x = float(x)
        self.spatial_y = float(y)
        self.region_id = None
        self.region_weight = 1.0
        self.ys = None  # pixel coords of irregular mask
        self.xs = None
        self.npix = None
        self.noise_pix = None  # per-pixel noise variance aligned with substamp

def populate_stamp_vectors(stamp, template, image, config, kernel_vecs, oversample=1):
    """
    Populates the stamp with data, template, and basis vectors.
    Returns True if successful, False if stamp is out of bounds or invalid.

    Stamp (x, y) are always science-image (LR) coordinates. When oversample>1,
    `template` is HR and the patch is taken at y0*oversample (etc.), then
    basis convolutions are downsampled to the LR stamp grid.
    """
    half_stamp = config.rss
    # Determine kernel radius in HR pixels
    h_basis_hr, w_basis_hr = kernel_vecs[0].shape
    half_r_hr = w_basis_hr // 2
    
    n_comp_ker = len(kernel_vecs)
    bg_order = config.bgo
    
    # 1. Extract Data (Image)
    y, x = stamp.y, stamp.x
    y0 = int(y - half_stamp)
    y1 = int(y + half_stamp + 1)
    x0 = int(x - half_stamp)
    x1 = int(x + half_stamp + 1)
    
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return False
        
    data_stamp = image[y0:y1, x0:x1]
    if np.any(np.isnan(data_stamp)):
        return False # Reject stamps with NaNs in data
        
    stamp.substamp = data_stamp.flatten()
    stamp.image_cutout = data_stamp.copy()
    
    # 2. Extract Template Patch
    # HR extraction logic
    y0_t_hr = int(y0 * oversample - half_r_hr)
    y1_t_hr = int(y1 * oversample + half_r_hr)
    x0_t_hr = int(x0 * oversample - half_r_hr)
    x1_t_hr = int(x1 * oversample + half_r_hr)
    
    if y0_t_hr < 0 or x0_t_hr < 0 or y1_t_hr > template.shape[0] or x1_t_hr > template.shape[1]:
        return False
        
    template_patch = template[y0_t_hr:y1_t_hr, x0_t_hr:x1_t_hr]
    if np.any(np.isnan(template_patch)):
        return False
        
    stamp.template_cutout = template_patch.copy()
    
    # 3. generate Basis Vectors
    vectors = []
    basis_cutouts = []
    
    for k in range(n_comp_ker):
        basis_k = kernel_vecs[k]
        conv_res_hr = jit_convolve_patch(template_patch, basis_k)
        
        if oversample > 1:
            conv_res = downsample_image(conv_res_hr, oversample)
        else:
            conv_res = conv_res_hr
            
        # Verify shape
        if conv_res.shape != data_stamp.shape:
            # Should not happen if logic is correct
            return False
            
        v = conv_res.flatten()
        vectors.append(v)
        basis_cutouts.append(conv_res)
        
    # 4. Background Vectors — order must match fillStamp / get_background
    # ax*=xf outer, ay*=yf inner (NOT total-degree order).
    nx_glob = template.shape[1] / oversample
    ny_glob = template.shape[0] / oversample

    gy, gx = np.indices(data_stamp.shape)
    gy = gy + y0
    gx = gx + x0

    ny_norm = (gy - 0.5 * ny_glob) / (0.5 * ny_glob)
    nx_norm = (gx - 0.5 * nx_glob) / (0.5 * nx_glob)

    ax = np.ones_like(nx_norm)
    for idegx in range(bg_order + 1):
        ay = np.ones_like(ny_norm)
        for idegy in range(bg_order - idegx + 1):
            vectors.append((ax * ay).flatten())
            ay = ay * ny_norm
        ax = ax * nx_norm

    stamp.vectors = np.array(vectors)
    stamp.basis_vectors = np.array(basis_cutouts)
    
    if np.any(np.isnan(stamp.vectors)):
        return False
        
    return True


def populate_region_vectors(
    stamp,
    template,
    image,
    labels,
    region,
    config,
    kernel_vecs,
    oversample=1,
    input_mask=None,
    noise_sq=None,
):
    """
    Fill an irregular connected-region stamp: science = all good pixels with
    labels==region.id; template bases convolved on bbox dilated by rkernel,
    then gathered at those pixels.
    """
    from .regions import effective_min_npix
    from .utils import FLAG_INPUT_ISBAD

    F = int(oversample)
    half_r_hr = kernel_vecs[0].shape[1] // 2
    n_comp_ker = len(kernel_vecs)
    bg_order = config.bgo
    min_npix = effective_min_npix(config)

    ys, xs = np.where(labels == region.id)
    if ys.size == 0:
        return False

    good = np.isfinite(image[ys, xs])
    if input_mask is not None:
        good &= (input_mask[ys, xs].astype(np.int32) & FLAG_INPUT_ISBAD) == 0
    ys, xs = ys[good], xs[good]
    if ys.size < min_npix:
        return False

    stamp.ys = ys
    stamp.xs = xs
    stamp.npix = int(ys.size)
    stamp.substamp = image[ys, xs].astype(np.float64)
    stamp.region_id = int(region.id)
    stamp.spatial_x = float(region.x_flux)
    stamp.spatial_y = float(region.y_flux)
    stamp.x = int(round(region.x_flux))
    stamp.y = int(round(region.y_flux))

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    # Visualization cutout (dense bbox, NaN outside mask)
    cut = np.full((y1 - y0, x1 - x0), np.nan, dtype=np.float32)
    cut[ys - y0, xs - x0] = image[ys, xs]
    stamp.image_cutout = cut

    # Template patch: bbox dilated by half_r so jit_convolve_patch output
    # aligns with [y0:y1, x0:x1] (same as populate_stamp_vectors).
    y0_t = int(y0 * F - half_r_hr)
    y1_t = int(y1 * F + half_r_hr)
    x0_t = int(x0 * F - half_r_hr)
    x1_t = int(x1 * F + half_r_hr)

    if y0_t < 0 or x0_t < 0 or y1_t > template.shape[0] or x1_t > template.shape[1]:
        return False

    template_patch = template[y0_t:y1_t, x0_t:x1_t]
    if np.any(np.isnan(template_patch)):
        return False
    stamp.template_cutout = template_patch.copy()

    vectors = []
    for k in range(n_comp_ker):
        conv_hr = jit_convolve_patch(template_patch, kernel_vecs[k])
        if F > 1:
            conv_lr = downsample_image(conv_hr.astype(np.float64), F)
            # LR output should match [y0:y1, x0:x1]
            if conv_lr.shape != (y1 - y0, x1 - x0):
                # tolerate 1-pixel rounding: gather via absolute mapping
                yl = ys - y0
                xl = xs - x0
                if (
                    yl.min() < 0
                    or xl.min() < 0
                    or yl.max() >= conv_lr.shape[0]
                    or xl.max() >= conv_lr.shape[1]
                ):
                    return False
                gathered = conv_lr[yl, xl]
            else:
                gathered = conv_lr[ys - y0, xs - x0]
        else:
            # OS=1: conv shape == (y1-y0, x1-x0)
            if conv_hr.shape != (y1 - y0, x1 - x0):
                return False
            gathered = conv_hr[ys - y0, xs - x0]
        vectors.append(np.asarray(gathered, dtype=np.float64).ravel())

    # Background at global pixel coords
    nx_glob = template.shape[1] / F
    ny_glob = template.shape[0] / F
    ny_norm = (ys.astype(np.float64) - 0.5 * ny_glob) / (0.5 * ny_glob)
    nx_norm = (xs.astype(np.float64) - 0.5 * nx_glob) / (0.5 * nx_glob)
    ax = np.ones_like(nx_norm)
    for idegx in range(bg_order + 1):
        ay = np.ones_like(ny_norm)
        for idegy in range(bg_order - idegx + 1):
            vectors.append(ax * ay)
            ay = ay * ny_norm
        ax = ax * nx_norm

    stamp.vectors = np.array(vectors, dtype=np.float64)
    stamp.basis_vectors = None
    if noise_sq is not None:
        stamp.noise_pix = noise_sq[ys, xs].astype(np.float64)
    if np.any(~np.isfinite(stamp.vectors)):
        return False
    return True


def _sigma_clip_mean_stdev(data, stat_sig=3.0, maxiter=10):
    """
    Match functions.c sigma_clip: iterative reject |x-mean|/stdev > stat_sig,
    sample stdev with (n-1). Returns (mean, stdev).
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    if n == 0:
        return 0.0, 1e30
    smask = np.zeros(n, dtype=bool)
    mean = 0.0
    stdev = 1e30
    ncnt = n
    cnt = -1
    for _ in range(maxiter):
        if ncnt == cnt:
            break
        cnt = ncnt
        good = ~smask
        ncnt = int(np.sum(good))
        if ncnt == 0:
            return 0.0, 1e30
        vals = data[good]
        mean = float(np.mean(vals))
        if ncnt > 1:
            stdev = float(np.sqrt(np.sum((vals - mean) ** 2) / (ncnt - 1)))
        else:
            return mean, 1e30
        if stdev == 0.0:
            break
        # Reject high and low outliers (same as C)
        for i in range(n):
            if not smask[i] and abs(data[i] - mean) / stdev > stat_sig:
                smask[i] = True
        ncnt = int(np.sum(~smask))
    return mean, stdev


def fit_stamps_locally(
    stamps,
    template,
    image,
    config,
    kernel_vecs,
    oversample=1,
    region_map=None,
    input_mask=None,
    noise_sq=None,
):
    """
    Perform initial local fit on stamps to reject outliers.

    Matches alard.c check_stamps kernel-sum logic:
    - Local solve uses nCompKer + 1 (bases + constant bg)
    - norm = first coefficient (check_vec[1]), not sum of all basis amps
    - Reject when |norm - kmean| / kstdev >= kerSigReject (config.ks)

    If region_map is set, each stamp is filled via populate_region_vectors.
    """
    valid_stamps = []
    n_comp_ker = len(kernel_vecs)
    ker_sig_reject = float(getattr(config, "ks", 2.0))
    stat_sig = float(getattr(config, "stat_sig", 3.0))
    regions_by_id = {r.id: r for r in region_map.regions} if region_map is not None else {}

    for idx, s_obj in enumerate(stamps):
        stamp = Stamp(s_obj.x, s_obj.y, orig_idx=idx)
        rid = getattr(s_obj, "region_id", None)
        if rid is None and region_map is not None:
            rid = getattr(s_obj, "stamp_group_id", None)

        if region_map is not None and rid is not None and int(rid) in regions_by_id:
            stamp.region_id = int(rid)
            ok = populate_region_vectors(
                stamp,
                template,
                image,
                region_map.labels,
                regions_by_id[int(rid)],
                config,
                kernel_vecs,
                oversample,
                input_mask=input_mask,
                noise_sq=noise_sq,
            )
            if not ok:
                continue
        else:
            if not populate_stamp_vectors(stamp, template, image, config, kernel_vecs, oversample):
                continue

        # Local Fit: Kernel Basis + Constant Background (C nComps = nCompKer + 1)
        n_fit = n_comp_ker + 1

        vectors_fit = stamp.vectors[:n_fit]
        M_fit = vectors_fit @ vectors_fit.T
        b_fit = vectors_fit @ stamp.substamp

        try:
            coeffs_fit = np.linalg.solve(M_fit, b_fit)

            coeffs = np.zeros(len(stamp.vectors))
            coeffs[:n_fit] = coeffs_fit

            # C: sum = check_vec[1]  (basis-0 amplitude only)
            stamp.norm = float(coeffs_fit[0])
            stamp.local_solution = coeffs[:n_fit]

            model_vec = coeffs @ stamp.vectors
            resid_vec = stamp.substamp - model_vec
            stamp.residuals = resid_vec
            stamp.chi2 = float(np.sqrt(np.mean(resid_vec * resid_vec)))
            if stamp.image_cutout is not None and stamp.ys is None:
                stamp.convolved_model_local = model_vec.reshape(stamp.image_cutout.shape)
            else:
                stamp.convolved_model_local = model_vec

            if np.isnan(stamp.chi2) or np.isnan(stamp.norm):
                continue

            valid_stamps.append(stamp)

        except np.linalg.LinAlgError:
            continue

    if len(valid_stamps) < 3:
        for s in valid_stamps:
            s.diff = 0.0
            s.ignore = False
        return valid_stamps

    k_sums = np.array([s.norm for s in valid_stamps], dtype=np.float64)
    kmean, kstdev = _sigma_clip_mean_stdev(k_sums, stat_sig=stat_sig, maxiter=10)

    for s in valid_stamps:
        if kstdev > 0 and kstdev < 1e29:
            s.diff = abs((s.norm - kmean) / kstdev)
        else:
            s.diff = 0.0
        # C: survived_check = (diff < kerSigReject)
        s.ignore = s.diff >= ker_sig_reject

    return valid_stamps

@njit(cache=True)
def build_matrix_numba(n_comp_kernel, n_spatial, n_bg,
                       stamp_vectors, stamp_weights,
                       n_stamps, n_params):
    """
    Build global matrix A matching alard.c build_matrix.

    Parameterization (0-based, length mat_size):
      [0]                 : constant amplitude for basis 0
      [1 : 1+(M-1)*S]     : spatial polys for bases 1..M-1
      [1+(M-1)*S :]       : background polys
    where M=n_comp_kernel, S=n_spatial.
    """
    A = np.zeros((n_params, n_params))
    n_var = n_comp_kernel - 1  # bases 1..M-1

    for s in range(n_stamps):
        vecs = stamp_vectors[s]
        wxy = stamp_weights[s]
        M = vecs @ vecs.T

        # Basis-0 self term
        A[0, 0] += M[0, 0]

        # Variable bases × variable bases
        for k in range(n_var):
            for p in range(n_spatial):
                row = 1 + k * n_spatial + p
                # cross with basis 0
                A[row, 0] += wxy[p] * M[k + 1, 0]
                A[0, row] = A[row, 0]
                for l in range(n_var):
                    for q in range(n_spatial):
                        col = 1 + l * n_spatial + q
                        A[row, col] += wxy[p] * wxy[q] * M[k + 1, l + 1]

        # Background
        bg0 = 1 + n_var * n_spatial
        for ib in range(n_bg):
            row = bg0 + ib
            v_bg = n_comp_kernel + ib
            # bg × basis 0
            A[row, 0] += M[v_bg, 0]
            A[0, row] = A[row, 0]
            # bg × variable bases
            for k in range(n_var):
                for p in range(n_spatial):
                    col = 1 + k * n_spatial + p
                    val = M[v_bg, k + 1] * wxy[p]
                    A[row, col] += val
                    A[col, row] += val
            # bg × bg
            for jb in range(n_bg):
                col = bg0 + jb
                A[row, col] += M[v_bg, n_comp_kernel + jb]

    return A


@njit(cache=True)
def build_rhs_numba(n_comp_kernel, n_spatial, n_bg,
                    stamp_vectors, stamp_weights, stamp_data_pixels,
                    n_stamps, n_params):
    """Build RHS matching alard.c build_scprod (0-based mat_size layout)."""
    b = np.zeros(n_params)
    n_var = n_comp_kernel - 1

    for s in range(n_stamps):
        vecs = stamp_vectors[s]
        data = stamp_data_pixels[s].astype(np.float64)
        wxy = stamp_weights[s]
        P = vecs @ data

        b[0] += P[0]
        for k in range(n_var):
            for p in range(n_spatial):
                b[1 + k * n_spatial + p] += wxy[p] * P[k + 1]

        bg0 = 1 + n_var * n_spatial
        for ib in range(n_bg):
            b[bg0 + ib] += P[n_comp_kernel + ib]

    return b


def accumulate_normal_equations(stamps, n_comp_ker, n_spatial, n_bg, n_params):
    """
    Accumulate A, b for stamps that may have different vector lengths
    (connected-region mode). Applies stamp.region_weight as a scale on M,P.

    Vectorized (kron) form of the same layout as build_matrix_numba / build_rhs_numba.
    """
    A = np.zeros((n_params, n_params), dtype=np.float64)
    b = np.zeros(n_params, dtype=np.float64)
    n_var = n_comp_ker - 1
    bg0 = 1 + n_var * n_spatial
    for stamp in stamps:
        w_reg = float(getattr(stamp, "region_weight", 1.0) or 1.0)
        vecs = np.asarray(stamp.vectors, dtype=np.float64)
        wxy = np.asarray(stamp.weights, dtype=np.float64)
        data = np.asarray(stamp.substamp, dtype=np.float64)
        M = (vecs @ vecs.T) * w_reg
        P = (vecs @ data) * w_reg

        A[0, 0] += M[0, 0]
        if n_var > 0:
            # Bases 1.. vs basis-0 and vs each other (k outer, p=spatial inner)
            A[1:bg0, 0] += np.kron(M[1:n_comp_ker, 0], wxy)
            A[0, 1:bg0] = A[1:bg0, 0]
            A[1:bg0, 1:bg0] += np.kron(
                M[1:n_comp_ker, 1:n_comp_ker], np.outer(wxy, wxy)
            )

        if n_bg > 0:
            A[bg0 : bg0 + n_bg, 0] += M[n_comp_ker : n_comp_ker + n_bg, 0]
            A[0, bg0 : bg0 + n_bg] = A[bg0 : bg0 + n_bg, 0]
            if n_var > 0:
                M_bg_var = M[n_comp_ker : n_comp_ker + n_bg, 1:n_comp_ker]
                block = (M_bg_var[:, :, None] * wxy[None, None, :]).reshape(
                    n_bg, n_var * n_spatial
                )
                A[bg0 : bg0 + n_bg, 1:bg0] += block
                A[1:bg0, bg0 : bg0 + n_bg] += block.T
            A[bg0 : bg0 + n_bg, bg0 : bg0 + n_bg] += M[
                n_comp_ker : n_comp_ker + n_bg, n_comp_ker : n_comp_ker + n_bg
            ]

        b[0] += P[0]
        if n_var > 0:
            b[1:bg0] += np.kron(P[1:n_comp_ker], wxy)
        if n_bg > 0:
            b[bg0 : bg0 + n_bg] += P[n_comp_ker : n_comp_ker + n_bg]
    return A, b


def _local_coeffs_from_solution(solution, weights, n_comp_ker, n_spatial, n_bg):
    """Map packed 0-based solution → per-stamp basis+bg coefficients."""
    c = np.zeros(n_comp_ker + n_bg)
    c[0] = solution[0]
    n_var = n_comp_ker - 1
    for k in range(n_var):
        val = 0.0
        for p in range(n_spatial):
            val += solution[1 + k * n_spatial + p] * weights[p]
        c[k + 1] = val
    bg0 = 1 + n_var * n_spatial
    c[n_comp_ker:] = solution[bg0 : bg0 + n_bg]
    return c


def _pack_c_layout_solution(
    solution, n_comp_ker, n_spatial, n_bg, n_comp_total, ncomp_ker_layout=None
):
    """
    Pack 0-based mat_size solution into C kernelSol layout (length nCompTotal+1).
    C: [0]=0, [1]=basis0, [2:...]=spatial for bases 1.., then background.

    `ncomp_ker_layout` is config.ncomp_ker (sum of deg_fixe terms). Background
    coeffs must be packed at (ncomp_ker_layout-1)*n_spatial+1+k to match
    get_background in alard.c, even when the fit uses fewer active bases.
    """
    if ncomp_ker_layout is None:
        ncomp_ker_layout = n_comp_ker
    out = np.zeros(n_comp_total + 1, dtype=np.float64)
    out[1] = solution[0]
    n_var = n_comp_ker - 1
    n_var_params = n_var * n_spatial
    out[2 : 2 + n_var_params] = solution[1 : 1 + n_var_params]
    # C get_background: ncompBG = (nCompKer-1)*nComp + 1; uses kernelSol[ncompBG+k]
    ncomp_bg = (ncomp_ker_layout - 1) * n_spatial + 1
    out[ncomp_bg + 1 : ncomp_bg + 1 + n_bg] = solution[1 + n_var_params : 1 + n_var_params + n_bg]
    return out


def _set_spatial_weights(stamp, ker_order, nx, ny):
    """Spatial polynomial weights at stamp center (C build_matrix wxy order)."""
    r_pix_x_2 = 0.5 * nx
    r_pix_y_2 = 0.5 * ny
    xf = (float(getattr(stamp, "spatial_x", stamp.x)) - r_pix_x_2) / r_pix_x_2
    yf = (float(getattr(stamp, "spatial_y", stamp.y)) - r_pix_y_2) / r_pix_y_2
    weights = []
    a1 = 1.0
    for ideg1 in range(ker_order + 1):
        a2 = 1.0
        for ideg2 in range(ker_order - ideg1 + 1):
            weights.append(a1 * a2)
            a2 *= yf
        a1 *= xf
    stamp.weights = np.asarray(weights, dtype=np.float64)


def _advance_stamp_group(group_stamp, template, image, config, kernel_vecs, oversample, ker_order, nx, ny):
    """
    Advance to the next substamp in a group (C check_again: sscnt++ then fillStamp).
    Returns True if a valid substamp was loaded, False if the group is exhausted.
    """
    while True:
        group_stamp.sscnt += 1
        if group_stamp.sscnt >= group_stamp.nss:
            group_stamp.ignore = True
            return False
        x, y = group_stamp.substamp_coords[group_stamp.sscnt]
        group_stamp.x = int(x)
        group_stamp.y = int(y)
        if populate_stamp_vectors(group_stamp, template, image, config, kernel_vecs, oversample):
            _set_spatial_weights(group_stamp, ker_order, nx, ny)
            group_stamp.ignore = False
            return True
        # Failed fill — try next substamp (same as C looping fillStamp failures)


def fit_kernel(stamps, template, image, config, kernel_vecs, oversample=1, verbose=0, skip_local_reject=False, noise_sq=None, mask=None, region_map=None):
    """
    Main Fitting Driver.

    `stamps` may be:
      - a flat list of substamp-like objects (.x, .y), or
      - a list of groups (each a list of substamp-like objects), matching C's
        stamp_struct with xss[]/yss[] and sscnt advancement in check_again.

    If region_map is provided, delegates to connected-region irregular stamp fit
    (star-footprint exclusion instead of sscnt advancement).

    If skip_local_reject is True (typical after C-like FOM filtering), only
    populate vectors and run the global iterative fit — do not sigma-clip on
    local kernel sums first.

    noise_sq: optional combined noise-variance image for figMerit='v' (default).
    mask: optional int bitflag mask; FLAG_INPUT_ISBAD pixels skipped in sig.
    """
    if region_map is not None:
        from .region_fitting import fit_kernel_regions

        return fit_kernel_regions(
            stamps,
            template,
            image,
            config,
            kernel_vecs,
            region_map,
            oversample=oversample,
            verbose=verbose,
            skip_local_reject=skip_local_reject,
            noise_sq=noise_sq,
            mask=mask,
        )

    FLAG_INPUT_ISBAD = 0x80

    # Normalize to stamp groups (list of list of coord objects)
    if stamps and isinstance(stamps[0], (list, tuple)):
        stamp_groups = stamps
    else:
        stamp_groups = [[s] for s in stamps]

    n_comp_ker = len(kernel_vecs)
    ker_order = config.ko if hasattr(config, "ko") else 2
    bg_order = config.bgo
    ker_sig_reject = float(getattr(config, "ks", 2.0))
    stat_sig = float(getattr(config, "stat_sig", 3.0))
    fig_merit = str(getattr(config, "fom", "v") or "v")
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    n_bg = (bg_order + 1) * (bg_order + 2) // 2
    n_params = (n_comp_ker - 1) * n_spatial + n_bg + 1
    n_comp_total = getattr(config, "n_comp_total", n_comp_ker * n_spatial + n_bg)
    ncomp_ker_layout = int(getattr(config, "ncomp_ker", n_comp_ker))
    lambda_reg = float(getattr(config, "lambda_reg", 0.0) or 0.0)
    if oversample > 1 and lambda_reg > 0:
        lambda_reg *= float(oversample) ** 2

    ny, nx = template.shape
    lr_ny = ny // oversample if oversample > 1 else ny
    lr_nx = nx // oversample if oversample > 1 else nx
    half_stamp = int(config.rss)

    valid_stamps = []
    for gidx, group in enumerate(stamp_groups):
        if not group:
            continue
        coords = [(int(s.x), int(s.y)) for s in group]
        stamp = Stamp(coords[0][0], coords[0][1], orig_idx=gidx)
        stamp.substamp_coords = coords
        stamp.nss = len(coords)
        stamp.sscnt = 0

        ok = populate_stamp_vectors(stamp, template, image, config, kernel_vecs, oversample)
        while not ok and stamp.sscnt + 1 < stamp.nss:
            stamp.sscnt += 1
            stamp.x, stamp.y = coords[stamp.sscnt]
            ok = populate_stamp_vectors(stamp, template, image, config, kernel_vecs, oversample)
        if not ok:
            continue

        _set_spatial_weights(stamp, ker_order, lr_nx, lr_ny)
        valid_stamps.append(stamp)

    if not skip_local_reject:
        local_ok = []
        for stamp in valid_stamps:
            n_fit = n_comp_ker + 1
            vectors_fit = stamp.vectors[:n_fit]
            try:
                coeffs_fit = np.linalg.solve(vectors_fit @ vectors_fit.T, vectors_fit @ stamp.substamp)
                stamp.norm = float(coeffs_fit[0])
                stamp.local_solution = coeffs_fit
                local_ok.append(stamp)
            except np.linalg.LinAlgError:
                stamp.ignore = True
        if len(local_ok) >= 3:
            k_sums = np.array([s.norm for s in local_ok], dtype=np.float64)
            kmean, kstdev = _sigma_clip_mean_stdev(k_sums, stat_sig=stat_sig, maxiter=10)
            for s in local_ok:
                s.diff = abs((s.norm - kmean) / kstdev) if kstdev > 0 and kstdev < 1e29 else 0.0
                s.ignore = s.diff >= ker_sig_reject

    solution = None
    packed = None
    final_active_stamps = []
    fit_stats = {"meansig": 0.0, "scatter": 0.0, "n_skipped": 0}

    def _stamp_sigma(stamp, coeffs):
        """Match getStampSig: 'v' = mean(diff^2/noise), else RMS of residuals."""
        model = coeffs @ stamp.vectors
        resid = stamp.substamp - model
        y0 = int(stamp.y - half_stamp)
        x0 = int(stamp.x - half_stamp)
        y1 = y0 + 2 * half_stamp + 1
        x1 = x0 + 2 * half_stamp + 1
        fw = 2 * half_stamp + 1
        good = np.isfinite(resid) & (np.abs(stamp.substamp) > 1e-20)
        if mask is not None and 0 <= y0 and 0 <= x0 and y1 <= mask.shape[0] and x1 <= mask.shape[1]:
            mcut = mask[y0:y1, x0:x1].ravel().astype(np.int32)
            if mcut.size == resid.size:
                good &= (mcut & FLAG_INPUT_ISBAD) == 0
        if fig_merit.startswith("v") and noise_sq is not None:
            if y0 < 0 or x0 < 0 or y1 > noise_sq.shape[0] or x1 > noise_sq.shape[1]:
                return float(np.sqrt(np.mean(resid * resid))) if resid.size else -1.0
            ncut = noise_sq[y0:y1, x0:x1].ravel().astype(np.float64)
            if ncut.size != resid.size:
                return float(np.sqrt(np.mean(resid * resid)))
            good &= np.isfinite(ncut) & (ncut > 0)
            if not np.any(good):
                return -1.0
            return float(np.mean((resid[good] ** 2) / ncut[good]))
        if not np.any(good):
            return -1.0
        return float(np.sqrt(np.mean(resid[good] * resid[good])))

    for iteration in range(10):
        active_stamps = [s for s in valid_stamps if not s.ignore and s.sscnt < s.nss]
        final_active_stamps = active_stamps
        n_active = len(active_stamps)

        if n_active == 0:
            if verbose >= 1:
                print("No stamps left!")
            break

        if verbose >= 2:
            print(f"DEBUG: Iteration {iteration} - Active Stamps: {n_active}")

        s_vectors = np.stack([s.vectors for s in active_stamps])
        s_weights = np.stack([s.weights for s in active_stamps])
        s_data = np.stack([s.substamp for s in active_stamps])

        A = build_matrix_numba(n_comp_ker, n_spatial, n_bg, s_vectors, s_weights, n_active, n_params)
        b = build_rhs_numba(n_comp_ker, n_spatial, n_bg, s_vectors, s_weights, s_data, n_active, n_params)

        try:
            solution = _tikhonov_solve(A, b, lambda_reg=lambda_reg)
            packed = _pack_c_layout_solution(
                solution, n_comp_ker, n_spatial, n_bg, n_comp_total, ncomp_ker_layout
            )
            fit_stats["lambda_reg"] = lambda_reg
            fit_stats["solution_maxabs"] = float(np.max(np.abs(solution)))

            if verbose >= 2:
                print(f"DEBUG: Iter {iteration} Solution[:10]: {solution[:10]}")

        except np.linalg.LinAlgError:
            print("Singular matrix")
            break

        sigmas = []
        bad_fill = []
        for si, s in enumerate(active_stamps):
            c = _local_coeffs_from_solution(solution, s.weights, n_comp_ker, n_spatial, n_bg)
            sigma = _stamp_sigma(s, c)
            if sigma < 0:
                bad_fill.append(si)
                sigmas.append(np.nan)
            else:
                sigmas.append(sigma)
                s.chi2 = sigma

        need_refit = False
        for si in bad_fill:
            s = active_stamps[si]
            if verbose >= 2:
                print(f"    stamp ({s.x},{s.y}) BAD sig; advance sscnt")
            _advance_stamp_group(
                s, template, image, config, kernel_vecs, oversample,
                ker_order, lr_nx, lr_ny,
            )
            need_refit = True

        good_sigmas = np.array([sg for sg in sigmas if np.isfinite(sg)], dtype=np.float64)
        if good_sigmas.size == 0:
            break

        mean_sig, std_sig = _sigma_clip_mean_stdev(good_sigmas, stat_sig=stat_sig, maxiter=10)
        fit_stats["meansig"] = mean_sig
        fit_stats["scatter"] = std_sig
        fit_stats["n_skipped"] = sum(1 for s in valid_stamps if s.ignore or s.sscnt >= s.nss)

        if std_sig == 0 or std_sig >= 1e29:
            if not need_refit:
                break
        else:
            for s, sigma in zip(active_stamps, sigmas):
                if not np.isfinite(sigma):
                    continue
                if (sigma - mean_sig) > ker_sig_reject * std_sig:
                    if verbose >= 2:
                        print(
                            f"    stamp ({s.x},{s.y}) sig={sigma:.3f} > "
                            f"{mean_sig + ker_sig_reject * std_sig:.3f}; advance sscnt"
                        )
                    _advance_stamp_group(
                        s, template, image, config, kernel_vecs, oversample,
                        ker_order, lr_nx, lr_ny,
                    )
                    need_refit = True

        if not need_refit:
            break

    if solution is not None:
        for s in final_active_stamps:
            if s.ignore:
                continue
            c = _local_coeffs_from_solution(solution, s.weights, n_comp_ker, n_spatial, n_bg)
            model = c @ s.vectors
            if s.image_cutout is not None:
                s.convolved_model_global = model.reshape(s.image_cutout.shape)
            else:
                dim = int(np.sqrt(model.shape[0]))
                s.convolved_model_global = model.reshape((dim, dim))

    for s in final_active_stamps:
        s.fit_stats = fit_stats

    return packed if packed is not None else solution, final_active_stamps

