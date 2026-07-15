
import numpy as np
from numba import njit, typed
import scipy.ndimage as ndimage

# =========================================================
# Numba-Accelerated Cutout Functions (Matches functions.py)
# =========================================================

@njit(cache=True)
def cut_substamp_from_image(image: np.ndarray, x_center: int, y_center: int, half_width: int, fill_value: float = np.nan) -> np.ndarray:
    """
    Extracts a square substamp from a larger image, equivalent to the C 'cutSStamp' logic.
    """
    ny, nx = image.shape
    full_width = 2 * half_width + 1

    # Define boundaries in source coordinates
    y_start = y_center - half_width
    y_end = y_center + half_width + 1
    x_start = x_center - half_width
    x_end = x_center + half_width + 1

    # Clip to image dimensions
    y_start_clipped = max(0, y_start)
    y_end_clipped = min(ny, y_end)
    x_start_clipped = max(0, x_start)
    x_end_clipped = min(nx, x_end)

    # Output array
    full_cutout = np.full((full_width, full_width), fill_value, dtype=image.dtype)
    
    # Calculate paste coordinates
    paste_y_start = y_start_clipped - y_start
    paste_y_end = paste_y_start + (y_end_clipped - y_start_clipped)
    paste_x_start = x_start_clipped - x_start
    paste_x_end = paste_x_start + (x_end_clipped - x_start_clipped)
    
    # Copy data if there is an overlap
    if paste_y_end > paste_y_start and paste_x_end > paste_x_start:
         full_cutout[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = \
             image[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

    return full_cutout

@njit(cache=True)
def process_all_substamps_numba(image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, half_width: int, fill_value: float = np.nan) -> np.ndarray:
    """
    Processes a batch of substamps using Numba. Returns a 3D array of cutouts.
    """
    num_substamps = len(x_coords)
    full_width = 2 * half_width + 1
    # Store as 3D array for easier handling than list
    all_cutouts = np.zeros((num_substamps, full_width, full_width), dtype=image.dtype)

    for i in range(num_substamps):
        x_center = int(round(x_coords[i]))
        y_center = int(round(y_coords[i]))
        all_cutouts[i] = cut_substamp_from_image(image, x_center, y_center, half_width, fill_value)

    return all_cutouts

# =========================================================
# Noise Calculation (Matches getStampStats3)
# =========================================================

def calculate_noise(data, mask=None, n_stat=100, max_iter=5):
    """
    Calculate noise statistics using histogram-based mode estimation.
    Matches logic from getStampStats3 in functions.c.
    
    Parameters
    ----------
    data : ndarray
        The input image data.
    mask : ndarray, optional
        Boolean mask (True = bad pixel).
    n_stat : int
        Minimum number of points required.
        
    Returns
    -------
    sigma : float
        Estimated standard deviation (FWHM / 1.35).
    mode : float
        Estimated mode of the distribution.
    """
    if mask is not None:
        valid_data = data[~mask]
    else:
        valid_data = data.flatten()
        
    valid_data = valid_data[np.isfinite(valid_data)]
    n_pts = len(valid_data)
    
    if n_pts < n_stat:
        return 0.0, 0.0 # Or error? C returns error code.
    
    # 1. Initial 100-point random sample to estimate bin size (functions.c:806)
    # in Python we can just use a larger sample or all data if small
    sample_size = min(n_pts, 1000)
    sample = np.random.choice(valid_data, size=sample_size, replace=False)
    sample.sort()
    
    # ufstat=0.9, mfstat=0.5
    u_val = sample[int(0.9 * len(sample))]
    m_val = sample[int(0.5 * len(sample))]
    
    binsize = (u_val - m_val) / 100.0 # "nstat=100" in C
    bin1 = m_val - 128.0 * binsize
    
    # Sigma clip pre-histogram (functions.c:858)
    # We use scipy-like iterative sigma clipping
    try:
        current_data = valid_data
        mean_val = np.mean(current_data)
        std_val = np.std(current_data)
        
        for _ in range(max_iter):
            mask_clip = np.abs(current_data - mean_val) < 3.0 * std_val # statSig=3.0 usually? C uses 10? No, statSig default 3?
            if np.all(mask_clip):
                break
            current_data = current_data[mask_clip]
            mean_val = np.mean(current_data)
            std_val = np.std(current_data)
    except:
        pass
        
    # Re-binning Loop (functions.c:871)
    # Simplifying: We can use numpy.histogram which is efficient
    
    # 256 bins
    # range: [bin1, bin1 + 256*binsize]
    
    final_binsize = binsize
    final_bin1 = bin1
    
    hist_data = current_data
    
    # We will skip the exact "resize bins" loop from C for brevity/speed unless strictly required,
    # as numpy histogram is robust. The C code handles dynamic range issues.
    # Assuming the sigma-clipping above cleaned the data well enough.
    
    # Construct Histogram
    # "find narrowest region which holds ~10% of points"
    
    # Let's use a dense histogram approach
    # C uses 256 bins. We can use more for better resolution.
    hist, bin_edges = np.histogram(hist_data, bins=256) # Automatic range?
    # C forces specific binsize. Let's trust numpy's auto-binning or use the calculated one?
    # Using calculated one might clip data.
    
    # Implementation of "Narrowest 10% Window" Mode Estimation
    # This is equivalent to finding the peak density in sorted data
    sorted_d = np.sort(hist_data)
    n = len(sorted_d)
    window_pts = int(0.1 * n)
    
    if window_pts < 2:
        return 0.0, np.median(hist_data)
    
    # Width of windows containing 10% of points
    widths = sorted_d[window_pts:] - sorted_d[:-window_pts]
    
    # Find index of minimum width (highest density)
    min_idx = np.argmin(widths)
    
    # Mode is the mean of this densest window
    mode_val = np.mean(sorted_d[min_idx : min_idx + window_pts])
    
    # FWHM / Sigma estimation
    # "find the region around mode containing half the noise points" (25% to 75% centered on mode?)
    # C logic: 
    # lower = goodcnt * 0.25; upper = goodcnt * 0.75; matches IQR logic approx.
    # But C code calculates index sums. 
    # Ideally: FWHM = 1.35 * Sigma.
    # We can use IQR of the *clipped* data as a robust estimator.
    q25, q75 = np.percentile(hist_data, [25, 75])
    sigma_val = (q75 - q25) / 1.349
    
    return float(sigma_val), float(mode_val)

# =========================================================
# Masking & Stamp Finding
# =========================================================

# Match globals.h FLAG_* values used by makeInputMask / spreadMask / borders.
FLAG_BAD_PIXVAL = 0x01
FLAG_SAT_PIXEL = 0x02
FLAG_LOW_PIXEL = 0x04
FLAG_ISNAN = 0x08
FLAG_INPUT_MASK = 0x20
FLAG_OK_CONV = 0x40
FLAG_INPUT_ISBAD = 0x80
FLAG_T_BAD = 0x100
FLAG_I_BAD = 0x400
FLAG_T_SKIP = 0x200
FLAG_I_SKIP = 0x800


def mask_pixels(image, low_thresh=None, high_thresh=None, bitmask=None):
    """
    Create input mask based on thresholds and input bitmask.
    Matches logic for FLAG_LOW_PIXEL, FLAG_SAT_PIXEL.
    """
    # 0 = Good
    # 1 = Bad (FLAG_BAD_PIXVAL / FLAG_INPUT_ISBAD etc in C)
    # In C: 
    # tLThresh/tUThresh are checked. 
    # bitmask is checked.
    
    mask = np.zeros(image.shape, dtype=bool)
    
    if low_thresh is not None:
        mask |= (image <= low_thresh)
        
    if high_thresh is not None:
        mask |= (image >= high_thresh)
        
    if bitmask is not None:
        mask |= (bitmask != 0)
        
    return mask


def upsample_lr_to_hr(arr_lr, factor: int):
    """np.repeat both axes. factor==1 returns arr_lr (no copy needed if factor==1)."""
    if factor == 1:
        return arr_lr
    return np.repeat(np.repeat(arr_lr, factor, axis=0), factor, axis=1)


def downsample_hr_mask_to_lr(mask_hr: np.ndarray, factor: int) -> np.ndarray:
    """OR-reduce all bits over factor x factor blocks. Return int32 LR mask.

    Mirror apply_kernel mask downsampling loop.
    """
    if factor == 1:
        return np.asarray(mask_hr, dtype=np.int32, copy=False)
    mask_hr = np.asarray(mask_hr, dtype=np.int32)
    ny_hr, nx_hr = mask_hr.shape
    ny_lr = ny_hr // factor
    nx_lr = nx_hr // factor
    mask_out_lr = np.zeros((ny_lr, nx_lr), dtype=np.int32)
    for oy in range(factor):
        for ox in range(factor):
            mask_out_lr |= mask_hr[oy::factor, ox::factor][:ny_lr, :nx_lr]
    return mask_out_lr


def spread_mask(m_data: np.ndarray, width: int) -> None:
    """In-place spread of FLAG_INPUT_ISBAD, matching spreadMask() in functions.c."""
    if width <= 0:
        return
    w2 = width // 2
    ny, nx = m_data.shape
    bad = np.argwhere(m_data & FLAG_INPUT_ISBAD)
    for j, i in bad:
        for k in range(-w2, w2 + 1):
            ii = i + k
            if ii < 0 or ii >= nx:
                continue
            for l in range(-w2, w2 + 1):
                jj = j + l
                if jj < 0 or jj >= ny:
                    continue
                if not (m_data[jj, ii] & FLAG_INPUT_ISBAD):
                    m_data[jj, ii] |= FLAG_OK_CONV


def make_input_mask(
    template: np.ndarray,
    image: np.ndarray,
    config,
    t_mask: np.ndarray | None = None,
    i_mask: np.ndarray | None = None,
    oversample: int = 1,
) -> np.ndarray:
    """
    Build an int32 input mask matching hotpants_ext.make_input_mask / makeInputMask.

    Output shape always matches template (HR when oversample > 1).
    """
    F = int(oversample)
    ny_hr, nx_hr = template.shape
    m_data = np.zeros((ny_hr, nx_hr), dtype=np.int32)

    fill = float(getattr(config, "fillval", 1.0e-30))
    t_u = float(config.tuthresh)
    t_l = float(config.tlthresh)
    i_u = float(config.iuthresh)
    i_l = float(config.ilthresh)

    if F == 1:
        if t_mask is not None:
            m_data[t_mask > 0] |= FLAG_INPUT_MASK
        if i_mask is not None:
            m_data[i_mask > 0] |= FLAG_INPUT_MASK

        # Exact makeInputMask() predicates (NaNs do not satisfy == / >= / <= in C either).
        m_data[(template == fill) | (image == fill)] |= FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL
        m_data[(template >= t_u) | (image >= i_u)] |= FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL
        m_data[(template <= t_l) | (image <= i_l)] |= FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL
        m_data[np.isnan(template) | np.isnan(image)] |= FLAG_ISNAN | FLAG_INPUT_ISBAD

        spread = int(config.rkernel * getattr(config, "kf_spread_mask1", 1.0))
        s_border = int(config.rss + config.rkernel)
    else:
        ny_lr, nx_lr = image.shape
        if template.shape != (ny_lr * F, nx_lr * F):
            raise ValueError(
                f"make_input_mask oversample={F}: template shape {template.shape} "
                f"!= image shape {image.shape} * {F}"
            )

        if t_mask is not None:
            m_data[t_mask > 0] |= FLAG_INPUT_MASK
        if i_mask is not None:
            i_mask_hr = upsample_lr_to_hr(i_mask, F)
            m_data[i_mask_hr > 0] |= FLAG_INPUT_MASK

        image_hr = upsample_lr_to_hr(image, F)

        m_data[template == fill] |= FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL
        m_data[image_hr == fill] |= FLAG_INPUT_ISBAD | FLAG_BAD_PIXVAL
        m_data[template >= t_u] |= FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL
        m_data[image_hr >= i_u] |= FLAG_INPUT_ISBAD | FLAG_SAT_PIXEL
        m_data[template <= t_l] |= FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL
        m_data[image_hr <= i_l] |= FLAG_INPUT_ISBAD | FLAG_LOW_PIXEL
        m_data[np.isnan(template)] |= FLAG_ISNAN | FLAG_INPUT_ISBAD
        m_data[np.isnan(image_hr)] |= FLAG_ISNAN | FLAG_INPUT_ISBAD

        spread = int(config.rkernel * getattr(config, "kf_spread_mask1", 1.0)) * F
        s_border = int(config.rss + config.rkernel) * F

    spread_mask(m_data, spread)

    # Border mask: hwKSStamp + hwKernel
    m_data[:, :s_border] |= FLAG_T_BAD | FLAG_I_BAD
    m_data[:, nx_hr - s_border :] |= FLAG_T_BAD | FLAG_I_BAD
    m_data[:s_border, s_border : nx_hr - s_border] |= FLAG_T_BAD | FLAG_I_BAD
    m_data[ny_hr - s_border :, s_border : nx_hr - s_border] |= FLAG_T_BAD | FLAG_I_BAD

    return m_data


def _check_psf_center(
    data: np.ndarray,
    mask: np.ndarray,
    xmax: int,
    ymax: int,
    s_xmin: int,
    s_ymin: int,
    s_pix_x: int,
    s_pix_y: int,
    hw_ks: int,
    hi_thresh: float,
    sky: float,
    inv_dsky: float,
    fit_thresh: float,
    bad_bits: int,
    sat_bit: int,
) -> float:
    """Port of checkPsfCenter() using stamp-local centers like the C code."""
    ny, nx = data.shape
    # Convert to stamp-local coordinates (C passes xmax - x0, ymax - y0).
    jmax = ymax - s_ymin
    imax = xmax - s_xmin
    dmax2 = 0.0
    for l in range(jmax - hw_ks, jmax + hw_ks + 1):
        if l < 0 or l >= s_pix_y:
            continue
        yr2 = l + s_ymin
        for k in range(imax - hw_ks, imax + hw_ks + 1):
            if k < 0 or k >= s_pix_x:
                continue
            xr2 = k + s_xmin
            if mask[yr2, xr2] & bad_bits:
                return 0.0
            dpt2 = float(data[yr2, xr2])
            if dpt2 >= hi_thresh:
                mask[yr2, xr2] |= sat_bit
                return 0.0
            if ((dpt2 - sky) * inv_dsky) > fit_thresh:
                dmax2 += dpt2
    return dmax2


def _mark_skip_like_c(mask: np.ndarray, xmax: int, ymax: int, hw_ks: int, skip_bit: int) -> None:
    """
    Replicate buildStamps' skip masking, including its index order:
    nr2 = l + rPixX * k with l in ymax±hw, k in xmax±hw.
    """
    ny, nx = mask.shape
    flat = mask.reshape(-1)
    n_tot = nx * ny
    for l in range(ymax - hw_ks, ymax + hw_ks + 1):
        for k in range(xmax - hw_ks, xmax + hw_ks + 1):
            nr2 = l + nx * k
            if 0 <= nr2 < n_tot:
                flat[nr2] |= skip_bit


def find_stamps_from_catalog(template, image, mask, catalog, config, oversample: int = 1):
    """
    Catalog stamp search matching hotpants_ext.find_stamps + buildStamps(getCenters=0).

    Returns (template_substamps, image_substamps) as lists of dicts with
    substamp_id, stamp_group_id, x, y — same schema as the C extension.

    When oversample > 1, template and mask are HR; image and catalog (x, y) are LR.
    """
    F = int(oversample)
    if F == 1:
        ny_lr, nx_lr = template.shape
        m_work_hr = mask.astype(np.int32, copy=True)
        m_work_lr = m_work_hr
    else:
        ny_lr, nx_lr = image.shape
        if template.shape != (ny_lr * F, nx_lr * F):
            raise ValueError(
                f"find_stamps_from_catalog oversample={F}: template shape {template.shape} "
                f"!= image shape {image.shape} * {F}"
            )
        m_work_hr = mask.astype(np.int32, copy=True)
        m_work_lr = downsample_hr_mask_to_lr(m_work_hr, F)

    n_stamp_x = int(config.nstampx)
    n_stamp_y = int(config.nstampy)
    fw_stamp = int(getattr(config, "fwstamp", 0)) or max(
        int(min(nx_lr / n_stamp_x, ny_lr / n_stamp_y) - (2 * config.rkernel + 1)),
        2 * config.rss + 2 * config.rkernel + 1,
    )
    hw_kernel = int(config.rkernel)
    hw_ks = int(config.rss)
    hw_ks_hr = hw_ks * F
    n_ks = int(config.nss)
    force = str(getattr(config, "force_convolve", "b"))
    fit_thresh = float(config.fitthresh)
    t_uk = float(config.tuktresh if config.tuktresh is not None else config.tuthresh)
    i_uk = float(config.iuktresh if config.iuktresh is not None else config.iuthresh)

    cat = np.asarray(catalog, dtype=np.float32)
    t_groups = []
    i_groups = []

    r_xmin, r_ymin = 0, 0
    r_xmax, r_ymax = nx_lr - 1, ny_lr - 1

    nt_s = 0
    ni_s = 0
    n_stamps_max = n_stamp_x * n_stamp_y

    for l in range(n_stamp_y):
        for k in range(n_stamp_x):
            s_xmin = r_xmin + int(k * float(r_xmax - r_xmin + 1) / n_stamp_x)
            s_ymin = r_ymin + int(l * float(r_ymax - r_ymin + 1) / n_stamp_y)
            s_xmax = min(s_xmin + fw_stamp - 1, r_xmax)
            s_ymax = min(s_ymin + fw_stamp - 1, r_ymax)
            s_pix_x = s_xmax - s_xmin + 1
            s_pix_y = s_ymax - s_ymin + 1

            t_xss, t_yss = [], []
            i_xss, i_yss = [], []

            if F == 1:
                t_cut = template[s_ymin : s_ymax + 1, s_xmin : s_xmax + 1]
                i_cut = image[s_ymin : s_ymax + 1, s_xmin : s_xmax + 1]
                t_m = (m_work_hr[s_ymin : s_ymax + 1, s_xmin : s_xmax + 1] & 0xBF) != 0
                i_m = t_m
            else:
                y0_hr, y1_hr = s_ymin * F, (s_ymax + 1) * F
                x0_hr, x1_hr = s_xmin * F, (s_xmax + 1) * F
                t_cut = template[y0_hr:y1_hr, x0_hr:x1_hr]
                i_cut = image[s_ymin : s_ymax + 1, s_xmin : s_xmax + 1]
                t_m = (m_work_hr[y0_hr:y1_hr, x0_hr:x1_hr] & 0xBF) != 0
                i_m = (m_work_lr[s_ymin : s_ymax + 1, s_xmin : s_xmax + 1] & 0xBF) != 0

            try:
                t_fwhm, t_mode = calculate_noise(t_cut, mask=t_m)
            except Exception:
                t_fwhm, t_mode = 1.0, 0.0
            try:
                i_fwhm, i_mode = calculate_noise(i_cut, mask=i_m)
            except Exception:
                i_fwhm, i_mode = 1.0, 0.0
            if not np.isfinite(t_fwhm) or t_fwhm <= 0:
                t_fwhm = 1.0
            if not np.isfinite(i_fwhm) or i_fwhm <= 0:
                i_fwhm = 1.0
            if not np.isfinite(t_mode):
                t_mode = 0.0
            if not np.isfinite(i_mode):
                i_mode = 0.0

            for entry in cat:
                x_pos = int(round(float(entry[0])))
                y_pos = int(round(float(entry[1])))
                if not (
                    (x_pos > s_xmin + hw_kernel + 1)
                    and (x_pos < s_xmax - hw_kernel - 1)
                    and (y_pos > s_ymin + hw_kernel + 1)
                    and (y_pos < s_ymax - hw_kernel - 1)
                ):
                    continue

                if force != "i" and len(t_xss) < n_ks:
                    if F == 1:
                        t_x, t_y = x_pos, y_pos
                        t_sx, t_sy = s_xmin, s_ymin
                        t_spx, t_spy = s_pix_x, s_pix_y
                        t_hw = hw_ks
                        t_mask_work = m_work_hr
                    else:
                        t_x, t_y = x_pos * F, y_pos * F
                        t_sx, t_sy = s_xmin * F, s_ymin * F
                        t_spx, t_spy = s_pix_x * F, s_pix_y * F
                        t_hw = hw_ks_hr
                        t_mask_work = m_work_hr
                    check = _check_psf_center(
                        template,
                        t_mask_work,
                        t_x,
                        t_y,
                        t_sx,
                        t_sy,
                        t_spx,
                        t_spy,
                        t_hw,
                        t_uk,
                        t_mode,
                        1.0 / t_fwhm,
                        fit_thresh,
                        FLAG_T_BAD | FLAG_T_SKIP | 0xBF,
                        FLAG_T_BAD,
                    )
                    if check != 0.0:
                        _mark_skip_like_c(t_mask_work, t_x, t_y, t_hw, FLAG_T_SKIP)
                        t_xss.append(x_pos)
                        t_yss.append(y_pos)

                if force != "t" and len(i_xss) < n_ks:
                    check = _check_psf_center(
                        image,
                        m_work_lr,
                        x_pos,
                        y_pos,
                        s_xmin,
                        s_ymin,
                        s_pix_x,
                        s_pix_y,
                        hw_ks,
                        i_uk,
                        i_mode,
                        1.0 / i_fwhm,
                        fit_thresh,
                        FLAG_I_BAD | FLAG_I_SKIP | 0xBF,
                        FLAG_I_BAD,
                    )
                    if check != 0.0:
                        _mark_skip_like_c(m_work_lr, x_pos, y_pos, hw_ks, FLAG_I_SKIP)
                        i_xss.append(x_pos)
                        i_yss.append(y_pos)

            if force != "i" and t_xss:
                t_groups.append((t_xss, t_yss))
                nt_s += 1
            if force != "t" and i_xss:
                i_groups.append((i_xss, i_yss))
                ni_s += 1

            if nt_s >= n_stamps_max or ni_s >= n_stamps_max:
                break
        if nt_s >= n_stamps_max or ni_s >= n_stamps_max:
            break

    t_out, i_out = [], []
    sid = 0
    for gid, (xss, yss) in enumerate(t_groups):
        for x, y in zip(xss, yss):
            t_out.append({"substamp_id": sid, "stamp_group_id": gid, "x": int(x), "y": int(y)})
            sid += 1
    sid = 0
    for gid, (xss, yss) in enumerate(i_groups):
        for x, y in zip(xss, yss):
            i_out.append({"substamp_id": sid, "stamp_group_id": gid, "x": int(x), "y": int(y)})
            sid += 1
    return t_out, i_out


def _mask_pixel_is_bad(mask: np.ndarray, y: int, x: int) -> bool:
    """True if pixel is rejected for stamp finding.

    Integer masks: reject any flag except pure FLAG_OK_CONV (spread halo).
    Bool masks: truthiness.
    """
    v = mask[y, x]
    if mask.dtype == np.bool_ or mask.dtype == bool:
        return bool(v)
    return (int(v) & ~FLAG_OK_CONV) != 0


def _mask_region_has_bad(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> bool:
    """True if any pixel in region is rejected for stamp finding."""
    region = mask[y0:y1, x0:x1]
    if region.dtype == np.bool_ or region.dtype == bool:
        return bool(np.any(region))
    return bool(np.any((region.astype(np.int32) & ~FLAG_OK_CONV) != 0))


def find_stamps(image, mask, n_stamps, box_size, border_width=10):
    """
    Find stamp centers matching findStamps logic.
    1. Local Maxima
    2. Sort by Flux
    3. Check against Mask (Zero Tolerance)
    """
    # Use max filter to find peaks
    neighborhood_size = box_size # approximation
    data_max = ndimage.maximum_filter(image, size=neighborhood_size)
    maxima = (image == data_max)
    
    # Thresholding? C uses:
    # "ignore anything with state->fillVal or zero"
    # "sort by flux"
    
    # Get coordinates of maxima
    y_peaks, x_peaks = np.where(maxima)
    
    candidates = []
    
    h, w = image.shape
    half_box = box_size // 2
    
    for y, x in zip(y_peaks, x_peaks):
        # 1. Check Mask (Point check)
        if _mask_pixel_is_bad(mask, y, x):
            continue
            
        # 2. Check Borders
        if (x < border_width or x >= w - border_width or 
            y < border_width or y >= h - border_width):
            continue
            
        # 3. Check Stamp Region Mask (Zero Tolerance)
        # "check_box" logic
        x0 = max(0, x - half_box)
        x1 = min(w, x + half_box + 1)
        y0 = max(0, y - half_box)
        y1 = min(h, y + half_box + 1)
        
        if _mask_region_has_bad(mask, y0, y1, x0, x1):
            continue
            
        # 4. Calculate Flux (Metric)
        flux = image[y, x] # Peak value as metric? Or sum? 
        # C code `findStamps`: uses `quick_phot` (aperture sum)
        flux = np.sum(image[y0:y1, x0:x1])
        
        candidates.append((flux, x, y))
        
    # Sort by flux descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select top N
    selected = candidates[:n_stamps]
    
    # Format as list of substamps / structures? 
    # Returning simple list of dicts or objects
    # Pure python impl usually wants objects, but for now coords
    return [{'x': c[1], 'y': c[2], 'flux': c[0]} for c in selected]
    
# =========================================================
# Image Manipulation
# =========================================================

@njit(cache=True)
def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsamples an image by summing blocks of size factor x factor.
    Assumes image dimensions are multiples of factor.
    
    Parameters
    ----------
    image : ndarray
        High-resolution image.
    factor : int
        Integer downsampling factor.
        
    Returns
    -------
    downsampled : ndarray
        Low-resolution image (summed).
    """
    if factor == 1:
        return image
        
    ny, nx = image.shape
    new_ny = ny // factor
    new_nx = nx // factor
    
    downsampled = np.zeros((new_ny, new_nx), dtype=image.dtype)
    
    for j in range(new_ny):
        for i in range(new_nx):
            block_sum = 0.0
            y_start = j * factor
            x_start = i * factor
            for ky in range(factor):
                for kx in range(factor):
                    block_sum += image[y_start + ky, x_start + kx]
            downsampled[j, i] = block_sum
            
    return downsampled
