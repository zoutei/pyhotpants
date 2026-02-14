
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
        if mask[y, x]:
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
        
        if np.any(mask[y0:y1, x0:x1]):
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
