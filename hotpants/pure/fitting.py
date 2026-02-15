
import numpy as np
from numba import njit, prange
import scipy.linalg
from .convolution import jit_xy_conv_stamp, jit_make_kernel, jit_convolve_patch
from .kernel import get_spatial_polynomials
from .utils import downsample_image

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
        
        # Iteration state
        self.sscnt = 0 
        self.nss = 1

def fit_stamps_locally(stamps, template, image, config, kernel_vecs, oversample=1):
    """
    Perform initial local fit on stamps to reject outliers.
    Matches check_stamps in alard.c.
    Returns: List of valid Stamp objects with vectors pre-calculated.
    
    If oversample > 1, template is assumed to be High-Res (shape ~ oversample * image.shape).
    kernel_vecs must be High-Res basis functions.
    convolution is performed in High-Res, then result is downsampled to Low-Res for fitting.
    """
    good_stamps = []
    
    # Constants
    n_comp_ker = len(kernel_vecs)
    bg_order = 0 # config.bgo
    
    # Generate Spatial Polynomials (Global) ?
    # No, for local fit we essentially assume constant spatial variation LOCALLY
    # or we construct the FULL vector set including spatial vars?
    # alard.c check_stamps calls fillStamp.
    # fillStamp constructs vectors for ALL spatial variations?
    # No. 
    # check_stamps uses `build_matrix0` (local matrix). 
    # `build_matrix0` uses `vectors`.
    # `vectors` in `fillStamp` are:
    #   Outer Loop: Gaussian Bases.
    #     Inner Loop: Spatial Decomp (deg_fixe).
    #       `xy_conv_stamp`.
    # Wait, `xy_conv_stamp` generates ONE vector per Gaussian*Degree component.
    # These ARE the basis vectors.
    # My `kernel_vecs` are the PRE-COMBINED Basis Functions (Gaussian * Poly_degree).
    # So `n_comp_ker` is the number of basis functions.
    # Yes.
    
    # Background vectors?
    # `fillStamp` adds background polys.
    # We should add them too.
    
    half_stamp = config.rss
    half_r = config.rkernel
    # If oversample > 1, the kernel radius in HR pixels is larger
    # But half_r comes from config. In pure python flow we likely update config or pass scaled versions?
    # Actually, if we use HR basis functions, `kernel_vecs` already have the HR size.
    # The padding needed in Template image is defined by the kernel size.
    # We should infer HR kernel radius from the basis vectors provided.
    
    h_basis_hr, w_basis_hr = kernel_vecs[0].shape
    half_r_hr = w_basis_hr // 2
    
    # half_stamp is in LOW RES pixels (defined by grid on Science Image).
    
    valid_stamps = []
    
    for idx, s_obj in enumerate(stamps):
        # 1. Extract Data (Image) - RHS
        # The Image to be matched (Difference = Template*K - Image)
        # We model Image as Template*K.
        # So `b` = Template_Basis * Image.
        
        # We need to extract the stamp from IMAGE.
        # (Assuming s_obj has x, y)
        y, x = s_obj.y, s_obj.x
        
        stamp = Stamp(x, y, orig_idx=idx)
        
        # Safe slices
        y0 = int(y - half_stamp)
        y1 = int(y + half_stamp + 1)
        x0 = int(x - half_stamp)
        x1 = int(x + half_stamp + 1)
        
        # Padding bounds for generic safety
        # We leave this to caller usually, but here:
        if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
            continue
            
        # Extract Data Stamp (I)
        data_stamp = image[y0:y1, x0:x1]
        stamp.substamp = data_stamp.flatten()
        stamp.image_cutout = data_stamp.copy()
        
        # 2. Extract Template Patch (T) - For Convolutions
        # Needs kernel padding
        # If oversample > 1, we work in HR coordinates for Template extraction
        
        # Center in HR coords
        # Shift x, y by 0.5 before scaling to center on the pixel grid?
        # Standard: Center of pixel (x,y) in LR corresponds to center of pixel block in HR.
        # Pixel (0,0) LR covers [0, F) x [0, F) HR. Center at F/2 - 0.5.
        # Let's assume simplified: Output x aligns with Input x * F.
        
        # LR Slice: [y0, y1)
        # HR Slice: [y0*F, y1*F)
        # Plus Kernel Padding in HR pixels.
        
        y0_t_hr = int(y0 * oversample - half_r_hr)
        y1_t_hr = int(y1 * oversample + half_r_hr) # No +1 needed? 
        # Wait, convolution output size = Input - Kernel + 1
        # We want Output Size = (y1-y0)*oversample.
        # So Input Size must be Output + Kernel - 1.
        # Let's verify:
        # target_hr_h = (y1 - y0) * oversample
        # needed_input_h = target_hr_h + (2*half_r_hr) 
        
        # So we range from y0*oversample - half_r_hr TO y1*oversample + half_r_hr
        # Height = (y1*F + R) - (y0*F - R) = (y1-y0)F + 2R. Correct.
        
        x0_t_hr = int(x0 * oversample - half_r_hr)
        x1_t_hr = int(x1 * oversample + half_r_hr)
        
        if y0_t_hr < 0 or x0_t_hr < 0 or y1_t_hr > template.shape[0] or x1_t_hr > template.shape[1]:
            continue
            
        template_patch = template[y0_t_hr:y1_t_hr, x0_t_hr:x1_t_hr]

        stamp.template_cutout = template_patch.copy()
        
        # 3. Convolve to get Basis Vectors (V)
        # kernel_vecs are the basis functions.
        vectors = []
        basis_cutouts = [] # These will be LR now
        vec0 = None
        
        for k in range(n_comp_ker):
            basis_k = kernel_vecs[k]
            # Convolve Template Patch with Basis
            # result size should match data stamp * oversample
            conv_res_hr = jit_convolve_patch(template_patch, basis_k)
            
            # Downsample to LR
            if oversample > 1:
                conv_res = downsample_image(conv_res_hr, oversample)
            else:
                conv_res = conv_res_hr
            
            # Store
            v = conv_res.flatten()
            vectors.append(v)
            basis_cutouts.append(conv_res)
            
            if k == 0: 
                vec0 = v
                if idx < 2:
                     print(f"DEBUG: Stamp {idx} Basis 0: BasisSum={np.sum(basis_k):.4f}, TemplateMean={np.mean(template_patch):.4f}, ConvMean={np.mean(conv_res):.4f}, VecMax={np.max(v):.4f}")

            
        # 4. Background Vectors
        # x^i y^j on the stamp grid
        # Center of stamp is 0,0 locally?
        # alard.c `fillStamp`:
        # xf = (i - rPixX2) / rPixX2. GLOBAL coords.
        
        nx_glob = template.shape[1]
        ny_glob = template.shape[0]
        
        gy, gx = np.indices(data_stamp.shape)
        # Shift to global coords
        gy = gy + y0
        gx = gx + x0
        
        # Normalize [-1, 1]
        ny_norm = (gy - 0.5 * ny_glob) / (0.5 * ny_glob)
        nx_norm = (gx - 0.5 * nx_glob) / (0.5 * nx_glob)
        
        for d in range(bg_order + 1):
            for dx in range(d + 1):
                dy = d - dx
                bg_v = (nx_norm ** dx) * (ny_norm ** dy)
                vectors.append(bg_v.flatten())
                
        stamp.vectors = np.array(vectors) # (n_params, n_pix)
        stamp.basis_vectors = np.array(basis_cutouts)
        
        # 5. Local Fit
        # Matrix M = V @ V.T
        # RHS b = V @ I
        
        M = stamp.vectors @ stamp.vectors.T
        b = stamp.vectors @ stamp.substamp
        
        try:
            # Solve
            # Use pseudoinverse or robust solve?
            # alard.c uses ludcmp.
            coeffs = np.linalg.solve(M, b)
            
            # Kernel Sum (sum of first n_comp_ker coeffs)
            # wait, basis 0 is normalized sum=1? 
            # In kernel.py we normalized basis.
            # So sum(Kernel) = sum(coeffs).
            
            # The background coeffs shouldn't be included.
            k_sum = np.sum(coeffs[:n_comp_ker])
            stamp.norm = k_sum
            stamp.local_solution = coeffs

            model_vec = coeffs @ stamp.vectors
            resid_vec = stamp.substamp - model_vec
            stamp.residuals = resid_vec
            stamp.chi2 = float(np.sqrt(np.mean(resid_vec * resid_vec)))
            stamp.convolved_model_local = model_vec.reshape(data_stamp.shape)
            
            valid_stamps.append(stamp)
            
            # Debug Local Fit
            # if config.verbose >= 2 and idx < 5:
            #     print(f"DEBUG: Local Fit Stamp {idx}: Norm={stamp.norm:.4f}, Chi2={stamp.chi2:.4f}, ModelMean={np.mean(model_vec):.4f}, DataMean={np.mean(stamp.substamp):.4f}")
            
        except np.linalg.LinAlgError:
            continue
            
    # 6. Sigma Clip on Kernel Sums
    if len(valid_stamps) < 3:
        return valid_stamps
        
    k_sums = np.array([s.norm for s in valid_stamps])
    
    # Iterative clipping
    mask = np.ones(len(k_sums), dtype=bool)
    for _ in range(10): # 10 iter default
        curr_sums = k_sums[mask]
        if len(curr_sums) < 2: break
        
        mean = np.mean(curr_sums)
        std = np.std(curr_sums)
        
        if std == 0: break
        
        # Reject outliers > statSig (e.g. 3 or 10?)
        # alard.c check_stamps uses sigma_clip with 10 iters.
        # rejection threshold?
        # It calculates diff = abs(val - mean) / std.
        # If diff > threshold?
        # It just stores diff.
        # "stamps[i].diff = fabs((stamps[i].norm - kmean) / kstdev);"
        # The rejection happens in MAIN usually, or by sorting?
        # `check_stamps` in alard.c allows user to reject based on this?
        # No, `check_stamps` essentially seeds `diff`.
        # `fitKernel` uses `stamps` but might skip if `diff` is high?
        # Actually `check_stamps` just computes stats.
        
        # Let's perform a simple 3-sigma clip to Mark BAD stamps
        # User kerSigReject ?
        sig_reject = 3.0 # Default
        
        bad = np.abs(k_sums - mean) > sig_reject * std
        mask[bad] = False
        
        if not np.any(bad): break
        
    # Mark ignored stamps
    for i, s in enumerate(valid_stamps):
        if not mask[i]:
            s.ignore = True
            
    return valid_stamps

@njit(cache=True)
def build_matrix_numba(n_comp_kernel, n_spatial, n_bg, 
                       stamp_vectors, stamp_weights, 
                       n_stamps, n_params):
    """
    Build Global Matrix A (LHS).
    matches build_matrix in alard.c logic.
    """
    # A size: n_params x n_params
    A = np.zeros((n_params, n_params))
    
    # Basis 0 is index 0.
    # Basis 1..M (Variable) * Spatial 0..S maps to indices 1 .. (M-1)*S + S?
    # Index mapping:
    # Param 0: Basis 0 (Const)
    # Param 1..S: Basis 1 * Poly 0..S-1
    # Param S+1..2S: Basis 2 * Poly 0..S-1
    # ...
    
    # Loop over stamps
    for s in range(n_stamps):
        vecs = stamp_vectors[s] # (n_basis + n_bg, n_pix)
        wxy = stamp_weights[s]  # (n_spatial,)
        
        M = vecs @ vecs.T 
        
        # 1. Variable Bases (All Bases 0..M-1 vary spatially)
        for k in range(n_comp_kernel):
            for p in range(n_spatial):
                # row index for Basis k, Poly p
                row_idx = k * n_spatial + p
                
                # Matched with other bases
                for l in range(n_comp_kernel):
                    for q in range(n_spatial):
                         col_idx = l * n_spatial + q
                         
                         val = wxy[p] * wxy[q] * M[k, l]
                         A[row_idx, col_idx] += val
                        
        # 2. Background
        bg_start_idx = n_comp_kernel * n_spatial
        
        for ib in range(n_bg):
            row_idx = bg_start_idx + ib
            vec_idx = n_comp_kernel + ib
            
            # Cross term with Variable Kernel
            for k in range(n_comp_kernel):
                for p in range(n_spatial):
                    col_idx = k * n_spatial + p
                    
                    val = M[vec_idx, k] * wxy[p]
                    A[row_idx, col_idx] += val
                    A[col_idx, row_idx] += val
                    
            # Background-Background
            for jb in range(n_bg):
                 col_idx = bg_start_idx + jb
                 vec_jdx = n_comp_kernel + jb
                 A[row_idx, col_idx] += M[vec_idx, vec_jdx]
                 
    return A

@njit(cache=True)
def build_rhs_numba(n_comp_kernel, n_spatial, n_bg, 
                    stamp_vectors, stamp_weights, stamp_data_pixels,
                    n_stamps, n_params):
    """
    Build RHS Vector b.
    b_i = sum (Vector_i * Data_Image)
    """
    b = np.zeros(n_params)
    
    for s in range(n_stamps):
        vecs = stamp_vectors[s]
        data = stamp_data_pixels[s].astype(np.float64)
        wxy = stamp_weights[s]
        
        P = vecs @ data
        
        # 1. Variable Bases (All 0..M-1)
        for k in range(n_comp_kernel):
            for p in range(n_spatial):
                idx = k * n_spatial + p
                b[idx] += wxy[p] * P[k]
                
        # 2. Background
        bg_start_idx = n_comp_kernel * n_spatial
        for ib in range(n_bg):
            idx = bg_start_idx + ib
            b[idx] += P[n_comp_kernel + ib]
            
    return b

def fit_kernel(stamps, template, image, config, kernel_vecs, oversample=1):
    """
    Main Fitting Driver.
    """
    # 0. Initial Local Fit
    valid_stamps = fit_stamps_locally(stamps, template, image, config, kernel_vecs, oversample=oversample)
    
    # 1. Prepare Data for Global Fit
    n_comp_ker = len(kernel_vecs)
    ker_order = config.ko if hasattr(config, 'ko') else 2
    bg_order = config.bgo
    ker_sig_reject = config.ks if hasattr(config, 'ks') else 2.0  
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    n_bg = (bg_order + 1) * (bg_order + 2) // 2
    
    n_params = n_comp_ker * n_spatial + n_bg
    
    # Global Spatial Polynomials pre-calc?
    # No, we compute local weights for each stamp center
    # "fillStamp" / "check_stamps" does this.
    
    ny, nx = template.shape
    r_pix_x_2 = 0.5 * nx # Should use state rPix? Yes, matches make_kernel
    r_pix_y_2 = 0.5 * ny
    
    # Pre-calculate spatial weights for all stamps
    for s in valid_stamps:
        xf = (s.x - r_pix_x_2) / r_pix_x_2
        yf = (s.y - r_pix_y_2) / r_pix_y_2
        
        weights = []
        for d in range(ker_order + 1):
            for dx in range(d + 1):
                dy = d - dx
                w = (xf ** dx) * (yf ** dy)
                weights.append(w)
        s.weights = np.array(weights)
        
    solution = None
    final_active_stamps = []
    
    # Iteration Loop
    for iteration in range(10): # Max iter
        
        active_stamps = [s for s in valid_stamps if not s.ignore]
        final_active_stamps = active_stamps
        n_active = len(active_stamps)
        
        if n_active == 0:
            print("No stamps left!")
            break
            
        print(f"DEBUG: Iteration {iteration} - Active Stamps: {n_active}")
            
        # Pack Data for Numba
        s_vectors = np.stack([s.vectors for s in active_stamps]) # (n_stamps, n_vec, n_pix)
        s_weights = np.stack([s.weights for s in active_stamps]) # (n_stamps, n_spa)
        s_data = np.stack([s.substamp for s in active_stamps])   # (n_stamps, n_pix)
        
        # Build System
        A = build_matrix_numba(n_comp_ker, n_spatial, n_bg, s_vectors, s_weights, n_active, n_params)
        b = build_rhs_numba(n_comp_ker, n_spatial, n_bg, s_vectors, s_weights, s_data, n_active, n_params)
        
        # Solve
        try:
            # Solve Ax = b
            
            # Precondition A to improve numerical stability (Jacobi Preconditioning)
            # Scale columns and rows by 1/sqrt(diag) so diagonal becomes 1.
            diag_A = np.diag(A).copy()
            # Avoid zero division
            threshold = 1e-20
            diag_A[diag_A <= threshold] = 1.0 
            scale = 1.0 / np.sqrt(diag_A)
            
            # A_scaled = D^-1 * A * D^-1
            # Broadcasting: (N, N) * (N, 1) * (1, N)
            A_scaled = A * scale[:, None] * scale[None, :]
            b_scaled = b * scale
            
            if iteration == 0:
                 print(f"DEBUG: Pre-scaling A_00={A[0,0]:.2e}, b_0={b[0]:.2e}")
            
            # Use lstsq with machine precision threshold
            x_scaled, residuals, rank, s = np.linalg.lstsq(A_scaled, b_scaled, rcond=None)
            
            if iteration == 0:
                 print(f"DEBUG: Scaled A cond={s[0]/s[-1]:.2e}, rank={rank}, max_sv={s[0]:.2e}, min_sv={s[-1]:.2e}")
            
            # Recover solution: x = D^-1 * x_scaled
            solution = x_scaled * scale
            
            print(f"DEBUG: Iter {iteration} Solution[:10]: {solution[:10]}")
            print(f"DEBUG: Iter {iteration} b[:10]: {b[:10]}")
            print(f"DEBUG: Iter {iteration} Solution[430:440]: {solution[430:440]}")
            print(f"DEBUG: Iter {iteration} b[430:440]: {b[430:440]}")

        except np.linalg.LinAlgError:
            print("Singular matrix")
            break
            
        # Sigma Clipping (Post-Fit)
        # Calculate chi2 per stamp
        # This requires re-calculating residuals per stamp using the NEW solution
        
        # We need model per stamp
        # kernel_sol has shape (n_params).
        # We need to map it back to basis coeffs per stamp.
        # But global fit solves for SPATIAL coefficients.
        # Stamp K_n = Sum_k (a_nk * x^deg * y^deg).
        
        # Check Residuals & Sigma Clip (check_again)
        # We need to construct the MODEL for each stamp using the solution `x`.
        
        residuals = []
        sigmas = []
        
        for idx, s in enumerate(active_stamps):
            # Calculate Local Coefficients `c` from Global `x`
            c = np.zeros(n_comp_ker + n_bg)
            
            # Coeff for Basis k: Sum(x[idx] * w[p])
            for k in range(n_comp_ker):
                val = 0.0
                for p in range(n_spatial):
                    idx_x = k * n_spatial + p
                    val += solution[idx_x] * s.weights[p]
                c[k] = val
                
            # Coeff for Background
            bg_start = n_comp_ker * n_spatial
            c[n_comp_ker:] = solution[bg_start : bg_start + n_bg]
            
            # Model = c @ vectors
            model = c @ s.vectors
            resid = s.substamp - model # Data - Model
            
            # Statistics
            # Sigma of residuals
            sigma = np.std(resid)
            sigmas.append(sigma)
            s.chi2 = sigma
            
        sigmas = np.array(sigmas)
        mean_sig = np.mean(sigmas)
        std_sig = np.std(sigmas)

        if std_sig == 0: break
        
        # Reject (One-sided rejection per alard.c check_again)
        # "keep good stamps kerSigReject on the low side" -> Reject high sigma outliers
        threshold = ker_sig_reject 
        
        # C check: (chisq - mean) > threshold * sigma
        bad_indices = np.where((sigmas - mean_sig) > threshold * std_sig)[0]
        
        if len(bad_indices) == 0:
            break
            
        # Mark bad
        rejection_happened = False
        for bi in bad_indices:
            s_bad = active_stamps[bi]
            if not s_bad.ignore:
                s_bad.ignore = True
                rejection_happened = True
                
        if not rejection_happened:
            break
            
    # Populate Global Models for Visualization
    if solution is not None:
        for s in final_active_stamps:
            # Calculate Local Coefficients `c` from Global `x`
            c = np.zeros(n_comp_ker + n_bg)
            for k in range(n_comp_ker):
                val = 0.0
                for p in range(n_spatial):
                    idx_x = k * n_spatial + p
                    val += solution[idx_x] * s.weights[p]
                c[k] = val
            
            bg_start = n_comp_ker * n_spatial
            c[n_comp_ker:] = solution[bg_start : bg_start + n_bg]
            
            # Model = c @ vectors
            model = c @ s.vectors
            if s.image_cutout is not None:
                s.convolved_model_global = model.reshape(s.image_cutout.shape)
            else:
                # Fallback if image_cutout missing
                dim = int(np.sqrt(model.shape[0]))
                s.convolved_model_global = model.reshape((dim, dim))
            
    return solution, final_active_stamps

