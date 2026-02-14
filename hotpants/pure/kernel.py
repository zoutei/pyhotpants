
import numpy as np

def calculate_kernel_basis(shape, sigma_gauss, deg_fixe):
    """
    Calculate the kernel basis functions exactly matching alard.c logic.
    
    Parameters
    ----------
    shape : tuple
        Shape of the kernel (ny, nx). Should be odd.
    sigma_gauss : list of floats
        Sigma widths for the Gaussian components.
    deg_fixe : list of ints
        Polynomial degrees for each Gaussian component.
        
    Returns
    -------
    basis : list of ndarray
        List of 2D kernel basis images.
    """
    basis = []
    
    ny, nx = shape
    y0, x0 = (ny - 1) // 2, (nx - 1) // 2
    
    # Generate 1D coordinates (centered)
    # alard.c: x = (double)(ix - state->hwKernel);
    # ix goes from 0 to fwKernel-1. hwKernel = (fwKernel-1)/2.
    # so x goes from -hwKernel to +hwKernel.
    x_coords = np.arange(nx) - x0
    y_coords = np.arange(ny) - y0
    
    # Store the first basis vector (k=0) for orthogonalization
    b0 = None
    nvec = 0
    
    for ig, sigma in enumerate(sigma_gauss):
        degree = deg_fixe[ig]
        
        # alard.c stores 1/(2*sigma^2) in state->sigma_gauss
        # and uses exp(-x*x * state->sigma_gauss)
        # We calculate inv_2sigma2 to match that usage
        inv_2sigma2 = 1.0 / (2.0 * sigma**2)
        
        # Pre-compute Gaussian part for 1D arrays
        # qe = exp(-x * x * state->sigma_gauss[ig])
        g_x = np.exp(-x_coords**2 * inv_2sigma2)
        g_y = np.exp(-y_coords**2 * inv_2sigma2)
        
        # Loop over degrees exactly as in alard.c / fillStamp
        # for (idegx = 0; idegx <= state->deg_fixe[ig]; idegx++)
        #    for (idegy = 0; idegy <= state->deg_fixe[ig]-idegx; idegy++)
        for deg_x in range(degree + 1):
            for deg_y in range(degree - deg_x + 1):
                
                # Check for orthogonalization (ren flag in C)
                # dx = (deg_x / 2) * 2 - deg_x; (0 if even, !=0 if odd)
                # dy = (deg_y / 2) * 2 - deg_y;
                # if (dx == 0 && dy == 0 && nvec > 0) ren = 1;
                is_even_x = (deg_x % 2 == 0)
                is_even_y = (deg_y % 2 == 0)
                ren = (is_even_x and is_even_y and nvec > 0)
                
                # Construct 1D filters
                # state->filter_x[k] = qe * pow(x, deg_x);
                filter_x = g_x * (x_coords ** deg_x)
                filter_y = g_y * (y_coords ** deg_y)
                
                # Normalize 1D filters
                # sum_x = 1. / sum_x;
                # state->filter_x[ix] *= sum_x;
                sum_x = np.sum(filter_x)
                sum_y = np.sum(filter_y)
                
                if sum_x != 0:
                    filter_x /= sum_x
                if sum_y != 0:
                    filter_y /= sum_y
                    
                # Compute 2D basis vector (Outer Product)
                # vector[i+state->fwKernel*j] = state->filter_x[i...] * state->filter_y[j...]
                # This matches np.outer(filter_y, filter_x)
                b = np.outer(filter_y, filter_x)
                
                # Orthogonalization
                if ren:
                    # Subtract off kernel_vec[0]
                    # vector[i] -= kernel0[i];
                    if b0 is not None:
                        b -= b0
                
                # Store b0 if this is the very first vector (nvec=0)
                if nvec == 0:
                    b0 = b.copy()
                
                basis.append(b)
                nvec += 1
                
    return basis

def get_spatial_polynomials(shape, degree):
    """
    Generate spatial polynomial basis maps x^p * y^q.
    Range is normalized to [-1, 1] across the image dimensions.
    
    Parameters
    ----------
    shape : tuple
        Image shape (ny, nx).
    degree : int
        Maximum degree of the spatial polynomials.
        
    Returns
    -------
    polys : list of ndarray
        List of 2D polynomial maps.
    """
    ny, nx = shape
    y, x = np.indices(shape)
    
    # Normalize to [-1, 1] as in alard.c (build_matrix)
    # fx = (xstamp - rPixX2) / rPixX2; rPixX2 = 0.5 * state->rPixX
    
    half_w = 0.5 * nx
    half_h = 0.5 * ny
    
    # Avoid division by zero
    if half_w == 0: half_w = 1.0
    if half_h == 0: half_h = 1.0
    
    x_norm = (x - half_w) / half_w
    y_norm = (y - half_h) / half_h
    
    polys = []
    
    # Loop matches alard.c build_matrix lines 769+
    # for (ideg1 = 0; ideg1 <= state->kerOrder; ideg1++)
    #    for (ideg2 = 0; ideg2 <= state->kerOrder - ideg1; ideg2++)
    
    for deg_x in range(degree + 1):
        for deg_y in range(degree - deg_x + 1):
            
            # wxy = fx^deg_x * fy^deg_y
            p = (x_norm ** deg_x) * (y_norm ** deg_y)
            polys.append(p)
            
    return polys
