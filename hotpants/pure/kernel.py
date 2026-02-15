
import numpy as np

def calculate_kernel_basis(shape, sigma_gauss, deg_fixe):
    """
    Calculate the kernel basis functions exactly matching alard.c logic.
    """
    basis = []
    
    ny, nx = shape
    y0, x0 = (ny - 1) // 2, (nx - 1) // 2
    
    # Generate 1D coordinates
    x_coords = np.arange(nx) - x0
    y_coords = np.arange(ny) - y0
    
    # Store the first basis vector (k=0) for orthogonalization
    b0 = None
    nvec = 0
    
    for ig, sigma in enumerate(sigma_gauss):
        degree = deg_fixe[ig]
        
        # Match C extension behavior: sigma_gauss values are used DIRECTLY as coefficients
        # alard.c: qe = exp(-x * x * state->sigma_gauss[ig])
        # The config should contain 1/(2*sigma^2) values, NOT pixel widths
        # If your config has pixel widths, they need to be pre-converted
        sigma_coeff = sigma
        
        # Pre-compute Gaussian part
        g_x = np.exp(-x_coords**2 * sigma_coeff)
        g_y = np.exp(-y_coords**2 * sigma_coeff)
        
        for deg_x in range(degree + 1):
            for deg_y in range(degree - deg_x + 1):
                
                # Check Parity
                is_even_x = (deg_x % 2 == 0)
                is_even_y = (deg_y % 2 == 0)
                
                # 1. Construct 1D filters (Raw)
                filter_x = g_x * (x_coords ** deg_x)
                filter_y = g_y * (y_coords ** deg_y)
                
                # 2. Normalize ONLY if BOTH are even (Strict alard.c replication)
                if is_even_x and is_even_y:
                    sum_x = np.sum(filter_x)
                    sum_y = np.sum(filter_y)
                    
                    if sum_x != 0: filter_x /= sum_x
                    if sum_y != 0: filter_y /= sum_y
                
                # 3. Compute 2D basis vector (Outer Product)
                b = np.outer(filter_y, filter_x)
                
                # 4. Orthogonalization (Renormalization)
                if is_even_x and is_even_y and nvec > 0:
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
