# functions.py
"""
Python utility functions for the HOTPANTS wrapper, mirroring logic from the C implementation.
"""

import numpy as np
from numba import njit, typed


def cut_substamp_from_image(image: np.ndarray, x_center: int, y_center: int, half_width: int, fill_value: float = np.nan) -> np.ndarray:
    """
    Extracts a square substamp from a larger image, equivalent to the C 'cutSStamp' logic.

    This function handles edge cases where the requested cutout extends beyond the
    image boundaries by padding the output array with a specified fill value.

    Args:
        image (np.ndarray): The full 2D image from which to extract the cutout.
        x_center (int): The integer x-coordinate for the center of the cutout.
        y_center (int): The integer y-coordinate for the center of the cutout.
        half_width (int): The half-width of the desired cutout (e.g., hwksstamp).
                          The full width will be 2 * half_width + 1.
        fill_value (float, optional): The value to use for padding pixels that fall
                                      outside the source image. Defaults to np.nan.

    Returns:
        np.ndarray: A 2D array of shape (2*half_width+1, 2*half_width+1) containing
                    the extracted image data and any necessary padding.
    """
    ny, nx = image.shape
    full_width = 2 * half_width + 1

    # 1. Define the desired boundaries in the source image's coordinate system
    y_start = y_center - half_width
    y_end = y_center + half_width + 1
    x_start = x_center - half_width
    x_end = x_center + half_width + 1

    # 2. Clip these boundaries to the actual dimensions of the source image
    y_start_clipped = max(0, y_start)
    y_end_clipped = min(ny, y_end)
    x_start_clipped = max(0, x_start)
    x_end_clipped = min(nx, x_end)

    # 3. Extract the valid (partial) cutout from the source image
    partial_cutout = image[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

    # 4. Create a full-sized destination array initialized with the fill value
    full_cutout = np.full((full_width, full_width), fill_value, dtype=image.dtype)

    # 5. Calculate the position to paste the partial cutout into the full-sized array
    paste_y_start = y_start_clipped - y_start
    paste_y_end = paste_y_start + (y_end_clipped - y_start_clipped)
    paste_x_start = x_start_clipped - x_start
    paste_x_end = paste_x_start + (x_end_clipped - x_start_clipped)

    # 6. Paste the data and return
    full_cutout[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = partial_cutout

    return full_cutout


@njit(cache=True)
def _numba_cut_substamp(image: np.ndarray, full_cutout: np.ndarray, x_center: int, y_center: int, half_width: int):
    """
    Numba-accelerated core logic for extracting a single substamp.

    This function iterates over the destination cutout array and copies pixels from the
    source image, performing boundary checks for each pixel.

    Args:
        image (np.ndarray): The full source image.
        full_cutout (np.ndarray): The destination array (pre-filled with a fill_value).
        x_center (int): The center x-coordinate in the source image.
        y_center (int): The center y-coordinate in the source image.
        half_width (int): The half-width of the cutout.

    Returns:
        np.ndarray: The populated cutout array.
    """
    ny, nx = image.shape
    # Iterate over the *destination* cutout array's pixels
    for j in range(full_cutout.shape[0]):
        for i in range(full_cutout.shape[1]):
            # Calculate the corresponding coordinate in the source image
            src_y = y_center - half_width + j
            src_x = x_center - half_width + i

            # If the source coordinate is within the image bounds, copy the pixel
            if 0 <= src_y < ny and 0 <= src_x < nx:
                full_cutout[j, i] = image[src_y, src_x]
    return full_cutout


@njit(cache=True)
def process_all_substamps_numba(image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, half_width: int, fill_value: float) -> typed.List:
    """
    Processes a batch of substamps using Numba to accelerate the main loop.

    This function iterates through arrays of coordinates, creating a cutout for each one.
    It is designed to be JIT-compiled for high performance.

    Args:
        image (np.ndarray): The full 2D image from which to extract cutouts.
        x_coords (np.ndarray): A 1D array of x-coordinates for substamp centers.
        y_coords (np.ndarray): A 1D array of y-coordinates for substamp centers.
        half_width (int): The half-width of the desired cutouts.
        fill_value (float): The value for padding pixels outside the source image.

    Returns:
        numba.typed.List: A Numba typed list containing all the generated 2D cutout arrays.
    """
    num_substamps = len(x_coords)

    # Numba requires a typed list to store arrays.
    full_width = 2 * half_width + 1
    all_cutouts = np.zeros((num_substamps, full_width, full_width), dtype=image.dtype)

    for i in range(num_substamps):
        x_center = int(round(x_coords[i]))
        y_center = int(round(y_coords[i]))

        # Call the core helper to populate this single cutout
        all_cutouts[i] = cut_substamp_from_image(image, x_center, y_center, half_width)

    return all_cutouts
