# hotpants/models.py
"""
Data models for the HOTPANTS wrapper.
"""

import numpy as np
from typing import Dict, Optional
from enum import Enum, auto


class SubstampStatus(Enum):
    """Enumeration for the status of a substamp during the fitting process."""

    FOUND = auto()
    PASSED_FOM_CHECK = auto()
    REJECTED_FOM_CHECK = auto()
    USED_IN_FINAL_FIT = auto()
    REJECTED_ITERATIVE_FIT = auto()


class Substamp:
    """
    Data class to hold all information about a single substamp.

    A substamp is a small cutout of an image centered on a bright, isolated
    star, used to model the convolution kernel. This class tracks the data
    and status of each substamp throughout the pipeline.

    Attributes:
        id (int): A unique identifier for the substamp.
        stamp_group_id (int): The identifier for the parent stamp region.
        x (float): The x-coordinate of the substamp center.
        y (float): The y-coordinate of the substamp center.
        status (SubstampStatus): The current processing status of the substamp.
        image_cutout (np.ndarray): The pixel data from the science image.
        template_cutout (np.ndarray): The pixel data from the template image.
        noise_variance_cutout (np.ndarray): The noise variance in the cutout region.
        fit_results (dict): A dictionary containing results from local fits.
        local_kernel_solution (np.ndarray): The kernel solution coefficients
            derived from this substamp alone.
        convolved_model_local (np.ndarray): The convolved model of this
            substamp using its local solution.
        convolved_model_global (np.ndarray): The convolved model of this
            substamp using the final global solution.
        basis_vectors (np.ndarray): A 3D array representing the kernel basis
            functions convolved with the image data at this location. This is
            the core component for the linear fit, as described in the Alard &
            Lupton (1998) paper. Each 2D slice of this array is one of the
            Gaussian-Laguerre basis functions convolved with the local image
            data, forming the design matrix for the least-squares fit.
    """

    def __init__(self, substamp_id: int, stamp_group_id: int, x: int, y: int):
        self.id: int = substamp_id
        self.stamp_group_id: int = stamp_group_id
        self.x: int = int(round(x))
        self.y: int = int(round(y))
        self.status: SubstampStatus = SubstampStatus.FOUND
        self.image_cutout: Optional[np.ndarray] = None
        self.template_cutout: Optional[np.ndarray] = None
        self.noise_variance_cutout: Optional[np.ndarray] = None
        self.fit_results: Dict[str, Dict[str, float]] = {}
        self.local_kernel_solution: Optional[np.ndarray] = None
        self.convolved_model_local: Optional[np.ndarray] = None
        self.convolved_model_global: Optional[np.ndarray] = None
        self.basis_vectors: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"Substamp(id={self.id}, group={self.stamp_group_id}, coords=({self.x}, {self.y}), status={self.status.name})"
