"""
HOTPanTS Python Wrapper

A Python interface for the HOTPanTS image differencing software.
"""

from .core import Hotpants, HotpantsConfig, HotpantsError, Substamp, SubstampStatus
from .convolve import KernelModel, convolve_template

__version__ = "0.1.2"

# Make the main classes and functions available at package level
__all__ = [
    "Hotpants",
    "HotpantsConfig",
    "HotpantsError",
    "Substamp",
    "SubstampStatus",
    "KernelModel",
    "convolve_template",
    "__version__",
]
