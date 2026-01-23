"""
HOTPanTS Python Wrapper

A Python interface for the HOTPanTS image differencing software.
"""

from .core import Hotpants, HotpantsConfig, Substamp, SubstampStatus

__version__ = "0.1.1"

# Make the main classes and functions available at package level
__all__ = ["Hotpants", "HotpantsConfig", "HotpantsError", "Substamp", "SubstampStatus", "__version__"]
