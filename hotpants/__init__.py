"""
HOTPanTS Python Wrapper

A Python interface for the HOTPanTS image differencing software.
"""

from .hotpants import Hotpants, HotpantsConfig, Substamp, SubstampStatus

__version__ = "1.0.0"

# Make the main classes and functions available at package level
__all__ = ["Hotpants", "HotpantsConfig", "HotpantsError", "Substamp", "SubstampStatus", "__version__"]
