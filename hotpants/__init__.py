"""
HOTPanTS Python Wrapper

A Python interface for the HOTPanTS image differencing software.
"""

from .hotpants import Hotpants, HotpantsConfig, HotpantsError

__version__ = "1.0.0"

# Make the main classes and functions available at package level
__all__ = ["Hotpants", "HotpantsConfig", "HotpantsError", "__version__"]
