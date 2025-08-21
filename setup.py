import sys
from setuptools import setup, Extension, find_packages
import numpy as np

# --- Platform-specific configuration ---
if sys.platform == "win32":
    # Windows-specific settings
    libraries = []
    extra_compile_args = []
    define_macros = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        # ("_CRT_SECURE_NO_WARNINGS", None),
    ]
else:
    libraries = ["m"]
    extra_compile_args = ["-g", "-O2", "-std=c99"]
    define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
# ------------------------------------

hotpants_ext = Extension(
    "hotpants.hotpants_ext",
    sources=[
        "hotpants/hotpants_ext.c",
        "src/alard.c",
        "src/functions.c",
    ],
    include_dirs=["src", np.get_include()],
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    define_macros=define_macros,
)

setup(
    name="hotpants",
    packages=find_packages(),
    package_dir={"hotpants": "hotpants"},
    include_package_data=True,
    ext_modules=[hotpants_ext],
)
