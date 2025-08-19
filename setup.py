from setuptools import setup, Extension
import numpy as np

hotpants_ext = Extension(
    "hotpants_wrapper.hotpants_ext",
    sources=[
        "hotpants_wrapper/hotpants_ext.c",
        "src/alard.c",
        "src/functions.c",
    ],
    include_dirs=["src", np.get_include()],
    libraries=["m"],
    extra_compile_args=["-g", "-O2", "-std=c99"],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(ext_modules=[hotpants_ext])
