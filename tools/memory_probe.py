"""Run native lifecycles under Valgrind, or enumerate allocation failures.

No package installation is changed. Pass --package to the installed/built
hotpants package directory. OOM mode requires the Linux/glibc preload helper.
"""

import argparse
import ctypes
import gc
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--package", type=Path, default=Path(__file__).resolve().parents[1] / "hotpants"
)
parser.add_argument("--oom", action="store_true")
parser.add_argument("--repeat", type=int, default=30)
args = parser.parse_args()
path = next(args.package.glob("hotpants_ext*.so"))
spec = importlib.util.spec_from_file_location("hotpants_ext", path)
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
Config = runpy.run_path(str(args.package / "config.py"))["HotpantsConfig"]
c = Config(
    nx=128,
    ny=128,
    rkernel=3,
    rss=5,
    ngauss=1,
    deg_fixe=[2],
    sigma_gauss=[0.7],
    nstampx=3,
    nstampy=3,
    ko=0,
    bgo=0,
    force_convolve="b",
    tlthresh=-1e6,
    ilthresh=-1e6,
    tuthresh=1e9,
    iuthresh=1e9,
)
d = c.to_dict()
rng = np.random.default_rng(42)
a = rng.normal(100, 2, (128, 128)).astype(np.float32)
noise = np.ones_like(a)
coords = [
    dict(stamp_group_id=j, substamp_id=j, x=24 + 40 * j, y=24 + 40 * j)
    for j in range(3)
]
cat = np.array([[q["x"], q["y"]] for q in coords], np.float32)
groups = [{"substamps": [(q["x"], q["y"])]} for q in coords]
k = np.zeros(c.n_comp_total + 1)
k[1] = 1
mask = np.zeros_like(a, np.int32)


def prepare():
    s = e.HotpantsState(128, 128, d)
    e.make_input_mask(s, a, a, None, None)
    return s


def action(s, name):
    if name == "init":
        return e.HotpantsState(128, 128, d)
    if name == "mask":
        return e.make_input_mask(s, a, a, None, None)
    if name == "noise":
        return e.calculate_noise_image(s, a, True)
    if name == "find":
        return e.find_stamps(s, a, a, 1.0, cat)
    if name == "find_auto":
        return e.find_stamps(s, a, a, 1.0, None)
    if name == "fom":
        return e.fit_stamps_and_get_fom(s, a, a, noise, "t", coords)
    if name == "fit":
        return e.fit_kernel(s, groups, a, a, noise)
    if name == "apply":
        return e.apply_kernel(s, a, k, noise)
    if name == "background":
        return e.get_background_image(s, k)
    if name == "rescale":
        return e.rescale_noise_ok(s, a, noise, mask)
    if name == "stats":
        return e.calculate_final_stats(s, a, noise, mask)
    if name == "visualize":
        return e.visualize_kernel(s, (64, 64), k, 2.0)


phases = [
    "init",
    "mask",
    "noise",
    "find",
    "find_auto",
    "fom",
    "fit",
    "apply",
    "background",
    "rescale",
    "stats",
    "visualize",
]
if args.oom:
    injector = ctypes.CDLL(None)
    injector.hp_fail_after.argtypes = [ctypes.c_long]
    for name in phases:
        for nth in range(10000):
            s = prepare()
            injector.hp_fail_after(nth)
            try:
                result = action(s, name)
            except MemoryError:
                assert injector.hp_did_fail(), (name, nth, "unexpected OOM")
            else:
                assert not injector.hp_did_fail(), (
                    name,
                    nth,
                    "allocation failure was ignored",
                )
                del result
                injector.hp_fail_after(-1)
                del s
                gc.collect()
                print(
                    json.dumps({"phase": name, "allocation_sites_checked": nth}),
                    flush=True,
                )
                break
            finally:
                injector.hp_fail_after(-1)
            del s
            gc.collect()
        else:
            raise AssertionError("allocation enumeration did not terminate")
else:
    for repeat in range(args.repeat):
        s = prepare()
        for name in phases:
            action(s, name)
        # Repeated stateful stages, plus destruction immediately after FOM.
        for _ in range(3):
            action(s, "find")
            action(s, "fom")
        del s
        bad = d.copy()
        del bad["fwkernel"]
        try:
            e.HotpantsState(128, 128, bad)
        except KeyError:
            pass
        s = prepare()
        z = np.zeros_like(a)
        e.fit_stamps_and_get_fom(s, z, z, noise, "t", coords)
        try:
            e.fit_kernel(s, groups, z, z, noise)
        except ValueError:
            pass
        del s
        gc.collect()
    print(json.dumps({"native_cycles": args.repeat, "status": "passed"}), flush=True)

# Deterministic edge-data exercise: rectangular/tiny fields, masked and
# nonfinite pixels, constant data, and automated stamp search.
if not args.oom:
    for j in range(30):
        ny, nx = map(int, rng.integers(8, 96, size=2))
        cfg = d.copy()
        cfg["nstampx"] = cfg["nstampy"] = 3
        state = e.HotpantsState(nx, ny, cfg)
        arr = rng.normal(100, 3, (ny, nx)).astype(np.float32)
        if j % 4 == 0:
            arr.fill(0)
        if j % 4 == 1:
            arr.fill(100)
        if j % 4 == 2:
            arr.ravel()[::17] = np.nan
        if j % 4 == 3:
            arr.ravel()[::19] = np.inf
        mask_arr = e.make_input_mask(state, arr, arr, None, None)
        e.find_stamps(state, arr, arr, 1.0, None)
        e.calculate_final_stats(state, arr, np.ones_like(arr), mask_arr)
        e.apply_kernel(state, arr, k, np.ones_like(arr))
        del state
    gc.collect()
    print(json.dumps({"edge_data_cases": 30, "status": "passed"}), flush=True)
