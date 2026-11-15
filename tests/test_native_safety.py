"""Native boundary and lifetime regressions; run under Valgrind as well as pytest."""

import gc
import weakref

import numpy as np
import pytest

from hotpants import Hotpants, HotpantsConfig
from hotpants import hotpants_ext as ext
from hotpants.pure import convolution, kernel, utils


def config(**kwargs):
    values = dict(
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
        force_convolve="t",
        tlthresh=-1e6,
        ilthresh=-1e6,
        tuthresh=1e9,
        iuthresh=1e9,
        fitthresh=1,
        verbose=0,
    )
    values.update(kwargs)
    return HotpantsConfig(**values)


def inputs(**kwargs):
    c = config(**kwargs)
    s = ext.HotpantsState(128, 128, c.to_dict())
    a = np.random.default_rng(2).normal(100, 2, (128, 128)).astype(np.float32)
    noise = np.ones_like(a)
    k = np.zeros(c.n_comp_total + 1, dtype=np.float64)
    k[1] = 1
    return s, c, a, noise, k


def coordinates(n=3):
    return [
        dict(stamp_group_id=j, substamp_id=j, x=24 + 40 * j, y=24 + 40 * j)
        for j in range(n)
    ]


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((1, 1), np.float32),
        np.ones((128, 128), np.float64),
        np.ones((128, 128), dtype=">f4"),
        np.ones((128, 128), np.float32).T,
        np.ones((128, 128), np.float32)[:, ::-1],
        np.ones(128 * 128, np.float32),
    ],
)
def test_reject_unsafe_buffers(bad):
    s, _, a, noise, k = inputs()
    for call in (
        lambda: ext.calculate_noise_image(s, bad, True),
        lambda: ext.make_input_mask(s, bad, a, None, None),
        lambda: ext.apply_kernel(s, bad, k, noise),
        lambda: ext.find_stamps(s, bad, a, 1.0, None),
        lambda: ext.fit_stamps_and_get_fom(s, a, bad, noise, "t", coordinates()),
        lambda: ext.fit_kernel(s, [{"substamps": [(64, 64)]}], bad, a, noise),
        lambda: ext.calculate_final_stats(s, bad, noise, np.zeros_like(a, np.int32)),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("n", [1, 8, 24, 25, 32])
def test_small_image_mask_is_bounded(n):
    a = np.ones((n, n), np.float32)
    h = Hotpants(a, a)
    mask = h.results["input_mask"]
    assert mask.shape == a.shape
    assert np.all(mask & utils.FLAG_T_BAD)


def test_uninitialized_and_reinitialized_state():
    s = ext.HotpantsState.__new__(ext.HotpantsState)
    with pytest.raises(RuntimeError, match="initialized"):
        ext.calculate_noise_image(s, np.ones((1, 1), np.float32), True)
    s.__init__(128, 128, config().to_dict())
    with pytest.raises(RuntimeError, match="reinitialized"):
        s.__init__(128, 128, config().to_dict())


@pytest.mark.parametrize(
    "key,value",
    [
        ("nss", 0),
        ("nstampx", 0),
        ("rkernel", -1),
        ("ko", -1),
        ("bgo", -2),
        ("tgain", 0),
        ("sigma_gauss", [0]),
        ("deg_fixe", [2**40]),
        ("fwkernel", 100),
        ("ncomp_ker", 1),
        ("force_convolve", "template"),
        ("use_pca", True),
    ],
)
def test_raw_config_validation(key, value):
    d = config().to_dict()
    d[key] = value
    with pytest.raises((ValueError, OverflowError)):
        ext.HotpantsState(128, 128, d)


def test_parse_failure_after_allocations():
    d = config().to_dict()
    del d["fwkernel"]
    for _ in range(20):
        with pytest.raises(KeyError):
            ext.HotpantsState(128, 128, d)


def test_repeat_search_and_array_lifetimes():
    s, c, a, noise, k = inputs()
    initial = ext.make_input_mask(s, a, a, None, None)
    cat = np.array([[24, 24], [64, 64], [104, 104]], np.float32)
    first = ext.find_stamps(s, a, a, 1.0, cat)
    for _ in range(20):
        assert ext.find_stamps(s, a, a, 1.0, cat) == first
    outputs = ext.apply_kernel(s, a, k, noise)
    saved = [v.copy() for v in outputs]
    del s
    gc.collect()
    for actual, expected in zip(outputs, saved):
        np.testing.assert_array_equal(actual, expected)
    ref = weakref.ref(outputs[0])
    del outputs
    gc.collect()
    assert ref() is None
    assert initial.shape == (128, 128)


def test_destroy_after_automatic_fom():
    for _ in range(5):
        s, c, a, noise, k = inputs(force_convolve="b")
        ext.fit_stamps_and_get_fom(s, a, a, noise, "t", coordinates())
        del s
        gc.collect()


@pytest.mark.parametrize("bgo", [-1, 0, 1, 2])
def test_background_orders_and_forced_fom(bgo):
    s, c, a, noise, k = inputs(bgo=bgo)
    fom, results = ext.fit_stamps_and_get_fom(s, a, a, noise, "t", coordinates())
    assert len(results) == 3
    assert results[0]["local_solution"].size == c.ncomp_ker + (bgo >= 0)
    assert np.isfinite(results[0]["local_solution"]).all()


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [None],
        [{}],
        [dict(stamp_group_id=-1, substamp_id=0, x=64, y=64)],
        [dict(stamp_group_id=0, substamp_id=0, x=0, y=0)],
        [dict(stamp_group_id=0, substamp_id=j, x=64 + j, y=64) for j in range(4)],
    ],
)
def test_bad_substamps_raise(bad):
    s, c, a, noise, k = inputs()
    with pytest.raises((ValueError, TypeError)):
        ext.fit_stamps_and_get_fom(s, a, a, noise, "t", bad)


def test_singular_local_rejected_global_raises():
    s, c, a, noise, k = inputs()
    a.fill(0)
    _, result = ext.fit_stamps_and_get_fom(s, a, a, noise, "t", coordinates(1))
    assert not result[0]["survived_check"]
    with pytest.raises(ValueError, match="singular"):
        ext.fit_kernel(s, [{"substamps": [(64, 64)]}], a, a, noise)


def test_noise_rescale_and_invalid_statistics():
    s, c, a, noise, k = inputs()
    mask = np.full(a.shape, utils.FLAG_OK_CONV, np.int32)
    output = ext.rescale_noise_ok(s, a - a.mean(), noise, mask)
    assert np.isfinite(output).all() and output.mean() > 1
    with pytest.raises(ValueError):
        ext.rescale_noise_ok(s, a, noise, np.full(a.shape, 0x8000, np.int32))
    stats = ext.calculate_final_stats(s, a, noise, np.full(a.shape, 0x8000, np.int32))
    assert np.isnan(stats["diff_mean"])


@pytest.mark.parametrize("conv_var", [False, True])
def test_fixed_operator_matches_python(conv_var):
    s, c, a, noise, k = inputs(ko=1, conv_var=conv_var)
    k[2:5] = [0.01, 0.02, 0.03]
    mask = np.zeros(a.shape, np.int32)
    mask[64, 63] = utils.FLAG_INPUT_MASK
    cmask = ext.make_input_mask(s, a, a, mask, None)
    pmask = utils.make_input_mask(a, a, c, mask, None)
    np.testing.assert_array_equal(cmask, pmask)
    co, mo, vo = ext.apply_kernel(s, a, k, noise)
    po, _, pv, pm = convolution.apply_kernel(
        a,
        k,
        noise,
        pmask,
        c,
        kernel.calculate_kernel_basis((7, 7), c.sigma_gauss, c.deg_fixe),
    )
    np.testing.assert_allclose(co, po, rtol=2e-6, atol=2e-5)
    np.testing.assert_allclose(vo, pv, rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(mo, pm)


def test_visualize_honors_position_and_preserves_application():
    s, c, a, noise, k = inputs(ko=1)
    k[3] = 0.2
    before = ext.apply_kernel(s, a, k, noise)[0]
    left = ext.visualize_kernel(s, (20, 20), k, 2.0)
    right = ext.visualize_kernel(s, (100, 100), k, 2.0)
    assert not np.array_equal(left, right)
    vecs = np.array(kernel.calculate_kernel_basis((7, 7), c.sigma_gauss, c.deg_fixe))
    expected = convolution.jit_make_kernel(
        k, 7, 3, 128.0, 128.0, c.ncomp_ker, c.ko, vecs, 20.0, 20.0
    )
    np.testing.assert_allclose(left[4:11, 4:11], expected, atol=1e-7)
    np.testing.assert_array_equal(before, ext.apply_kernel(s, a, k, noise)[0])


def test_statistics_do_not_mutate_readonly_mask():
    s, c, a, noise, k = inputs()
    a[50, 50] = np.nan
    mask = np.zeros_like(a, np.int32)
    mask.flags.writeable = False
    ext.calculate_final_stats(s, a, noise, mask)
    assert not np.any(mask)


def test_group_scoring_uses_first_substamp():
    s, c, a, noise, k = inputs()
    ref = a.copy()
    ref[19:30, 39:50] *= 4
    coords = coordinates()
    _, first = ext.fit_stamps_and_get_fom(s, a, ref, noise, "t", coords)
    coords.insert(1, dict(stamp_group_id=0, substamp_id=99, x=44, y=24))
    _, extended = ext.fit_stamps_and_get_fom(s, a, ref, noise, "t", coords)
    assert extended[0]["fom"] == pytest.approx(first[0]["fom"])


@pytest.mark.parametrize("native", [True, False])
def test_degenerate_gaussian_rejected(native):
    if native:
        s, c, a, noise, k = inputs(sigma_gauss=[1e20])
        with pytest.raises(ValueError):
            ext.apply_kernel(s, a, k, noise)
    else:
        with pytest.raises(ValueError):
            kernel.calculate_kernel_basis((7, 7), [1e20], [2])


@pytest.mark.parametrize("size", [8, 32, 128])
def test_automatic_search_on_constant_image_terminates(size):
    c = config(nx=size, ny=size)
    s = ext.HotpantsState(size, size, c.to_dict())
    a = np.zeros((size, size), np.float32)
    ext.make_input_mask(s, a, a, None, None)
    assert ext.find_stamps(s, a, a, 1.0, None) == ([], [])


def test_conversion_callbacks_cannot_mutate_borrowed_buffers():
    s, c, a, noise, k = inputs()
    called = []

    class Callback:
        def __index__(self):
            called.append(True)
            return 64

        def __float__(self):
            called.append(True)
            return 1.0

    bad = coordinates(1)
    bad[0]["x"] = Callback()
    with pytest.raises(TypeError):
        ext.fit_stamps_and_get_fom(s, a, a, noise, "t", bad)
    with pytest.raises(TypeError):
        ext.visualize_kernel(s, (Callback(), 64), k, 2.0)
    d = c.to_dict()
    d["sigma_gauss"] = [Callback()]
    with pytest.raises(TypeError):
        ext.HotpantsState(128, 128, d)
    assert not called
