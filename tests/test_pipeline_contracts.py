"""Public workflows, units, masking, repeatability and saved-kernel behavior."""

import gc

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from astropy.io import fits

from hotpants import (
    Hotpants,
    HotpantsConfig,
    HotpantsError,
    KernelModel,
    convolve_template,
)
from hotpants.pure.fitting import _tikhonov_solve
from hotpants.pure.os_precompute import precompute_basis_lr_maps, block_sum_downsample
from hotpants.pure import utils, kernel
from hotpants.functions import process_all_substamps_numba


def scene():
    rng = np.random.default_rng(875)
    yy, xx = np.mgrid[:128, :128]
    template = rng.normal(100, 1, (128, 128)).astype(np.float32)
    cat = []
    for y in (24, 64, 104):
        for x in (24, 64, 104):
            template += (
                rng.uniform(1500, 4000)
                * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.4**2))
            ).astype(np.float32)
            cat.append((x + 1, y + 1))
    science = (
        1.2 * gaussian_filter(template, 0.8) + rng.normal(0, 0.5, template.shape)
    ).astype(np.float32)
    return template, science, np.array(cat, np.float32)


def config(**kwargs):
    values = dict(
        nx=128,
        ny=128,
        rkernel=3,
        rss=5,
        ngauss=1,
        deg_fixe=[2],
        sigma_gauss=[0.8],
        nstampx=3,
        nstampy=3,
        ko=0,
        bgo=0,
        force_convolve="t",
        normalize="u",
        tlthresh=-1e6,
        ilthresh=-1e6,
        tuthresh=1e9,
        iuthresh=1e9,
        fitthresh=1,
    )
    values.update(kwargs)
    return HotpantsConfig(**values)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("direction", ["t", "i", "b"])
@pytest.mark.parametrize("bgo", [-1, 0, 1])
def test_full_workflow(native, direction, bgo):
    t, i, cat = scene()
    hp = Hotpants(
        t,
        i,
        star_catalog=cat,
        config=config(force_convolve=direction, bgo=bgo),
        use_c_extension=native,
    )
    out = hp.run_pipeline()
    assert out["diff_image"].shape == t.shape
    assert np.all(np.isfinite(out["kernel_solution"]))
    assert hp.results["final_fits"]
    assert np.all((out["output_mask"][0, :] & utils.FLAG_OUTPUT_ISBAD) != 0)
    good = out["output_mask"] == 0
    assert np.count_nonzero(good) > 100
    assert np.all(np.isfinite(out["noise_image"][good]))
    saved = out["diff_image"].copy()
    del hp
    gc.collect()
    np.testing.assert_array_equal(out["diff_image"], saved)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("direction", ["t", "i"])
def test_normalization_consistent(native, direction):
    t, i, cat = scene()
    outputs = []
    for normalization in ("u", direction):
        hp = Hotpants(
            t,
            i,
            star_catalog=cat,
            config=config(force_convolve=direction, normalize=normalization),
            use_c_extension=native,
        )
        outputs.append(hp.run_pipeline())
    raw, normalized = outputs
    scale = raw["kernel_solution"][1]
    np.testing.assert_allclose(raw["kernel_solution"], normalized["kernel_solution"])
    for key in ("diff_image", "noise_image", "convolved_image", "background"):
        np.testing.assert_allclose(
            normalized[key], raw[key] / scale, rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize("native", [True, False])
def test_pedestals_and_config_snapshot(native):
    t, i, cat = scene()
    c = config(tpedestal=10, ipedestal=20)
    hp = Hotpants(t + 10, i + 20, star_catalog=cat, config=c, use_c_extension=native)
    np.testing.assert_allclose(hp.template_data, t, atol=2e-4)
    np.testing.assert_allclose(hp.image_data, i, atol=2e-4)
    assert c.tuthresh == 1e9 and c.tpedestal == 10
    c.ko = 3
    hp.find_stamps()  # external config edits do not alter the existing instance
    hp.config.ko = 2
    with pytest.raises(HotpantsError, match="configuration changed"):
        hp.find_stamps()


@pytest.mark.parametrize("native", [True, False])
def test_repeat_pipeline_and_rescale(native):
    t, i, cat = scene()
    hp = Hotpants(
        t, i, star_catalog=cat, config=config(rescale_ok=True), use_c_extension=native
    )
    first = hp.run_pipeline()
    second = hp.run_pipeline()
    np.testing.assert_allclose(
        second["kernel_solution"], first["kernel_solution"], rtol=1e-8, atol=1e-8
    )
    assert np.isfinite(second["noise_image"][second["output_mask"] == 0]).all()


def test_saved_kernel_is_detached_and_matches_raw_convolution():
    t, i, cat = scene()
    hp = Hotpants(t, i, star_catalog=cat, config=config())
    out = hp.run_pipeline()
    model = KernelModel.from_hotpants(hp)
    actual = convolve_template(t, model)
    np.testing.assert_allclose(
        actual, out["convolved_image"] - out["background"], atol=3e-4
    )
    before = model.kernel_solution.copy()
    hp.results["kernel_solution"][1] = 99
    hp.config.ko = 7
    np.testing.assert_array_equal(model.kernel_solution, before)
    assert model.config.ko == 0
    with pytest.raises(ValueError):
        convolve_template(t, model.kernel_solution.reshape(1, -1), config=model.config)


def test_pathlike_fits_io(tmp_path):
    t, i, cat = scene()
    tp = tmp_path / "template.fits"
    ip = tmp_path / "image.fits"
    out = tmp_path / "difference.fits"
    fits.writeto(tp, t)
    fits.writeto(ip, i)
    hp = Hotpants(tp, ip, star_catalog=cat, config=config(output_file=out))
    result = hp.run_pipeline()
    np.testing.assert_array_equal(fits.getdata(out), result["diff_image"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"oversample": 0},
        {"oversample": 1.5},
        {"oversample": True},
        {"t_error": np.full((128, 128), -1.0)},
        {"star_catalog": np.array([[np.nan, 20.0]])},
    ],
)
def test_invalid_public_inputs(kwargs):
    t, i, _ = scene()
    with pytest.raises((ValueError, HotpantsError)):
        Hotpants(t, i, config=config(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rss": 0},
        {"ko": -1},
        {"bgo": -2},
        {"nstampx": 0},
        {"tgain": 0},
        {"sigma_gauss": [0.0]},
        {"force_convolve": "z"},
        {"normalize": ""},
        {"lambda_reg": -1.0},
        {"region_connectivity": 6},
        {"typo": 1},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        config(**kwargs)


def test_singular_python_solver_does_not_return_nan():
    with pytest.raises(np.linalg.LinAlgError):
        _tikhonov_solve(np.zeros((2, 2)), np.zeros(2))
    result = _tikhonov_solve(np.ones((2, 2)), np.ones(2))
    np.testing.assert_allclose(np.ones((2, 2)) @ result, np.ones(2))
    with pytest.raises(np.linalg.LinAlgError):
        _tikhonov_solve(np.full((2, 2), np.nan), np.ones(2))


def test_numba_cutouts_fill_and_shape_checks():
    image = np.ones((8, 8), np.float32)
    out = process_all_substamps_numba(
        image, np.array([100.0]), np.array([100.0]), 2, -99.0
    )
    assert np.all(out == -99)
    with pytest.raises(ValueError):
        process_all_substamps_numba(
            image, np.array([1.0, 2.0]), np.array([1.0]), 2, -99.0
        )
    with pytest.raises(ValueError):
        utils.downsample_image(image, 3)


def test_fft_nonfinite_support_is_local():
    tpl = np.ones((40, 40))
    tpl[10, 10] = np.nan
    kernels = np.array(kernel.calculate_kernel_basis((5, 5), [0.5], [0]))
    maps = precompute_basis_lr_maps(tpl, kernels, 2)
    assert np.isnan(maps[0, 5, 5])
    assert np.isfinite(maps[0, 15, 15])
    np.testing.assert_allclose(maps[0, 15, 15], 4.0)


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_block_factor_validation(factor):
    with pytest.raises(ValueError):
        block_sum_downsample(np.ones((8, 8)), factor)


@pytest.mark.parametrize("native", [True, False])
def test_fits_mask_roundtrip(tmp_path, native):
    t, i, cat = scene()
    path = tmp_path / "mask.fits"
    hp = Hotpants(
        t,
        i,
        star_catalog=cat,
        config=config(mask_image_file=path),
        use_c_extension=native,
    )
    result = hp.run_pipeline()
    saved = fits.getdata(path)
    np.testing.assert_array_equal(saved, result["output_mask"])
    assert np.all(saved[0, :] & utils.FLAG_OUTPUT_ISBAD)
    assert fits.getheader(path)["PHOTNORM"] == "u"


def test_caller_image_mutation_does_not_change_instance():
    t, i, cat = scene()
    hp = Hotpants(t, i, star_catalog=cat, config=config())
    saved = hp.template_data.copy()
    t.fill(0)
    np.testing.assert_array_equal(hp.template_data, saved)


@pytest.mark.parametrize("stamp_mode", ["grid", "connected_regions"])
def test_oversampled_workflow(stamp_mode):
    t, _, cat = scene()
    hr = np.repeat(np.repeat(t / 4, 2, axis=0), 2, axis=1)
    science = block_sum_downsample(gaussian_filter(hr, 1.6), 2).astype(np.float32)
    hp = Hotpants(
        hr,
        science,
        star_catalog=cat,
        oversample=2,
        use_c_extension=False,
        config=config(deg_fixe=[0], stamp_mode=stamp_mode, region_min_npix=20),
    )
    result = hp.run_pipeline()
    assert result["diff_image"].shape == t.shape
    assert np.isfinite(result["kernel_solution"]).all()
    assert hp.results["final_fits"]
    with pytest.raises(ValueError, match="oversampled"):
        KernelModel.from_hotpants(hp)


def test_connected_auto_cache_matches_selected_direction():
    t, i, cat = scene()
    cfg = config(stamp_mode="connected_regions", region_min_npix=20, force_convolve="b")
    auto = Hotpants(t, i, star_catalog=cat, config=cfg, use_c_extension=False)
    result = auto.run_pipeline()
    cfg.force_convolve = result["conv_direction"]
    forced = Hotpants(
        t, i, star_catalog=cat, config=cfg, use_c_extension=False
    ).run_pipeline()
    np.testing.assert_allclose(
        result["kernel_solution"], forced["kernel_solution"], rtol=1e-8, atol=1e-8
    )
