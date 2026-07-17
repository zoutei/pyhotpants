"""Unit tests for connected-region stamps (pure Python)."""

from __future__ import annotations

import numpy as np
import pytest

from hotpants.config import HotpantsConfig
from hotpants.core import Hotpants, HotpantsError
from hotpants.pure import kernel
from hotpants.pure.fitting import Stamp, populate_region_vectors
from hotpants.pure.region_fitting import assign_region_weights, score_member_stars
from hotpants.pure.regions import (
    build_connected_regions,
    effective_min_npix,
    pixel_owner_indices,
    remove_star_from_region,
)


def _bases(rkernel=5):
    return kernel.calculate_kernel_basis((2 * rkernel + 1, 2 * rkernel + 1), [1.0], [2])


def test_build_and_split_by_diameter():
    xy = np.array([[10.0, 10.0], [12.0, 10.0], [50.0, 50.0]])
    sci = np.ones((80, 80), dtype=np.float32)
    rmap = build_connected_regions(xy, (80, 80), rss=3, max_diameter=40.0, flux_image=sci)
    assert len(rmap.regions) == 2
    multi = next(r for r in rmap.regions if len(r.member_xy) == 2)
    assert multi.npix > (2 * 3 + 1) ** 2


def test_voronoi_ownership_and_exclusion():
    xy = np.array([[20.0, 30.0], [28.0, 30.0]])
    sci = np.ones((60, 60), dtype=np.float32) * 10
    rmap = build_connected_regions(xy, (60, 60), rss=4, flux_image=sci)
    reg = rmap.regions[0]
    ys, xs = np.where(rmap.labels == reg.id)
    owners = pixel_owner_indices(ys, xs, reg.member_xy)
    assert set(owners.tolist()) == {0, 1}
    kick = reg.member_xy[0]
    rmap2 = remove_star_from_region(rmap, reg.id, kick, rss=4, flux_image=sci)
    r2 = next(r for r in rmap2.regions if r.id == reg.id)
    assert kick in r2.excluded_xy
    assert r2.npix < reg.npix
    # other star core still present
    assert rmap2.labels[30, 28] == reg.id


def test_populate_region_vectors_matches_mask():
    ny, nx = 64, 64
    rss = 4
    xy = np.array([[32.0, 32.0]])
    sci = np.random.default_rng(0).normal(10, 0.1, (ny, nx)).astype(np.float32)
    tpl = sci.copy()
    rmap = build_connected_regions(xy, (ny, nx), rss=rss, flux_image=sci)
    reg = rmap.regions[0]
    cfg = HotpantsConfig(nx=nx, ny=ny, rss=rss, rkernel=5, ko=0, bgo=0, deg_fixe=[2], sigma_gauss=[1.0])
    cfg.region_min_npix = 9
    stamp = Stamp(32, 32)
    ok = populate_region_vectors(stamp, tpl, sci, rmap.labels, reg, cfg, _bases(), 1)
    assert ok
    assert stamp.npix == int(np.sum(rmap.labels == reg.id))
    assert stamp.vectors.shape[1] == stamp.npix


def test_padded_bbox_rejects_near_edge():
    ny, nx = 40, 40
    rss = 4
    xy = np.array([[6.0, 20.0]])  # too close to left for rkernel pad
    sci = np.ones((ny, nx), dtype=np.float32)
    rmap = build_connected_regions(xy, (ny, nx), rss=rss, flux_image=sci)
    # may still build region; fill should fail due to template pad
    if not rmap.regions:
        return
    reg = rmap.regions[0]
    cfg = HotpantsConfig(nx=nx, ny=ny, rss=rss, rkernel=8, ko=0, bgo=0, deg_fixe=[1], sigma_gauss=[1.0])
    cfg.region_min_npix = 5
    stamp = Stamp(6, 20)
    ok = populate_region_vectors(stamp, sci, sci, rmap.labels, reg, cfg, _bases(8), 1)
    assert ok is False


def test_region_weights_npix_and_cap():
    class S:
        pass

    a, b = S(), S()
    a.npix, b.npix = 100, 25
    a.substamp = np.ones(100)
    b.substamp = np.ones(25)
    assign_region_weights([a, b], mode="npix", cap=(0.25, 4.0))
    assert a.region_weight == pytest.approx(100 / 62.5)
    assert b.region_weight == pytest.approx(25 / 62.5)


def test_blame_picks_corrupt_star():
    ny, nx = 60, 60
    rss = 4
    xy = np.array([[20.0, 30.0], [28.0, 30.0]])
    sci = np.ones((ny, nx), dtype=np.float32) * 10
    tpl = sci.copy()
    yy, xx = np.mgrid[-rss : rss + 1, -rss : rss + 1]
    psf = np.exp(-(xx**2 + yy**2) / 2.0).astype(np.float32)
    for x in (20, 28):
        sci[30 - rss : 30 + rss + 1, x - rss : x + rss + 1] += 50 * psf
        tpl[30 - rss : 30 + rss + 1, x - rss : x + rss + 1] += 50 * psf
    sci[30 - rss : 30 + rss + 1, 28 - rss : 28 + rss + 1] += 200

    rmap = build_connected_regions(xy, (ny, nx), rss=rss, flux_image=sci)
    reg = rmap.regions[0]
    cfg = HotpantsConfig(nx=nx, ny=ny, rss=rss, rkernel=5, ko=0, bgo=0, deg_fixe=[1], sigma_gauss=[1.0])
    cfg.region_min_npix = 20
    stamp = Stamp(int(reg.x_flux), int(reg.y_flux))
    assert populate_region_vectors(
        stamp, tpl, sci, rmap.labels, reg, cfg, _bases(), 1, noise_sq=np.ones_like(sci)
    )
    c = np.zeros(stamp.vectors.shape[0])
    c[0] = 1.0
    scores = score_member_stars(stamp, c, reg, fig_merit="v")
    assert scores[0][0] == (28, 30)


def test_c_extension_rejected():
    img = np.ones((32, 32), dtype=np.float32)
    cat = np.array([[16.0, 16.0]], dtype=np.float32)
    cfg = HotpantsConfig(nx=32, ny=32, stamp_mode="connected_regions")
    with pytest.raises(HotpantsError, match="pure Python"):
        Hotpants(img, img, star_catalog=cat, config=cfg, use_c_extension=True)


def test_effective_min_npix_default():
    cfg = HotpantsConfig(nx=10, ny=10, rss=4)
    assert effective_min_npix(cfg) == 81


def test_e2e_connected_synthetic():
    rng = np.random.default_rng(1)
    ny, nx = 100, 100
    sky = 100.0
    tpl = np.full((ny, nx), sky, dtype=np.float32)
    sci = np.full((ny, nx), sky, dtype=np.float32)
    stars = [(25, 25), (29, 25), (60, 60), (75, 40)]
    for x, y in stars:
        yy, xx = np.mgrid[-5:6, -5:6]
        psf = np.exp(-(xx**2 + yy**2) / (2 * 1.2**2)).astype(np.float32)
        tpl[y - 5 : y + 6, x - 5 : x + 6] += 60 * psf
        sci[y - 5 : y + 6, x - 5 : x + 6] += 90 * psf
    sci += rng.normal(0, 0.5, sci.shape).astype(np.float32)
    tpl += rng.normal(0, 0.5, tpl.shape).astype(np.float32)
    cat = np.array([[x + 1.0, y + 1.0] for x, y in stars], dtype=np.float32)
    cfg = HotpantsConfig(
        nx=nx,
        ny=ny,
        rss=4,
        rkernel=5,
        ko=1,
        bgo=0,
        force_convolve="t",
        stamp_mode="connected_regions",
        region_weight="uniform",
        region_min_npix=20,
        deg_fixe=[2],
        sigma_gauss=[1.0],
        kf_spread_mask1=0.0,
        iuthresh=1e9,
        tuthresh=1e9,
        iuktresh=1e9,
        tuktresh=1e9,
        verbose=0,
    )
    hp = Hotpants(tpl, sci, star_catalog=cat, config=cfg, use_c_extension=False)
    hp.find_stamps()
    assert len(hp.results["region_map"].regions) >= 2
    hp.fit_and_select_direction()
    sol, used = hp.iterative_fit_and_clip()
    assert sol is not None
    assert len(used) >= 1
