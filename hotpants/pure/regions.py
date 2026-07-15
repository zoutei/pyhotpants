"""
Connected region stamps: paint catalog rss boxes, label, split by diameter,
and support star-footprint exclusion for iterative rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

STRUCT8 = np.ones((3, 3), dtype=bool)


@dataclass
class Region:
    id: int
    npix: int
    y0: int
    y1: int
    x0: int
    x1: int
    x_cen: float
    y_cen: float
    x_flux: float
    y_flux: float
    diameter: float
    weight: float = 1.0
    member_xy: List[Tuple[int, int]] = field(default_factory=list)
    excluded_xy: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class RegionMap:
    labels: np.ndarray
    regions: List[Region]


def _paint_boxes(ny: int, nx: int, xs: np.ndarray, ys: np.ndarray, rss: int) -> np.ndarray:
    mask = np.zeros((ny, nx), dtype=bool)
    for xi, yi in zip(xs.astype(int), ys.astype(int)):
        y0, y1 = max(0, yi - rss), min(ny, yi + rss + 1)
        x0, x1 = max(0, xi - rss), min(nx, xi + rss + 1)
        mask[y0:y1, x0:x1] = True
    return mask


def _component_diameter(label_map: np.ndarray, lid: int) -> float:
    ys, xs = np.where(label_map == lid)
    if ys.size == 0:
        return 0.0
    dy = float(ys.max() - ys.min())
    dx = float(xs.max() - xs.min())
    return float(max(dy, dx, np.hypot(dy, dx)))


def _stars_for_label(label_map: np.ndarray, xs: np.ndarray, ys: np.ndarray, lid: int) -> np.ndarray:
    at = label_map[ys, xs] == lid
    if np.any(at):
        return at
    ys_c, xs_c = np.where(label_map == lid)
    if ys_c.size == 0:
        return np.zeros(len(xs), dtype=bool)
    return (xs >= xs_c.min()) & (xs <= xs_c.max()) & (ys >= ys_c.min()) & (ys <= ys_c.max())


def _split_star_groups(
    star_idx: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    max_diameter: float,
    rss: int,
) -> List[np.ndarray]:
    if star_idx.size <= 1:
        return [star_idx]
    sx, sy = xs[star_idx], ys[star_idx]
    dy = float(sy.max() - sy.min()) + 2 * rss
    dx = float(sx.max() - sx.min()) + 2 * rss
    extent = max(dy, dx, float(np.hypot(dy, dx)))
    if extent <= max_diameter:
        return [star_idx]
    if dx >= dy:
        med = float(np.median(sx))
        left, right = star_idx[sx <= med], star_idx[sx > med]
    else:
        med = float(np.median(sy))
        left, right = star_idx[sy <= med], star_idx[sy > med]
    if left.size == 0 or right.size == 0:
        return [star_idx]
    out: List[np.ndarray] = []
    for part in (left, right):
        out.extend(_split_star_groups(part, xs, ys, max_diameter, rss))
    return out


def _paint_exclusive_labels(
    ny: int, nx: int, xs: np.ndarray, ys: np.ndarray, groups: List[np.ndarray], rss: int
) -> np.ndarray:
    labels = np.zeros((ny, nx), dtype=np.int32)
    for rid, star_idx in enumerate(groups, start=1):
        for i in star_idx:
            xi, yi = int(xs[i]), int(ys[i])
            y0, y1 = max(0, yi - rss), min(ny, yi + rss + 1)
            x0, x1 = max(0, xi - rss), min(nx, xi + rss + 1)
            sub = labels[y0:y1, x0:x1]
            sub[sub == 0] = rid
    return labels


def _flux_centroid(
    sci: Optional[np.ndarray], ys: np.ndarray, xs: np.ndarray
) -> Tuple[float, float, float, float]:
    x_cen = float(xs.mean()) if xs.size else 0.0
    y_cen = float(ys.mean()) if ys.size else 0.0
    if sci is None or xs.size == 0:
        return x_cen, y_cen, x_cen, y_cen
    vals = sci[ys, xs].astype(np.float64)
    w = np.where(np.isfinite(vals), np.abs(vals), 0.0)
    s = float(w.sum())
    if s <= 0:
        return x_cen, y_cen, x_cen, y_cen
    return x_cen, y_cen, float(np.sum(xs * w) / s), float(np.sum(ys * w) / s)


def _region_from_label(
    labels: np.ndarray,
    rid: int,
    member_xy: Sequence[Tuple[int, int]],
    flux_image: Optional[np.ndarray] = None,
    excluded_xy: Optional[List[Tuple[int, int]]] = None,
) -> Optional[Region]:
    ys, xs = np.where(labels == rid)
    if ys.size == 0:
        return None
    x_cen, y_cen, x_flux, y_flux = _flux_centroid(flux_image, ys, xs)
    dy = float(ys.max() - ys.min())
    dx = float(xs.max() - xs.min())
    return Region(
        id=rid,
        npix=int(ys.size),
        y0=int(ys.min()),
        y1=int(ys.max()) + 1,
        x0=int(xs.min()),
        x1=int(xs.max()) + 1,
        x_cen=x_cen,
        y_cen=y_cen,
        x_flux=x_flux,
        y_flux=y_flux,
        diameter=float(max(dy, dx, np.hypot(dy, dx))),
        member_xy=list(member_xy),
        excluded_xy=list(excluded_xy or []),
    )


def build_connected_regions(
    xy: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    rss: int,
    max_diameter: float = 40.0,
    max_area: int = 0,
    connectivity: int = 8,
    flux_image: Optional[np.ndarray] = None,
) -> RegionMap:
    """
    Build exclusive label map from catalog star centers (N,2) in 0-based LR pixels.
    """
    ny, nx = image_shape
    xy = np.asarray(xy, dtype=np.float64)
    if xy.size == 0:
        return RegionMap(labels=np.zeros((ny, nx), dtype=np.int32), regions=[])
    xs = np.rint(xy[:, 0]).astype(int)
    ys = np.rint(xy[:, 1]).astype(int)
    ok = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny)
    xs, ys = xs[ok], ys[ok]
    if xs.size == 0:
        return RegionMap(labels=np.zeros((ny, nx), dtype=np.int32), regions=[])

    occupied = _paint_boxes(ny, nx, xs, ys, rss)
    struct = STRUCT8 if connectivity >= 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled, n_comp = ndimage.label(occupied, structure=struct)

    groups: List[np.ndarray] = []
    for lid in range(1, n_comp + 1):
        members = np.flatnonzero(_stars_for_label(labeled, xs, ys, lid))
        if members.size == 0:
            continue
        if _component_diameter(labeled, lid) > max_diameter:
            groups.extend(_split_star_groups(members, xs, ys, max_diameter, rss))
        else:
            groups.append(members)

    # Optional area split: re-split groups whose painted exclusive area is huge
    if max_area and max_area > 0:
        refined: List[np.ndarray] = []
        tmp = _paint_exclusive_labels(ny, nx, xs, ys, groups, rss)
        for g in groups:
            rid_probe = 1  # temporary: measure via star extent
            # approximate area from exclusive paint of this group alone
            solo = _paint_exclusive_labels(ny, nx, xs, ys, [g], rss)
            npix = int(np.count_nonzero(solo))
            if npix > max_area and g.size > 1:
                refined.extend(_split_star_groups(g, xs, ys, max(1.0, max_diameter * 0.7), rss))
            else:
                refined.append(g)
        groups = refined
        del tmp

    labels = _paint_exclusive_labels(ny, nx, xs, ys, groups, rss)
    regions: List[Region] = []
    for rid, star_idx in enumerate(groups, start=1):
        members = [(int(xs[i]), int(ys[i])) for i in star_idx]
        reg = _region_from_label(labels, rid, members, flux_image=flux_image)
        if reg is not None:
            regions.append(reg)
    return RegionMap(labels=labels, regions=regions)


def region_pixel_coords(labels: np.ndarray, region_id: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.where(labels == region_id)


def pixel_owner_indices(
    ys: np.ndarray,
    xs: np.ndarray,
    member_xy: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """For each pixel, index of nearest member star (Voronoi)."""
    if not member_xy:
        return np.full(ys.shape, -1, dtype=np.int32)
    mx = np.array([p[0] for p in member_xy], dtype=np.float64)
    my = np.array([p[1] for p in member_xy], dtype=np.float64)
    # (Npix, Nmem)
    d2 = (xs[:, None] - mx[None, :]) ** 2 + (ys[:, None] - my[None, :]) ** 2
    return np.argmin(d2, axis=1).astype(np.int32)


def remove_star_from_region(
    region_map: RegionMap,
    region_id: int,
    star_xy: Tuple[int, int],
    rss: int,
    flux_image: Optional[np.ndarray] = None,
) -> RegionMap:
    """
    Clear pixels owned by star_xy (nearest-member Voronoi among remaining members)
    from the region label; update Region metadata.
    """
    labels = region_map.labels.copy()
    regions = []
    sx, sy = int(star_xy[0]), int(star_xy[1])

    for reg in region_map.regions:
        if reg.id != region_id:
            regions.append(reg)
            continue
        active = [p for p in reg.member_xy if p not in reg.excluded_xy and not (p[0] == sx and p[1] == sy)]
        ys, xs = np.where(labels == region_id)
        if ys.size == 0 or not reg.member_xy:
            labels[labels == region_id] = 0
            continue

        # Ownership among members still active *before* removal (include the kicked star)
        members_before = [p for p in reg.member_xy if p not in reg.excluded_xy]
        owners = pixel_owner_indices(ys, xs, members_before)
        # Index of kicked star in members_before
        kick_idx = None
        for i, p in enumerate(members_before):
            if p[0] == sx and p[1] == sy:
                kick_idx = i
                break
        if kick_idx is None:
            # also clear paint box ∩ region as fallback
            y0, y1 = max(0, sy - rss), min(labels.shape[0], sy + rss + 1)
            x0, x1 = max(0, sx - rss), min(labels.shape[1], sx + rss + 1)
            patch = labels[y0:y1, x0:x1]
            patch[patch == region_id] = 0
        else:
            clear = owners == kick_idx
            labels[ys[clear], xs[clear]] = 0

        excluded = list(reg.excluded_xy) + [(sx, sy)]
        new_reg = _region_from_label(labels, region_id, active, flux_image=flux_image, excluded_xy=excluded)
        if new_reg is not None:
            regions.append(new_reg)
        # if None, region emptied — drop from list; labels already cleared for those pixels

    return RegionMap(labels=labels, regions=regions)


def split_region(
    region_map: RegionMap,
    region_id: int,
    rss: int,
    flux_image: Optional[np.ndarray] = None,
    axis: Optional[int] = None,
) -> RegionMap:
    """
    Bisect a region along the longer axis using median of remaining member centers.
    Replaces one region id with two new ids (max_id+1, max_id+2).
    """
    labels = region_map.labels.copy()
    reg = next((r for r in region_map.regions if r.id == region_id), None)
    if reg is None:
        return region_map
    members = [p for p in reg.member_xy if p not in reg.excluded_xy]
    if len(members) < 2:
        return region_map

    mx = np.array([p[0] for p in members], dtype=float)
    my = np.array([p[1] for p in members], dtype=float)
    if axis is None:
        axis = 0 if (mx.max() - mx.min()) >= (my.max() - my.min()) else 1
    if axis == 0:
        med = float(np.median(mx))
        left = [p for p in members if p[0] <= med]
        right = [p for p in members if p[0] > med]
    else:
        med = float(np.median(my))
        left = [p for p in members if p[1] <= med]
        right = [p for p in members if p[1] > med]
    if not left or not right:
        return region_map

    max_id = int(labels.max())
    id_a, id_b = max_id + 1, max_id + 2
    ys, xs = np.where(labels == region_id)
    owners = pixel_owner_indices(ys, xs, members)
    # map owner index -> left/right
    left_set = set(left)
    for i, p in enumerate(members):
        mask_i = owners == i
        new_id = id_a if p in left_set else id_b
        labels[ys[mask_i], xs[mask_i]] = new_id

    new_regions = [r for r in region_map.regions if r.id != region_id]
    for nid, mems in ((id_a, left), (id_b, right)):
        nr = _region_from_label(labels, nid, mems, flux_image=flux_image)
        if nr is not None:
            new_regions.append(nr)
    return RegionMap(labels=labels, regions=new_regions)


def effective_min_npix(config) -> int:
    rss = int(getattr(config, "region_rss", None) or config.rss)
    mn = getattr(config, "region_min_npix", None)
    if mn is None:
        return (2 * rss + 1) ** 2
    return int(mn)


def gate_catalog_stars_for_regions(
    catalog_xy: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    rss: int,
    rkernel: int,
    ukthresh: Optional[float] = None,
    fitthresh: float = 5.0,
) -> np.ndarray:
    """
    Phase-0-style gate: border + rss box free of FLAG_INPUT_ISBAD
    and finite pixels. Optional ukthresh rejects saturated boxes.
    Returns (N,2) 0-based survivors.
    """
    from .utils import FLAG_INPUT_ISBAD

    del fitthresh  # reserved for future PSF-style soft gate
    xy = np.asarray(catalog_xy, dtype=np.float64)
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    ny, nx = image.shape
    border = int(rkernel) + int(rss) + 1
    bad_bits = FLAG_INPUT_ISBAD
    keep = []
    for x, y in xy:
        xi, yi = int(round(x)), int(round(y))
        if xi < border or yi < border or xi >= nx - border or yi >= ny - border:
            continue
        y0, y1 = yi - rss, yi + rss + 1
        x0, x1 = xi - rss, xi + rss + 1
        patch_m = mask[y0:y1, x0:x1].astype(np.int32)
        if np.any((patch_m & bad_bits) != 0):
            continue
        patch = image[y0:y1, x0:x1]
        if not np.all(np.isfinite(patch)):
            continue
        # Only apply saturation cut when ukthresh is finite and above image median
        # (avoids rejecting all stars when tuktresh was copied from template max).
        if ukthresh is not None and np.isfinite(ukthresh):
            med = float(np.nanmedian(image))
            if float(ukthresh) > med and np.any(patch >= float(ukthresh)):
                continue
        keep.append((xi, yi))
    if not keep:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(keep, dtype=np.float64)


def find_stamps_connected_regions(
    template,
    image,
    mask,
    catalog,
    config,
    oversample: int = 1,
    flux_image=None,
):
    """
    Build one substamp dict per connected region from a gated catalog.

    catalog / returned coords are 0-based LR pixels (Hotpants convention after -1).
    Returns (t_substamps, i_substamps, region_map) where i_substamps is empty
    when force_convolve='t' or oversample>1.
    """
    rss = int(getattr(config, "region_rss", None) or config.rss)
    rkernel = int(config.rkernel)
    force = str(getattr(config, "force_convolve", "b"))
    fitthresh = float(getattr(config, "fitthresh", 5.0))

    # Gate on science (LR) for painting — mask-clear + border (Phase-0 style).
    # Saturation is already reflected in the HOTPANTS input mask.
    gated = gate_catalog_stars_for_regions(
        catalog,
        image,
        mask if mask.shape == image.shape else mask,
        rss=rss,
        rkernel=rkernel,
        ukthresh=None,
        fitthresh=fitthresh,
    )
    flux = flux_image if flux_image is not None else image
    rmap = build_connected_regions(
        gated,
        image.shape,
        rss=rss,
        max_diameter=float(getattr(config, "region_max_diameter", 40.0)),
        max_area=int(getattr(config, "region_max_area", 0) or 0),
        connectivity=int(getattr(config, "region_connectivity", 8)),
        flux_image=flux,
    )
    min_npix = effective_min_npix(config)
    # Drop tiny edge-clipped regions at build time (rare)
    kept = [r for r in rmap.regions if r.npix >= min_npix]
    # Rebuild labels to only kept ids? Keep labels as-is; just skip tiny in list.
    rmap = RegionMap(labels=rmap.labels, regions=kept)

    t_out = []
    if force != "i":
        for i, reg in enumerate(rmap.regions):
            t_out.append(
                {
                    "substamp_id": i,
                    "stamp_group_id": int(reg.id),
                    "x": int(round(reg.x_flux)),
                    "y": int(round(reg.y_flux)),
                    "region_id": int(reg.id),
                }
            )
    i_out = []
    if force != "t" and int(oversample) == 1:
        for i, reg in enumerate(rmap.regions):
            i_out.append(
                {
                    "substamp_id": i,
                    "stamp_group_id": int(reg.id),
                    "x": int(round(reg.x_flux)),
                    "y": int(round(reg.y_flux)),
                    "region_id": int(reg.id),
                }
            )
    return t_out, i_out, rmap
