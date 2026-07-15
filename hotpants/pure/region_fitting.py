"""
Connected-region stamp filling helpers used by fit_kernel / FOM paths.
"""

from __future__ import annotations

import numpy as np

from .fitting import (
    Stamp,
    _local_coeffs_from_solution,
    _pack_c_layout_solution,
    _set_spatial_weights,
    _sigma_clip_mean_stdev,
    _tikhonov_solve,
    accumulate_normal_equations,
    populate_region_vectors,
)
from .regions import (
    effective_min_npix,
    pixel_owner_indices,
    remove_star_from_region,
    split_region,
)


def assign_region_weights(stamps, mode="uniform", cap=(0.25, 4.0)):
    """Set stamp.region_weight from npix/flux/invvar/uniform; clamp to cap."""
    mode = str(mode or "uniform").lower()
    lo, hi = (float(cap[0]), float(cap[1])) if cap is not None else (0.25, 4.0)
    if mode == "uniform" or not stamps:
        for s in stamps:
            s.region_weight = 1.0
        return
    if mode == "npix":
        vals = np.array(
            [float(getattr(s, "npix", None) or s.substamp.size) for s in stamps],
            dtype=np.float64,
        )
    elif mode == "flux":
        vals = np.array(
            [float(np.nansum(np.abs(s.substamp))) for s in stamps],
            dtype=np.float64,
        )
    elif mode == "invvar":
        vals = []
        for s in stamps:
            npx = getattr(s, "noise_pix", None)
            if npx is not None and np.any(np.asarray(npx) > 0):
                med = float(np.nanmedian(npx[np.asarray(npx) > 0]))
                vals.append(1.0 / med if med > 0 else 1.0)
            else:
                vals.append(1.0)
        vals = np.asarray(vals, dtype=np.float64)
    else:
        for s in stamps:
            s.region_weight = 1.0
        return
    med = float(np.median(vals[vals > 0])) if np.any(vals > 0) else 1.0
    if med <= 0:
        med = 1.0
    for s, v in zip(stamps, vals):
        s.region_weight = float(np.clip(float(v / med), lo, hi))


def stamp_sigma_region(stamp, coeffs, fig_merit="v", mask=None, FLAG_INPUT_ISBAD=0x80):
    model = coeffs @ stamp.vectors
    resid = stamp.substamp - model
    good = np.isfinite(resid) & (np.abs(stamp.substamp) > 1e-20)
    ys, xs = stamp.ys, stamp.xs
    if mask is not None and ys is not None:
        good &= (mask[ys, xs].astype(np.int32) & FLAG_INPUT_ISBAD) == 0
    if str(fig_merit).startswith("v"):
        ncut = getattr(stamp, "noise_pix", None)
        if ncut is None:
            return float(np.sqrt(np.mean(resid[good] * resid[good]))) if np.any(good) else -1.0
        ncut = np.asarray(ncut, dtype=np.float64)
        good &= np.isfinite(ncut) & (ncut > 0)
        if not np.any(good):
            return -1.0
        return float(np.mean((resid[good] ** 2) / ncut[good]))
    if not np.any(good):
        return -1.0
    return float(np.sqrt(np.mean(resid[good] * resid[good])))


def score_member_stars(stamp, coeffs, region, fig_merit="v"):
    model = coeffs @ stamp.vectors
    resid = stamp.substamp - model
    ys, xs = stamp.ys, stamp.xs
    active = [p for p in region.member_xy if p not in region.excluded_xy]
    if not active or ys is None:
        return []
    owners = pixel_owner_indices(ys, xs, active)
    noise = getattr(stamp, "noise_pix", None)
    scores = []
    for i, xy in enumerate(active):
        sel = owners == i
        if not np.any(sel):
            continue
        r = resid[sel]
        good = np.isfinite(r) & (np.abs(stamp.substamp[sel]) > 1e-20)
        if noise is not None:
            n = np.asarray(noise)[sel]
            good &= np.isfinite(n) & (n > 0)
            if not np.any(good):
                continue
            if str(fig_merit).startswith("v"):
                sig = float(np.mean((r[good] ** 2) / n[good]))
            else:
                sig = float(np.sqrt(np.mean(r[good] * r[good])))
        else:
            if not np.any(good):
                continue
            sig = float(np.sqrt(np.mean(r[good] * r[good])))
        scores.append((xy, sig))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def _sync_region_map(dst, src):
    """Update caller's RegionMap in place so core.results stays current."""
    dst.labels[:] = src.labels
    dst.regions[:] = list(src.regions)


def reject_region_star(
    stamp,
    region_map,
    template,
    image,
    config,
    kernel_vecs,
    oversample,
    ker_order,
    lr_nx,
    lr_ny,
    coeffs,
    input_mask,
    noise_sq,
    verbose=0,
):
    rid = stamp.region_id
    reg = next((r for r in region_map.regions if r.id == rid), None)
    if reg is None:
        stamp.ignore = True
        return region_map

    scores = score_member_stars(stamp, coeffs, reg, fig_merit=getattr(config, "fom", "v"))
    if not scores:
        stamp.ignore = True
        return region_map

    offender, off_sig = scores[0]
    top = off_sig
    close = [s for s in scores if s[1] >= 0.9 * top]
    if len(close) > 1:
        active = [p for p in reg.member_xy if p not in reg.excluded_xy]
        owners = pixel_owner_indices(stamp.ys, stamp.xs, active)

        def _frac(xy):
            return float(np.mean(owners == active.index(xy)))

        close.sort(key=lambda t: (_frac(t[0]), -t[1]))
        offender = close[0][0]

    rss = int(getattr(config, "region_rss", None) or config.rss)
    if verbose >= 2:
        print(f"    region {rid}: exclude star {offender} (sig≈{off_sig:.3f})")

    new_map = remove_star_from_region(region_map, rid, offender, rss, flux_image=image)
    _sync_region_map(region_map, new_map)
    reg = next((r for r in region_map.regions if r.id == rid), None)
    min_npix = effective_min_npix(config)
    n_bisects = int(getattr(stamp, "_n_bisects", 0) or 0)
    max_bisects = int(getattr(config, "region_max_bisects", 100))

    def _try_fill(r):
        return populate_region_vectors(
            stamp,
            template,
            image,
            region_map.labels,
            r,
            config,
            kernel_vecs,
            oversample,
            input_mask=input_mask,
            noise_sq=noise_sq,
        )

    if reg is not None and reg.npix >= min_npix and any(p not in reg.excluded_xy for p in reg.member_xy):
        if _try_fill(reg):
            _set_spatial_weights(stamp, ker_order, lr_nx, lr_ny)
            stamp.ignore = False
            return region_map

    if (
        getattr(config, "region_bisect_on_reject", False)
        and reg is not None
        and n_bisects < max_bisects
        and sum(1 for p in reg.member_xy if p not in reg.excluded_xy) >= 2
    ):
        if verbose >= 2:
            print(f"    region {rid}: bisect after failed exclusion")
        old_members = set(reg.member_xy)
        new_map = split_region(region_map, rid, rss, flux_image=image)
        _sync_region_map(region_map, new_map)
        stamp._n_bisects = n_bisects + 1
        children = [
            r
            for r in region_map.regions
            if set(r.member_xy).issubset(old_members) or (set(r.member_xy) & old_members)
        ]
        for child in children:
            stamp.region_id = child.id
            if _try_fill(child):
                _set_spatial_weights(stamp, ker_order, lr_nx, lr_ny)
                stamp.ignore = False
                return region_map

    stamp.ignore = True
    return region_map


def fit_kernel_regions(
    stamps,
    template,
    image,
    config,
    kernel_vecs,
    region_map,
    oversample=1,
    verbose=0,
    skip_local_reject=False,
    noise_sq=None,
    mask=None,
):
    """Iterative global fit for connected-region irregular stamps."""
    FLAG_INPUT_ISBAD = 0x80

    if stamps and isinstance(stamps[0], (list, tuple)):
        stamp_groups = stamps
    else:
        stamp_groups = [[s] for s in stamps]

    n_comp_ker = len(kernel_vecs)
    ker_order = config.ko if hasattr(config, "ko") else 2
    bg_order = config.bgo
    ker_sig_reject = float(getattr(config, "ks", 2.0))
    stat_sig = float(getattr(config, "stat_sig", 3.0))
    fig_merit = str(getattr(config, "fom", "v") or "v")
    n_spatial = (ker_order + 1) * (ker_order + 2) // 2
    n_bg = (bg_order + 1) * (bg_order + 2) // 2
    n_params = (n_comp_ker - 1) * n_spatial + n_bg + 1
    n_comp_total = getattr(config, "n_comp_total", n_comp_ker * n_spatial + n_bg)
    ncomp_ker_layout = int(getattr(config, "ncomp_ker", n_comp_ker))
    lambda_reg = float(getattr(config, "lambda_reg", 0.0) or 0.0)
    if oversample > 1 and lambda_reg > 0:
        lambda_reg *= float(oversample) ** 2

    ny, nx = template.shape
    lr_ny = ny // oversample if oversample > 1 else ny
    lr_nx = nx // oversample if oversample > 1 else nx

    regions_by_id = {r.id: r for r in region_map.regions}
    valid_stamps = []
    for gidx, group in enumerate(stamp_groups):
        if not group:
            continue
        rid = getattr(group[0], "region_id", None)
        if rid is None:
            rid = getattr(group[0], "stamp_group_id", None)
        reg = regions_by_id.get(int(rid)) if rid is not None else None
        if reg is None:
            continue
        stamp = Stamp(int(round(reg.x_flux)), int(round(reg.y_flux)), orig_idx=gidx)
        stamp.region_id = int(reg.id)
        stamp.nss = 1
        stamp.sscnt = 0
        stamp.substamp_coords = [(stamp.x, stamp.y)]
        ok = populate_region_vectors(
            stamp,
            template,
            image,
            region_map.labels,
            reg,
            config,
            kernel_vecs,
            oversample,
            input_mask=mask,
            noise_sq=noise_sq,
        )
        if not ok:
            continue
        _set_spatial_weights(stamp, ker_order, lr_nx, lr_ny)
        valid_stamps.append(stamp)

    assign_region_weights(
        valid_stamps,
        mode=getattr(config, "region_weight", "uniform"),
        cap=getattr(config, "region_weight_cap", (0.25, 4.0)),
    )

    if not skip_local_reject:
        local_ok = []
        for stamp in valid_stamps:
            n_fit = n_comp_ker + 1
            vectors_fit = stamp.vectors[:n_fit]
            try:
                coeffs_fit = np.linalg.solve(
                    vectors_fit @ vectors_fit.T, vectors_fit @ stamp.substamp
                )
                stamp.norm = float(coeffs_fit[0])
                stamp.local_solution = coeffs_fit
                local_ok.append(stamp)
            except np.linalg.LinAlgError:
                stamp.ignore = True
        if len(local_ok) >= 3:
            k_sums = np.array([s.norm for s in local_ok], dtype=np.float64)
            kmean, kstdev = _sigma_clip_mean_stdev(k_sums, stat_sig=stat_sig, maxiter=10)
            for s in local_ok:
                s.diff = abs((s.norm - kmean) / kstdev) if kstdev > 0 and kstdev < 1e29 else 0.0
                s.ignore = s.diff >= ker_sig_reject

    solution = None
    packed = None
    final_active_stamps = []
    fit_stats = {"meansig": 0.0, "scatter": 0.0, "n_skipped": 0}

    for iteration in range(10):
        active_stamps = [s for s in valid_stamps if not s.ignore]
        final_active_stamps = active_stamps
        n_active = len(active_stamps)
        if n_active == 0:
            if verbose >= 1:
                print("No region stamps left!")
            break
        if verbose >= 2:
            print(f"DEBUG: Region iter {iteration} - Active: {n_active}")
        elif verbose >= 1 and iteration == 0:
            print(f"Region fit: {n_active} active stamps, assembling A/b…", flush=True)

        t_asm = __import__("time").time()
        A, b = accumulate_normal_equations(
            active_stamps, n_comp_ker, n_spatial, n_bg, n_params
        )
        if verbose >= 1 and iteration == 0:
            print(f"  assemble done in {__import__('time').time() - t_asm:.1f}s; solving…", flush=True)
        try:
            solution = _tikhonov_solve(A, b, lambda_reg=lambda_reg)
            packed = _pack_c_layout_solution(
                solution, n_comp_ker, n_spatial, n_bg, n_comp_total, ncomp_ker_layout
            )
            fit_stats["lambda_reg"] = lambda_reg
            fit_stats["solution_maxabs"] = float(np.max(np.abs(solution)))
            fit_stats["solution_mat"] = np.asarray(solution, dtype=np.float64).copy()
        except np.linalg.LinAlgError:
            print("Singular matrix")
            break

        sigmas = []
        bad_fill = []
        for si, s in enumerate(active_stamps):
            c = _local_coeffs_from_solution(solution, s.weights, n_comp_ker, n_spatial, n_bg)
            sigma = stamp_sigma_region(s, c, fig_merit, mask, FLAG_INPUT_ISBAD)
            if sigma < 0:
                bad_fill.append(si)
                sigmas.append(np.nan)
            else:
                sigmas.append(sigma)
                s.chi2 = sigma

        need_refit = False
        for si in bad_fill:
            s = active_stamps[si]
            c = _local_coeffs_from_solution(solution, s.weights, n_comp_ker, n_spatial, n_bg)
            reject_region_star(
                s,
                region_map,
                template,
                image,
                config,
                kernel_vecs,
                oversample,
                ker_order,
                lr_nx,
                lr_ny,
                c,
                mask,
                noise_sq,
                verbose=verbose,
            )
            need_refit = True

        good_sigmas = np.array([sg for sg in sigmas if np.isfinite(sg)], dtype=np.float64)
        if good_sigmas.size == 0:
            break

        mean_sig, std_sig = _sigma_clip_mean_stdev(good_sigmas, stat_sig=stat_sig, maxiter=10)
        fit_stats["meansig"] = mean_sig
        fit_stats["scatter"] = std_sig
        fit_stats["n_skipped"] = sum(1 for s in valid_stamps if s.ignore)

        if std_sig == 0 or std_sig >= 1e29:
            if not need_refit:
                break
        else:
            for s, sigma in zip(active_stamps, sigmas):
                if not np.isfinite(sigma):
                    continue
                if (sigma - mean_sig) > ker_sig_reject * std_sig:
                    if verbose >= 2:
                        print(
                            f"    region ({s.x},{s.y}) sig={sigma:.3f} outlier; exclude star"
                        )
                    c = _local_coeffs_from_solution(
                        solution, s.weights, n_comp_ker, n_spatial, n_bg
                    )
                    reject_region_star(
                        s,
                        region_map,
                        template,
                        image,
                        config,
                        kernel_vecs,
                        oversample,
                        ker_order,
                        lr_nx,
                        lr_ny,
                        c,
                        mask,
                        noise_sq,
                        verbose=verbose,
                    )
                    need_refit = True

        if need_refit:
            assign_region_weights(
                [s for s in valid_stamps if not s.ignore],
                mode=getattr(config, "region_weight", "uniform"),
                cap=getattr(config, "region_weight_cap", (0.25, 4.0)),
            )
        else:
            break

    if solution is not None:
        for s in final_active_stamps:
            if s.ignore:
                continue
            c = _local_coeffs_from_solution(solution, s.weights, n_comp_ker, n_spatial, n_bg)
            s.convolved_model_global = c @ s.vectors

    for s in final_active_stamps:
        s.fit_stats = fit_stats

    out_sol = packed if packed is not None else solution
    return out_sol, final_active_stamps
