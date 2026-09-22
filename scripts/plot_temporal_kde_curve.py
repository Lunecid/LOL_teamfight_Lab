"""Reconstruct the pooled kill-gap KDE curve from the locked full-corpus pairs.

Not a new experiment: reloads ``fight_boundary_full/pairs_*.npz``, recomputes the
same ``kde_valley`` (bandwidth 0.08, grid (-0.5, 3.0, 1401), seed 7, max 1e6),
checks that modes/valley match ``config/fight_boundary/details_pooled.json``,
and writes the density curve figure for thesis §4.1.

Example:
    python scripts/plot_temporal_kde_curve.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOL_OUTPUT_ROOT", "D:/LOL_Project")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from analysis.kill_pairs import MatchRecord  # noqa: E402
from analysis.temporal_boundary import MAX_POINTS, kde_valley, mixture_crossing  # noqa: E402

LOCK = ROOT / "config" / "fight_boundary" / "details_pooled.json"
DEFAULT_PAIRS = Path(r"D:/LOL_Project/fusion_2615/features/fight_boundary_full")
OUT_DIR = ROOT / "config" / "fight_boundary"

# Locked values (must match exactly under the same subsample seed)
TOL_S = 1e-9


def _load_records(pairs_dir: Path) -> list[MatchRecord]:
    records: list[MatchRecord] = []
    for path in sorted(pairs_dir.glob("pairs_*.npz")):
        Z = np.load(path, allow_pickle=True)
        batch = [
            MatchRecord(str(m), str(p), np.asarray(t, dtype=np.float64), int(n))
            for m, p, t, n in zip(Z["rec_match_id"], Z["rec_patch"], Z["rec_ts"], Z["rec_n_kills"])
        ]
        records.extend(batch)
        print(f"loaded {path.name}: {len(batch):,} matches", flush=True)
    return records


def _kde_curve(log_dt: np.ndarray, bandwidth: float, seed: int, grid: tuple):
    """Same fit path as ``kde_valley``, but also returns (g, f) on the log10 grid."""
    from scipy.stats import gaussian_kde

    x = np.asarray(log_dt, dtype=np.float64)
    x = x[np.isfinite(x)]
    n_all = int(x.size)
    if MAX_POINTS and n_all > MAX_POINTS:
        x = np.random.default_rng(seed).choice(x, MAX_POINTS, replace=False)
    mix = mixture_crossing(x, seed=seed, max_points=None)  # already subsampled
    lo_b, hi_b = np.log10(mix["means_s"][0]), np.log10(mix["means_s"][1])
    kde = gaussian_kde(x, bw_method=bandwidth)
    g = np.linspace(*grid)
    f = kde(g)
    mid = 0.5 * (lo_b + hi_b)
    left = np.where(g <= mid)[0]
    right = np.where(g > mid)[0]
    m1 = left[int(np.argmax(f[left]))]
    m2 = right[int(np.argmax(f[right]))]
    between = np.arange(m1, m2 + 1)
    vi = between[int(np.argmin(f[between]))]
    return {
        "g_log10": g,
        "density": f,
        "mode1_s": float(10 ** g[m1]),
        "mode2_s": float(10 ** g[m2]),
        "valley_s": float(10 ** g[vi]),
        "depth": float(f[vi] / max(min(f[m1], f[m2]), 1e-12)),
        "n_all": n_all,
        "n_fit": int(x.size),
        "bandwidth": bandwidth,
        "search_lo_s": float(10 ** lo_b),
        "search_hi_s": float(10 ** hi_b),
        "idx_mode1": int(m1),
        "idx_mode2": int(m2),
        "idx_valley": int(vi),
    }


def _figure(curve: dict, lock_kde: dict, plateau_s: list[float], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, INK, INK2, GRID, BAND = (
        "#2a78d6", "#eb6834", "#1a1a19", "#5a5a58", "#e6e6e3", "#d9e8f7",
    )
    plt.rcParams.update({
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
    })
    g = curve["g_log10"]
    f = curve["density"]
    s = 10 ** g

    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor="#fcfcfb")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.14)

    # ARI >= 0.9 plateau (seconds), shaded on the density curve
    lo_p, hi_p = plateau_s
    mask = (s >= lo_p) & (s <= hi_p)
    ax.fill_between(s[mask], 0, f[mask], color=BAND, alpha=0.85, label=f"ARI ≥ 0.9 ({lo_p:.0f}–{hi_p:.0f} s)")

    ax.plot(s, f, color=BLUE, lw=1.6, label="KDE (bw = 0.08)")

    m1, m2, v = curve["mode1_s"], curve["mode2_s"], curve["valley_s"]
    i1, i2, iv = curve["idx_mode1"], curve["idx_mode2"], curve["idx_valley"]
    ax.plot([m1], [f[i1]], "o", color=BLUE, ms=6, zorder=5)
    ax.plot([m2], [f[i2]], "o", color=BLUE, ms=6, zorder=5)
    ax.plot([v], [f[iv]], "o", color=ORANGE, ms=7, zorder=5)

    ax.axvline(m1, color=BLUE, ls=":", lw=0.9, alpha=0.7)
    ax.axvline(m2, color=BLUE, ls=":", lw=0.9, alpha=0.7)
    ax.axvline(v, color=ORANGE, ls="--", lw=1.1)

    ymax = float(f.max()) * 1.08
    ax.text(m1, ymax * 0.96, f"mode₁\n{m1:.2f} s", ha="center", va="top", color=BLUE, fontsize=8)
    ax.text(m2, ymax * 0.96, f"mode₂\n{m2:.2f} s", ha="center", va="top", color=BLUE, fontsize=8)
    ax.text(v * 1.08, f[iv] + 0.02 * ymax, f"valley G = {v:.2f} s", ha="left", va="bottom",
            color=ORANGE, fontsize=8)

    ax.set_xscale("log")
    ax.set_xlim(10 ** (-0.5), 10 ** 3.0)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Inter-kill interval (s, log scale)")
    ax.set_ylabel("Density (log₁₀ interval)")
    ax.set_title(
        f"Pooled kill-gap KDE  ·  n = {lock_kde['n']:,} intervals  ·  "
        f"depth = {curve['depth']:.3f}",
        color=INK, pad=8,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_facecolor("#fcfcfb")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs-dir", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--lock", type=Path, default=LOCK)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--bandwidth", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", type=float, nargs=3, default=(-0.5, 3.0, 1401.0))
    args = ap.parse_args(argv)

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    lock_kde = lock["temporal"]["kde"]
    plateau = lock["temporal"]["plateau_s"]

    records = _load_records(args.pairs_dir)
    log_dt = np.concatenate([r.log_dt for r in records if r.log_dt.size])
    print(f"pooled matches={len(records):,} intervals={log_dt.size:,}", flush=True)

    # Sanity via the public estimator (modes/valley only)
    est = kde_valley(log_dt, bandwidth=args.bandwidth, seed=args.seed,
                     grid=(args.grid[0], args.grid[1], int(args.grid[2])))
    print("kde_valley:", est.as_dict(), flush=True)

    for key, got in [("mode1_s", est.mode1_s), ("mode2_s", est.mode2_s),
                     ("valley_s", est.valley_s), ("depth", est.depth)]:
        want = float(lock_kde[key])
        if abs(got - want) > TOL_S:
            raise SystemExit(f"MISMATCH {key}: got {got!r} want {want!r} (tol {TOL_S})")
    if int(est.n) != int(lock_kde["n"]):
        raise SystemExit(f"MISMATCH n: got {est.n} want {lock_kde['n']}")
    print("lock check OK (modes, valley, depth, n)", flush=True)

    grid = (args.grid[0], args.grid[1], int(args.grid[2]))
    curve = _kde_curve(log_dt, bandwidth=args.bandwidth, seed=args.seed, grid=grid)
    for key in ("mode1_s", "mode2_s", "valley_s", "depth"):
        if abs(curve[key] - float(lock_kde[key])) > TOL_S:
            raise SystemExit(f"curve MISMATCH {key}: {curve[key]!r} vs {lock_kde[key]!r}")

    out_png = args.out_dir / "temporal_kde_pooled.png"
    out_npz = args.out_dir / "temporal_kde_pooled.npz"
    out_json = args.out_dir / "temporal_kde_pooled_meta.json"

    _figure(curve, lock_kde, plateau, out_png)
    np.savez_compressed(
        out_npz,
        g_log10=curve["g_log10"],
        density=curve["density"],
        seconds=10 ** curve["g_log10"],
    )
    meta = {
        "source_pairs_dir": str(args.pairs_dir),
        "lock": str(args.lock),
        "n_matches": len(records),
        "n_intervals": int(curve["n_all"]),
        "n_fit": int(curve["n_fit"]),
        "bandwidth": args.bandwidth,
        "seed": args.seed,
        "grid": list(grid),
        "mode1_s": curve["mode1_s"],
        "mode2_s": curve["mode2_s"],
        "valley_s": curve["valley_s"],
        "depth": curve["depth"],
        "plateau_s": plateau,
        "lock_match": True,
        "png": str(out_png),
        "npz": str(out_npz),
    }
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_npz}", flush=True)
    print(f"wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
