#!/usr/bin/env python3
"""Phase-1 reviewer-response audit: p_pre shortcut vs engagement q.

Reads frozen ΔV-engagement artefacts under the sibling LOL_Teamfight data root
(default: Documents/LOL_Teamfight) and writes a self-contained report under
this repository's outputs/. Does not retrain V or the sealed q winners.

Questions answered
------------------
1. How much of q's ranking comes from p_pre alone (raw p_pre, isotonic b(p),
   sealed old p_pre spline/logistic) versus PT and full LightGBM?
2. Inside narrow p_pre bins (and B40/B45), does q still beat a p_pre-only score?
3. How does P(Y=1 | p_pre) move with p_pre? (asymmetric-range hypothesis check)
4. How is V calibrated by game-time band on match wins W? (reuse published V eval)

Exploratory: prior TEST exposure; labels are model-defined sign(ΔV), not ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"scikit-learn required: {e}") from e


# --------------------------------------------------------------------------- paths
def _default_data_root() -> Path:
    docs = Path.home() / "Documents" / "LOL_Teamfight"
    # Windows localized "Documents" folder name
    alt = Path.home() / "문서" / "LOL_Teamfight"
    for p in (docs, alt):
        if (p / "outputs" / "incremental_q_training_20260915").is_dir():
            return p
    return alt if alt.exists() else docs


REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = _default_data_root()
DEFAULT_OUT = REPO / "outputs" / "reviewer_response_phase1_20260919"

IQ = "incremental_q_training_20260915"
FC = "full_corpus_training_20260915"
CR = "cohort_role_training_20260915"

BOOT_REPS = 1000
BOOT_SEED = 20260919
MIN_MATCHES = 30
MIN_BOTH_CLASSES = 20

# Fixed p_pre bins requested by the reviewer (narrow bands that also pin who is ahead)
P_BINS: List[Tuple[float, float]] = [
    (0.20, 0.30), (0.30, 0.40), (0.40, 0.45), (0.45, 0.50),
    (0.50, 0.55), (0.55, 0.60), (0.60, 0.70), (0.70, 0.80),
]
BALANCED = {"B40": (0.40, 0.60), "B45": (0.45, 0.55)}


# --------------------------------------------------------------------------- metrics
def match_weights(groups: np.ndarray) -> np.ndarray:
    g = np.asarray(groups)
    _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    return (1.0 / counts[inv]).astype(np.float64)


def _safe_auc(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> Optional[float]:
    if len(np.unique(y)) < 2:
        return None
    if np.nanstd(p) < 1e-12:
        return 0.5
    try:
        return float(roc_auc_score(y, p, sample_weight=w))
    except ValueError:
        return None


def cell_metrics(y: np.ndarray, p: np.ndarray, g: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    g = np.asarray(g)
    n = int(len(y))
    if n == 0:
        return dict(n=0, matches=0, positives=0, positive_rate=None,
                    auc=None, brier=None, logloss=None, reason="empty")
    w = match_weights(g)
    matches = int(len(np.unique(g)))
    out: Dict[str, Any] = dict(
        n=n, matches=matches, positives=int(y.sum()),
        positive_rate=float(np.average(y, weights=w)),
        mean_p=float(np.average(p, weights=w)),
        sparse=matches < MIN_MATCHES,
    )
    out["auc"] = _safe_auc(y, p, w)
    out["brier"] = float(np.average((p - y) ** 2, weights=w))
    out["logloss"] = float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    return out


def paired_bootstrap(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, g: np.ndarray,
    reps: int = BOOT_REPS, seed: int = BOOT_SEED,
) -> Dict[str, Any]:
    """Match-clustered bootstrap of (metric_a - metric_b). Fixed predictions.

    Uses multiplicity weights instead of row concatenation: if match m is drawn
    k times, each of its rows gets weight k / n_rows(m). Equivalent to the
    concatenate-with-replacement scheme, O(n) per replicate.
    """
    y = np.asarray(y).astype(int)
    p_a = np.clip(np.asarray(p_a, dtype=float), 1e-8, 1.0 - 1e-8)
    p_b = np.clip(np.asarray(p_b, dtype=float), 1e-8, 1.0 - 1e-8)
    g = np.asarray(g)
    matches, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    n_m = len(matches)
    base_w = 1.0 / counts[inv]
    rng = np.random.default_rng(seed)
    d_auc = np.empty(reps, dtype=float)
    d_brier = np.empty(reps, dtype=float)
    d_ll = np.empty(reps, dtype=float)
    kept = 0
    skipped = 0
    for _ in range(reps):
        draw = rng.integers(0, n_m, size=n_m)
        mult = np.bincount(draw, minlength=n_m).astype(np.float64)
        w = base_w * mult[inv]
        if w.sum() <= 0 or len(np.unique(y[w > 0])) < 2:
            skipped += 1
            continue
        auc_a, auc_b = _safe_auc(y, p_a, w), _safe_auc(y, p_b, w)
        if auc_a is None or auc_b is None:
            skipped += 1
            continue
        d_auc[kept] = auc_a - auc_b
        d_brier[kept] = float(np.average((p_a - y) ** 2, weights=w)
                              - np.average((p_b - y) ** 2, weights=w))
        d_ll[kept] = float(log_loss(y, p_a, sample_weight=w, labels=[0, 1])
                           - log_loss(y, p_b, sample_weight=w, labels=[0, 1]))
        kept += 1
    def pack(xs: np.ndarray, n: int) -> Dict[str, Any]:
        if n == 0:
            return dict(mean=None, lo=None, hi=None, n=0)
        a = xs[:n]
        return dict(mean=float(a.mean()), lo=float(np.quantile(a, 0.025)),
                    hi=float(np.quantile(a, 0.975)), n=int(n))
    point = cell_metrics(y, p_a, g), cell_metrics(y, p_b, g)
    return dict(
        delta_auc=pack(d_auc, kept), delta_brier=pack(d_brier, kept), delta_logloss=pack(d_ll, kept),
        point_a=point[0], point_b=point[1],
        point_delta_auc=(None if point[0]["auc"] is None or point[1]["auc"] is None
                         else point[0]["auc"] - point[1]["auc"]),
        point_delta_brier=point[0]["brier"] - point[1]["brier"],
        skipped_replicates=skipped, replicates=reps, seed=seed,
        note="match-clustered bootstrap via multiplicity weights; models held fixed; exploratory",
    )


# --------------------------------------------------------------------------- data
def load_test_pack(data_root: Path, cohort: str = "T") -> Dict[str, np.ndarray]:
    path = data_root / "outputs" / IQ / "eval" / "predictions" / f"MAIN_TEST_h90_{cohort}.npz"
    z = np.load(path, allow_pickle=False)
    needed = ["match", "y", "p_pre", "time_minutes",
              "named__lgbm_winner", "named__pt_winner", "named__logit_winner",
              "named__old_p_pre_spline", "named__old_p_pre_logistic", "named__old_constant"]
    out = {k: z[k] for k in needed}
    out["s_ms"] = z["s_ms"]
    for name, (lo, hi) in BALANCED.items():
        out[f"cell__{name}"] = z[f"cell__{name}"] if f"cell__{name}" in z.files else (
            (out["p_pre"] >= lo) & (out["p_pre"] <= hi))
    # attach delta from labels (aligned by match + s)
    lab = np.load(data_root / "outputs" / FC / "labels" / "MAIN_TEST_labels.npz", allow_pickle=False)
    coh = np.load(data_root / "outputs" / CR / "cohorts" / "MAIN_TEST_cohort.npz", allow_pickle=False)
    assert np.array_equal(lab["match"], coh["match"]) and np.array_equal(lab["s"], coh["s"])
    key_lab = np.char.add(lab["match"].astype(str), np.char.add("|", lab["s"].astype(str)))
    key_te = np.char.add(out["match"].astype(str), np.char.add("|", out["s_ms"].astype(str)))
    # map via dict for the cohort slice
    if cohort == "T":
        keep = coh["cohort"] == 1
    else:
        keep = coh["cohort"] == 0
    keep = keep & (lab["valid_h90"] == 1)
    index = {k: i for i, k in enumerate(key_lab[keep])}
    idx = np.array([index[k] for k in key_te], dtype=np.int64)
    lab_keep = {k: lab[k][keep] for k in ("delta_h90", "p_post_h90", "Y_h90", "p_pre")}
    out["delta"] = lab_keep["delta_h90"][idx]
    out["p_post"] = lab_keep["p_post_h90"][idx]
    # consistency checks
    assert np.allclose(out["p_pre"], lab_keep["p_pre"][idx], atol=1e-9)
    assert np.array_equal(out["y"], lab_keep["Y_h90"][idx])
    return out


def load_qcal_t(data_root: Path) -> Dict[str, np.ndarray]:
    """Q_CAL ∩ teamfight ∩ valid_h90 — fit set for isotonic b(p)."""
    lab = np.load(data_root / "outputs" / FC / "labels" / "MAIN_VALIDATION_labels.npz", allow_pickle=False)
    coh = np.load(data_root / "outputs" / CR / "cohorts" / "MAIN_VALIDATION_cohort.npz", allow_pickle=False)
    mask = (lab["sub_role"] == "Q_CAL") & (coh["cohort"] == 1) & (lab["valid_h90"] == 1)
    y = lab["Y_h90"][mask].astype(int)
    # drop draws if any (-1); Y should already be 0/1 on valid rows
    ok = (y == 0) | (y == 1)
    return dict(
        match=lab["match"][mask][ok],
        y=y[ok],
        p_pre=lab["p_pre"][mask][ok].astype(float),
        delta=lab["delta_h90"][mask][ok].astype(float),
        time_minutes=(lab["s"][mask][ok].astype(float) / 60000.0),
        n=int(ok.sum()),
    )


def fit_isotonic_b(p_cal: np.ndarray, y_cal: np.ndarray, g_cal: np.ndarray) -> IsotonicRegression:
    w = match_weights(g_cal)
    iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
    iso.fit(p_cal, y_cal, sample_weight=w)
    return iso


# --------------------------------------------------------------------------- analyses
def score_table(pack: Dict[str, np.ndarray], scores: Dict[str, np.ndarray],
                mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if mask is None:
        mask = np.ones(len(pack["y"]), dtype=bool)
    y, g = pack["y"][mask], pack["match"][mask]
    if y.size < MIN_BOTH_CLASSES or len(np.unique(y)) < 2:
        return dict(n=int(y.size), matches=int(len(np.unique(g))) if y.size else 0,
                    skipped=True, reason="too few rows or single class")
    rows = {}
    for name, p in scores.items():
        rows[name] = cell_metrics(y, p[mask], g)
    return dict(n=int(y.size), matches=int(len(np.unique(g))), skipped=False, models=rows)


def p_sign_curve(p_pre: np.ndarray, y: np.ndarray, g: np.ndarray,
                 edges: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
    p_pre = np.asarray(p_pre, dtype=float)
    y = np.asarray(y).astype(int)
    g = np.asarray(g)
    if edges is None:
        edges = np.linspace(0.05, 0.95, 19)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_pre >= lo) & (p_pre < hi if hi < edges[-1] else p_pre <= hi)
        if not m.any():
            continue
        w = match_weights(g[m])
        out.append(dict(
            lo=float(lo), hi=float(hi), n=int(m.sum()), matches=int(len(np.unique(g[m]))),
            mean_p_pre=float(np.average(p_pre[m], weights=w)),
            P_Y1=float(np.average(y[m], weights=w)),
            mean_delta=None,  # filled by caller if delta present
        ))
    return out


def fill_delta_on_curve(curve: List[Dict[str, Any]], p_pre: np.ndarray, delta: np.ndarray,
                        g: np.ndarray, edges: np.ndarray) -> None:
    for row, (lo, hi) in zip(curve, zip(edges[:-1], edges[1:])):
        m = (p_pre >= lo) & (p_pre < hi if hi < edges[-1] else p_pre <= hi)
        if not m.any():
            continue
        w = match_weights(g[m])
        row["mean_delta"] = float(np.average(delta[m], weights=w))
        row["mean_abs_delta"] = float(np.average(np.abs(delta[m]), weights=w))
        # upside / downside magnitudes when signed
        pos, neg = delta[m] > 0, delta[m] < 0
        if pos.any():
            row["mean_delta_when_pos"] = float(np.average(delta[m][pos], weights=match_weights(g[m][pos])))
        if neg.any():
            row["mean_delta_when_neg"] = float(np.average(delta[m][neg], weights=match_weights(g[m][neg])))


def extract_v_time_calibration(data_root: Path) -> Dict[str, Any]:
    path = data_root / "outputs" / FC / "eval" / "results_v.json"
    rv = json.loads(path.read_text(encoding="utf-8"))
    main = rv["results"]["MAIN_TEST"]["bucket"]["raw"]
    keep = ("auc", "brier", "logloss", "n", "matches", "observed", "predicted",
            "slope", "intercept", "citl_intercept_offset", "ece_10bin")
    overall = {k: main["overall"].get(k) for k in keep}
    bands = {name: {k: cell.get(k) for k in keep} for name, cell in main["time_bands"].items()}
    return dict(
        source=str(path),
        v_chosen=rv.get("v_chosen"),
        semantics=rv.get("semantics"),
        overall=overall,
        time_bands=bands,
        note="V scored against final match win W on minute-bucket states; not ΔV labels.",
    )


# --------------------------------------------------------------------------- report
def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{x:.{nd}f}"


def write_report(out_dir: Path, payload: Dict[str, Any]) -> None:
    lines: List[str] = []
    w = lines.append
    w("# Phase-1 shortcut audit (reviewer response)")
    w("")
    w(f"Generated: {payload['generated']}")
    w(f"Data root: `{payload['data_root']}`")
    w(f"Cohort: MAIN_TEST h90 **{payload['cohort']}** ({payload['n_test']} rows / {payload['n_matches']} matches).")
    w("")
    w("**Nature:** exploratory re-scoring of sealed predictions after prior TEST exposure. "
      "Y = sign(ΔV) from the frozen value model — model-defined, not an independent fight-win truth.")
    w("")
    w("## 1. Overall: does q beat p_pre-only?")
    w("")
    w("| Model | AUC | Brier | Log loss |")
    w("|---|---:|---:|---:|")
    for name, m in payload["overall"]["models"].items():
        w(f"| {name} | {_fmt(m['auc'])} | {_fmt(m['brier'])} | {_fmt(m['logloss'])} |")
    w("")
    for key, title in payload["contrasts"].items():
        c = payload["bootstrap"][key]
        w(f"- **{title}**: ΔAUC {_fmt(c['point_delta_auc'])} "
          f"[{_fmt(c['delta_auc']['lo'])}, {_fmt(c['delta_auc']['hi'])}]; "
          f"ΔBrier {_fmt(c['point_delta_brier'])} "
          f"[{_fmt(c['delta_brier']['lo'])}, {_fmt(c['delta_brier']['hi'])}] "
          f"(match bootstrap {c['replicates']}, models fixed).")
    w("")
    w("Reading: `p_pre_raw` uses the frozen win probability as a ranking score for Y. "
      "`b_isotonic(p_pre)` is P(Y=1|p_pre) fitted on Q_CAL∩T only. "
      "`old_p_pre_spline` / `PT` / `lgbm` are sealed named predictors from `incremental_q_training_20260915`.")
    w("")
    w("## 2. Balanced cells (predeclared B40 / B45)")
    w("")
    for cell in ("B40", "B45"):
        block = payload["cells"][cell]
        if block.get("skipped"):
            w(f"### {cell}: skipped ({block.get('reason')})")
            continue
        w(f"### {cell} ({block['n']} rows / {block['matches']} matches)")
        w("")
        w("| Model | AUC | Brier |")
        w("|---|---:|---:|")
        for name, m in block["models"].items():
            w(f"| {name} | {_fmt(m['auc'])} | {_fmt(m['brier'])} |")
        w("")
    w("## 3. Narrow p_pre bins (who is ahead is pinned)")
    w("")
    w("Inside each bin the pure p_pre ranking has almost no score variation, so AUC near 0.5 for "
      "`p_pre_raw` is expected; any lift of `lgbm` over `b_isotonic` is information beyond the scalar p_pre.")
    w("")
    w("| Bin | n | matches | P(Y=1) | p_pre AUC | b(p) AUC | lgbm AUC | lgbm−b(p) AUC |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in payload["p_bins"]:
        if row.get("skipped"):
            w(f"| [{row['lo']:.2f},{row['hi']:.2f}) | {row['n']} | {row.get('matches', 0)} | — | — | — | — | skipped |")
            continue
        m = row["models"]
        d = None
        if m["lgbm_winner"]["auc"] is not None and m["b_isotonic"]["auc"] is not None:
            d = m["lgbm_winner"]["auc"] - m["b_isotonic"]["auc"]
        w(f"| [{row['lo']:.2f},{row['hi']:.2f}) | {row['n']} | {row['matches']} | "
          f"{_fmt(m['lgbm_winner']['positive_rate'], 3)} | "
          f"{_fmt(m['p_pre_raw']['auc'])} | {_fmt(m['b_isotonic']['auc'])} | "
          f"{_fmt(m['lgbm_winner']['auc'])} | {_fmt(d)} |")
    w("")
    w("## 4. P(Y=1 | p_pre) curve (asymmetric-range hypothesis)")
    w("")
    w("If the bounded-probability story holds, P(Y=1) should rise with p_pre even when mean ΔV ≈ 0.")
    w("")
    w("| p_pre bin | n | P(Y=1) | mean ΔV | mean ΔV\\|pos | mean ΔV\\|neg |")
    w("|---|---:|---:|---:|---:|---:|")
    for row in payload["sign_curve"]:
        w(f"| [{row['lo']:.2f},{row['hi']:.2f}) | {row['n']} | {_fmt(row['P_Y1'], 3)} | "
          f"{_fmt(row.get('mean_delta'), 4)} | {_fmt(row.get('mean_delta_when_pos'), 4)} | "
          f"{_fmt(row.get('mean_delta_when_neg'), 4)} |")
    w("")
    w("## 5. V calibration by game time (against match win W)")
    w("")
    w("Reused from `full_corpus_training_20260915/eval/results_v.json` (MAIN_TEST bucket, raw).")
    w("")
    v = payload["v_time_calibration"]
    w(f"Overall AUC {_fmt(v['overall']['auc'])}, ECE {_fmt(v['overall']['ece_10bin'])}, "
      f"slope {_fmt(v['overall']['slope'])}.")
    w("")
    w("| Time band | n | AUC | Brier | ECE | slope | CITL intercept |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for name, cell in v["time_bands"].items():
        w(f"| {name} | {cell['n']} | {_fmt(cell['auc'])} | {_fmt(cell['brier'])} | "
          f"{_fmt(cell['ece_10bin'])} | {_fmt(cell['slope'])} | {_fmt(cell['citl_intercept_offset'])} |")
    w("")
    w("Early game (2–10 min) discrimination is much weaker than late game; late bands inflate the pooled V AUC.")
    w("")
    w("## 6. Takeaway for the reply")
    w("")
    for line in payload["takeaways"]:
        w(f"- {line}")
    w("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cohort", choices=("T", "N"), default="T")
    ap.add_argument("--boot-reps", type=int, default=BOOT_REPS)
    ap.add_argument("--boot-seed", type=int, default=BOOT_SEED)
    args = ap.parse_args(argv)

    data_root: Path = args.data_root
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cal = load_qcal_t(data_root)
    iso = fit_isotonic_b(cal["p_pre"], cal["y"], cal["match"])

    pack = load_test_pack(data_root, cohort=args.cohort)
    b_p = iso.predict(pack["p_pre"])

    scores = {
        "p_pre_raw": pack["p_pre"],
        "b_isotonic": b_p,
        "old_p_pre_logistic": pack["named__old_p_pre_logistic"],
        "old_p_pre_spline": pack["named__old_p_pre_spline"],
        "pt_winner": pack["named__pt_winner"],
        "logit_winner": pack["named__logit_winner"],
        "lgbm_winner": pack["named__lgbm_winner"],
        "constant": pack["named__old_constant"],
    }

    overall = score_table(pack, scores)

    contrasts = {
        "lgbm_minus_b_isotonic": "full LightGBM − isotonic b(p_pre)",
        "lgbm_minus_p_pre_raw": "full LightGBM − raw p_pre",
        "lgbm_minus_pt": "full LightGBM − PT (sealed primary)",
        "pt_minus_b_isotonic": "PT − isotonic b(p_pre)",
        "b_isotonic_minus_p_pre_raw": "isotonic b(p_pre) − raw p_pre",
    }
    boot = {
        "lgbm_minus_b_isotonic": paired_bootstrap(
            pack["y"], scores["lgbm_winner"], scores["b_isotonic"], pack["match"],
            reps=args.boot_reps, seed=args.boot_seed),
        "lgbm_minus_p_pre_raw": paired_bootstrap(
            pack["y"], scores["lgbm_winner"], scores["p_pre_raw"], pack["match"],
            reps=args.boot_reps, seed=args.boot_seed + 1),
        "lgbm_minus_pt": paired_bootstrap(
            pack["y"], scores["lgbm_winner"], scores["pt_winner"], pack["match"],
            reps=args.boot_reps, seed=args.boot_seed + 2),
        "pt_minus_b_isotonic": paired_bootstrap(
            pack["y"], scores["pt_winner"], scores["b_isotonic"], pack["match"],
            reps=args.boot_reps, seed=args.boot_seed + 3),
        "b_isotonic_minus_p_pre_raw": paired_bootstrap(
            pack["y"], scores["b_isotonic"], scores["p_pre_raw"], pack["match"],
            reps=args.boot_reps, seed=args.boot_seed + 4),
    }

    cells = {}
    for name in BALANCED:
        cells[name] = score_table(pack, scores, mask=pack[f"cell__{name}"].astype(bool))
        # bootstrap primary contrast inside cell
        m = pack[f"cell__{name}"].astype(bool)
        if m.sum() >= MIN_BOTH_CLASSES and len(np.unique(pack["y"][m])) > 1:
            cells[name]["bootstrap_lgbm_minus_b"] = paired_bootstrap(
                pack["y"][m], scores["lgbm_winner"][m], scores["b_isotonic"][m],
                pack["match"][m], reps=args.boot_reps, seed=args.boot_seed + 10)

    p_bins = []
    for lo, hi in P_BINS:
        m = (pack["p_pre"] >= lo) & (pack["p_pre"] < hi)
        block = score_table(pack, {
            "p_pre_raw": scores["p_pre_raw"],
            "b_isotonic": scores["b_isotonic"],
            "lgbm_winner": scores["lgbm_winner"],
            "pt_winner": scores["pt_winner"],
        }, mask=m)
        block["lo"], block["hi"] = lo, hi
        p_bins.append(block)

    edges = np.linspace(0.05, 0.95, 19)
    curve = p_sign_curve(pack["p_pre"], pack["y"], pack["match"], edges=edges)
    fill_delta_on_curve(curve, pack["p_pre"], pack["delta"], pack["match"], edges)

    v_cal = extract_v_time_calibration(data_root)

    # concise takeaways from numbers
    ov = overall["models"]
    d_lb = boot["lgbm_minus_b_isotonic"]
    takeaways = [
        (f"On all T, raw p_pre already reaches AUC {_fmt(ov['p_pre_raw']['auc'])}; "
         f"isotonic b(p) {_fmt(ov['b_isotonic']['auc'])}; PT {_fmt(ov['pt_winner']['auc'])}; "
         f"LightGBM {_fmt(ov['lgbm_winner']['auc'])}."),
        (f"LightGBM − b(p) ΔAUC {_fmt(d_lb['point_delta_auc'])} "
         f"[{_fmt(d_lb['delta_auc']['lo'])}, {_fmt(d_lb['delta_auc']['hi'])}] — "
         "small but the interval excludes 0" if (
             d_lb["delta_auc"]["lo"] is not None and d_lb["delta_auc"]["lo"] > 0
         ) else
         f"LightGBM − b(p) ΔAUC {_fmt(d_lb['point_delta_auc'])} "
         f"[{_fmt(d_lb['delta_auc']['lo'])}, {_fmt(d_lb['delta_auc']['hi'])}]."),
        "Narrow p_pre bins are the right place to claim fight-specific signal: "
        "compare lgbm to b(p) inside each bin, not overall AUC alone.",
        (f"V overall AUC {_fmt(v_cal['overall']['auc'])} vs 2–10 min "
         f"{_fmt(v_cal['time_bands']['2-10']['auc'])}: late-game states inflate pooled V discrimination."),
        "Kill-less reference windows are deferred to a design discussion (phase 2/3); not run here.",
    ]

    from datetime import datetime, timezone
    payload = dict(
        generated=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        data_root=str(data_root),
        cohort=args.cohort,
        n_test=int(len(pack["y"])),
        n_matches=int(len(np.unique(pack["match"]))),
        qcal_fit=dict(n=cal["n"], n_matches=int(len(np.unique(cal["match"]))),
                      method="IsotonicRegression match-weighted on Q_CAL∩T∩valid_h90"),
        overall=overall,
        contrasts=contrasts,
        bootstrap=boot,
        cells=cells,
        p_bins=p_bins,
        sign_curve=curve,
        v_time_calibration=v_cal,
        takeaways=takeaways,
        protocol=dict(
            label="Y_h90 = sign(delta_h90) from frozen V",
            sealed_predictions="incremental_q_training_20260915/eval/predictions/MAIN_TEST_h90_T.npz",
            exploratory=True,
        ),
    )

    # JSON-friendly
    def convert(o: Any) -> Any:
        if isinstance(o, dict):
            return {str(k): convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        if isinstance(o, (np.floating, float)):
            x = float(o)
            return None if math.isnan(x) or math.isinf(x) else x
        if isinstance(o, (np.integer, int)):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if o is None:
            return None
        return o

    (out_dir / "results.json").write_text(
        json.dumps(convert(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir, payload)
    print(json.dumps({
        "out_dir": str(out_dir),
        "n": payload["n_test"],
        "auc_p_pre": ov["p_pre_raw"]["auc"],
        "auc_b": ov["b_isotonic"]["auc"],
        "auc_pt": ov["pt_winner"]["auc"],
        "auc_lgbm": ov["lgbm_winner"]["auc"],
        "delta_lgbm_minus_b": d_lb["point_delta_auc"],
        "ci": [d_lb["delta_auc"]["lo"], d_lb["delta_auc"]["hi"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
