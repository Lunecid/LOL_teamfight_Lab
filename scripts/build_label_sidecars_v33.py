"""Label-family sidecars for corpus v3.3: every label variant on the corpus rows, no feature matrix.

Reviewer R2 (CoG 2026 submission 118, major) asked for alternative labelling schemes and a
sensitivity analysis of the label's parameters.  The corpus shards store labels as int8 only, and
compute_label needs ref fields the shards do not keep (label_end_ts, first/last cluster kill,
anchor), so the engagement refs are rebuilt exactly as the v3.3 build rebuilt them and every
variant of the registry in gameplay/labels.py (LABEL_VARIANTS) is computed on the same rows in
one pass.  Reference followed: the v3.3 build itself --

    partition  scripts/build_corpus_shard.py shard_match_ids (seed, num_shards, n_matches from
               corpus_shards_v33/manifest.json)
    refs       data/index_split.build_fight_index under LOL_CFG_PRESET=v3.3 plus the manifest's
               effective LOL_CFG_OVERRIDES, index cache and fight dumps off (build_corpus_shard.py)
    rows       train/baseline.build_tabular_Xy keeps a ref when its match cache loads and
               gameplay/pipeline.build_ms_sequence returns a sample.  Its guards (context and label
               window inside the match, row label not None under market_event / tie 'random', via
               the same compute_label_targets call) are re-run here without building features.
    labels     scripts/build_corpus_shard.py label_ref -- the call behind every stored y_<type>
               column -- under the extra-label tie policy 'drop' (-1 = draw) and the variant's cfg
               overrides (gameplay/labels.py variant_cfg).

Every shard is verified row for row against corpus_shards_v33/shard_XXX.npz: groups, engage_ts,
patch, the four participation counts, the row label y and every stored label column the registry
reproduces (y_market_event, y_market_event@window, y_market_lex, y_market_lex@window,
y_attention_value_win) must be equal.  A mismatch aborts that shard (exit code 3, no npz written)
and the whole build.

    LOL_OUTPUT_ROOT=D:/LOL_Project .venv/Scripts/python.exe scripts/build_label_sidecars_v33.py \
        --out-dir D:/LOL_Project/fusion_2615/features/tog_revision/A2-label-family/label_sidecars_v33 \
        --parallel 8 --index-workers 2

Smoke: ``--shards 0,31 --limit-matches-per-shard 60 --parallel 2 --index-workers 1`` keeps the first
60 matches of each shard's own match order, so the sidecar is a prefix of the corpus shard and the
identity check stays exact.

Writes, per shard, labels_shard_XXX.npz (y_<variant> int8 for every variant, y_row, groups,
engage_ts, patch, cluster_blue/red, present_blue/red, corpus_row), labels_shard_XXX.json (identity
report, label shares, timing) and labels_shard_XXX.log; manifest.json records git commit, preset,
effective overrides, corpus run id, the variant registry, per-shard status, deviations, wall clock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CORPUS = Path("D:/LOL_Project/fusion_2615/corpus_shards_v33")
V33_DETECTOR = {
    "TF2_KILL_CLUSTER_GAP_MS": 13700,
    "CLUSTER_MAX_DIAMETER": 4264.0,
    "TF2_VALIDITY_RADIUS": 1600.0,
    "TF2_ENGAGE_PRE_KILL_MS": 15000,
    "FIGHT_HORIZON_SEC": 35,
}
V33_LABEL = {"row_label": "market_event", "row_tie_policy": "random", "extra_tie_policy": "drop", "gold_deadzone": 300.0}
V33_FEATURE_NAMES_SHA1 = "3c97f60383d303aa9d53e107ecb350cc0e1b4886"
# cfg values every variant starts from; a variant only changes what its overrides name
V33_LABEL_BASE = {
    "LABEL_EVENT_ATTRIBUTION": "engagement",
    "LABEL_GOLD_DEADZONE": 300.0,
    "LABEL_EVENT_PRICE_TABLE": "config/game_rules/event_prices.json",
    "LABEL_ATTRIBUTION_RADIUS_U": 0.0,
    "LABEL_EVENT_PRICE_NONKILL_SCALE": 1.0,
    "LABEL_TIE_SEED": 7,
}
ROW_KEYS = ("groups", "engage_ts", "patch", "cluster_blue", "cluster_red", "present_blue", "present_red")
STRING_KEYS = ("groups", "patch")
IDENTITY_EXIT = 3

DEVIATIONS = [
    "Rows are selected without building features: build_ms_sequence's guards (context and label window inside the "
    "match, row label not None under market_event / tie 'random', via the same compute_label_targets call) are re-run, "
    "but build_tabular_Xy's feature-sequence guards (sequence key found, sequence not None) are not evaluated; every "
    "shard is therefore verified row for row against its corpus shard instead of being trusted.",
    "build_fight_index may run with a different worker count than the v3.3 build (--index-workers); "
    "ProcessPoolExecutor.map keeps task order, so refs are unaffected, and the identity check confirms it.",
    "--limit-matches-per-shard (smoke only) keeps the first N matches of the shard's own match order; the sidecar is "
    "then checked against the corpus prefix and no later corpus row may belong to an included match.",
]


class SidecarIdentityError(RuntimeError):
    """A sidecar shard is not row-for-row identical to its corpus shard."""


def _sha1(path: Path):
    try:
        return hashlib.sha1(open(path, "rb").read()).hexdigest()
    except OSError:
        return None


def _git_state() -> tuple[str, bool | None]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT), capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=str(PROJECT_ROOT),
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except Exception:
        return "", None


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_script_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_shards(spec: str, num_shards: int) -> list[int]:
    """'all', or a comma list of shard ids and ranges ('0,5-7,31')."""
    if str(spec).strip().lower() == "all":
        return list(range(num_shards))
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < num_shards]
    if bad:
        raise SystemExit(f"shard ids out of range 0..{num_shards - 1}: {bad}")
    return sorted(set(out))


def _pooled_prices_at(commit: str):
    """The pooled price table of config/game_rules/event_prices.json at a git commit (None if unavailable)."""
    try:
        out = subprocess.run(["git", "show", f"{commit}:config/game_rules/event_prices.json"], cwd=str(PROJECT_ROOT),
                             capture_output=True, text=True, encoding="utf-8")
        if out.returncode != 0:
            return None
        return json.loads(out.stdout).get("pooled")
    except Exception:
        return None


def check_price_table(manifest: dict) -> dict:
    """Are the prices compute_label reads today the prices the v3.3 build read?

    The labels read only the table's ``pooled`` entries.  When the file's bytes still hash to the build's
    ``price_table_sha1`` the answer is yes; otherwise the pooled entries must equal those of the file at the
    build commit (text-only edits such as the unit / source / note fields are allowed).  The row-for-row
    y_market_event identity check is the final word either way.
    """
    path = PROJECT_ROOT / "config/game_rules/event_prices.json"
    built = (manifest.get("label") or {}).get("price_table_sha1")
    now = _sha1(path)
    report = {"sha1_build": built, "sha1_now": now, "same_file": bool(built and now == built)}
    if report["same_file"]:
        report["pooled_equal"] = True
        return report
    commit = manifest.get("git_commit") or ""
    try:
        now_pooled = {str(k): float(v) for k, v in (json.load(open(path, encoding="utf-8")).get("pooled") or {}).items()}
    except (OSError, ValueError):
        now_pooled = None
    old = _pooled_prices_at(commit) if commit else None
    old_pooled = {str(k): float(v) for k, v in old.items()} if isinstance(old, dict) else None
    report.update(build_commit=commit, pooled_equal=bool(now_pooled is not None and old_pooled is not None and now_pooled == old_pooled),
                  compared_with=f"git show {commit}:config/game_rules/event_prices.json (pooled)")
    return report


def load_corpus_manifest(corpus_dir: Path) -> dict:
    """The v3.3 build's settings; refuses any corpus that is not the complete v3.3 build."""
    mp = Path(corpus_dir) / "manifest.json"
    if not mp.exists():
        raise SystemExit(f"{corpus_dir} has no manifest.json")
    m = json.load(open(mp, encoding="utf-8"))
    problems = []
    if not m.get("complete", False):
        problems.append("manifest marks the build incomplete")
    eff = m.get("effective_overrides") or {}
    for k, v in V33_DETECTOR.items():
        if k not in eff or float(eff[k]) != float(v):
            problems.append(f"{k}={eff.get(k)!r} (v3.3: {v!r})")
    lab = m.get("label") or {}
    for k, v in V33_LABEL.items():
        if lab.get(k) != v:
            problems.append(f"label.{k}={lab.get(k)!r} (v3.3: {v!r})")
    if m.get("feature_names_sha1") != V33_FEATURE_NAMES_SHA1:
        problems.append(f"feature_names_sha1={m.get('feature_names_sha1')!r}")
    if _sha1(Path(corpus_dir) / "feature_names.json") != V33_FEATURE_NAMES_SHA1:
        problems.append("feature_names.json hash differs from the v3.3 corpus")
    prices = check_price_table(m)
    if not prices["pooled_equal"]:
        problems.append(f"config/game_rules/event_prices.json prices differ from the build's: {prices}")
    now_anchors = _sha1(PROJECT_ROOT / "config/game_rules/map_anchors.json")
    if lab.get("map_anchors_sha1") != now_anchors:
        problems.append(f"config/game_rules/map_anchors.json changed since the build ({now_anchors} vs {lab.get('map_anchors_sha1')})")
    if problems:
        raise SystemExit("corpus is not the v3.3 build this script reproduces: " + "; ".join(problems))
    return m


def verify_identity(side: dict, corpus: dict, *, prefix: bool, included_matches=None, stored_variants=()) -> dict:
    """Row-for-row identity of one sidecar shard against its corpus shard; raises on any mismatch.

    ``side`` holds the sidecar columns, ``corpus`` the corpus shard's non-X arrays.  Without a match
    limit the row counts must be equal.  With one, the sidecar must equal the corpus prefix and no
    corpus row after the prefix may belong to an included match (the prefix is complete).
    ``stored_variants`` names the variants whose y_<name> column the corpus stores; market_event is
    required among them.
    """
    if "market_event" not in set(stored_variants):
        raise SidecarIdentityError("the identity check needs y_market_event computed through the registry")
    n = int(len(side["groups"]))
    n_corpus = int(len(corpus["groups"]))
    if prefix:
        if n > n_corpus:
            raise SidecarIdentityError(f"sidecar has {n} rows, corpus shard only {n_corpus}")
        if included_matches is not None:
            late = set(np.unique(np.asarray(corpus["groups"])[n:]).astype(str).tolist()) & {str(x) for x in included_matches}
            if late:
                raise SidecarIdentityError(f"corpus rows after the {n}-row prefix belong to included matches: {sorted(late)[:5]}")
    elif n != n_corpus:
        raise SidecarIdentityError(f"row count {n} != corpus shard {n_corpus}")
    pairs = [(k, k) for k in ROW_KEYS] + [("y_row", "y")] + [(f"y_{v}", f"y_{v}") for v in stored_variants]
    checked = {}
    for sk, ck in pairs:
        if sk not in side or ck not in corpus:
            raise SidecarIdentityError(f"missing column: sidecar {sk!r} in={sk in side}, corpus {ck!r} in={ck in corpus}")
        a = np.asarray(side[sk])
        b = np.asarray(corpus[ck])[:n]
        if sk in STRING_KEYS:
            a, b = a.astype(str), b.astype(str)
        if a.shape != b.shape:
            raise SidecarIdentityError(f"{sk} shape {a.shape} != corpus {ck} {b.shape}")
        bad = np.flatnonzero(a != b)
        if len(bad):
            i = int(bad[0])
            raise SidecarIdentityError(f"{sk} vs corpus {ck}: {len(bad)} of {n} rows differ; first at row {i}: "
                                       f"sidecar={a[i]!r} corpus={b[i]!r} (groups={np.asarray(corpus['groups'])[i]!r})")
        checked[sk] = ck
    return {"ok": True, "n_rows": n, "n_corpus_rows": n_corpus, "prefix": bool(prefix), "checked": checked}


def row_label_for_ref(pack, r, *, ctx_ms: int, bin_ms: int, horizon_ms: int, prediction_gap_ms: int,
                      compute_label_targets, ref_fields) -> int | None:
    """The row label build_tabular_Xy stores for ``r``, or None where build_ms_sequence returns None.

    Mirrors gameplay/pipeline.build_ms_sequence as build_tabular_Xy calls it (t_start = -1, the ref's
    engage / label-end / kill / anchor fields): the context window must start inside the match, the
    label window must end inside it, and compute_label_targets must return a label under the row
    label settings cfg holds.  Features are not built.
    """
    engage, label_end, first_kill, last_kill, anchor = ref_fields
    engage_ts = engage(r)
    if engage_ts is None or engage_ts < 0:
        return None          # t_start = -1 with no engage_ts: build_ms_sequence returns None
    t_min = int(pack["minute_ts"][0])
    t_max = int(pack["minute_ts"][-1])
    label_start_ms = int(engage_ts)
    label_end_ms = label_start_ms + int(horizon_ms)
    cand = label_end(r)
    if cand is not None and int(cand) > label_start_ms:
        label_end_ms = int(cand)
    end_ms = int(label_start_ms - max(0, int(prediction_gap_ms)))
    start_ms = end_ms - int(ctx_ms)
    if end_ms < t_min or start_ms < t_min or label_end_ms > t_max:
        return None
    if int(ctx_ms // bin_ms) <= 0:
        return None
    y_pack = compute_label_targets(
        pack, pack["meta"]["team_map"], -1,
        engage_ts=label_start_ms, label_end_ts=label_end_ms, horizon_ms=horizon_ms,
        first_kill_ts=first_kill(r), last_kill_ts=last_kill(r), anchor_xy=anchor(r),
    )
    return None if y_pack is None else int(y_pack["y"])


def _refs_by_match(refs) -> list[tuple[str, list]]:
    out: list[tuple[str, list]] = []
    for r in refs:
        if out and out[-1][0] == r.match_id:
            out[-1][1].append(r)
        else:
            out.append((r.match_id, [r]))
    return out


def run_worker(args) -> int:
    t_start = time.time()
    from core.config import CACHE_DIR, cfg
    from core.timeutils import _get_bin_ms, _get_context_ms, _get_horizon_ms
    from data.cache_io import load_match_cache
    from data.index_split import build_fight_index
    from gameplay.labels import cfg_override, get_label_variant, variant_cfg
    from gameplay.pipeline import compute_label_targets
    from train.baseline import _ref_anchor_xy, _ref_engage_ts, _ref_first_kill_ts, _ref_label_end_ts, _ref_last_kill_ts
    bcs = _load_script("build_corpus_shard")

    m = load_corpus_manifest(args.corpus)
    wrong = {k: getattr(cfg, k, None) for k, v in V33_DETECTOR.items() if float(getattr(cfg, k, -1)) != float(v)}
    wrong.update({k: getattr(cfg, k, None) for k, v in V33_LABEL_BASE.items() if getattr(cfg, k, None) != v})
    if getattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", None):
        wrong["LABEL_EVENT_PRICE_OVERRIDES"] = cfg.LABEL_EVENT_PRICE_OVERRIDES
    if wrong:
        raise SystemExit(f"[shard {args.shard}] cfg is not the v3.3 base state (run with LOL_CFG_PRESET=v3.3): {wrong}")
    cfg.FIGHT_INDEX_CACHE_ENABLED = False
    cfg.DUMP_FIGHTS = False
    if args.index_workers > 0:
        cfg.FIGHT_INDEX_NUM_WORKERS = int(args.index_workers)

    variants = [get_label_variant(n) for n in args.variants.split(",") if n.strip()]
    corpus_path = Path(args.corpus) / f"shard_{args.shard:03d}.npz"
    with np.load(corpus_path, allow_pickle=False) as blob:
        corpus = {k: blob[k] for k in blob.files if k != "X"}
    stored = [v.name for v in variants if f"y_{v.name}" in corpus]
    missing_stored = [v.name for v in variants if v.stored and f"y_{v.name}" not in corpus]
    if missing_stored:
        raise SystemExit(f"[shard {args.shard}] registry marks {missing_stored} as stored but {corpus_path} lacks them")

    full_mine = bcs.shard_match_ids(bcs.cached_match_ids(CACHE_DIR), args.shard, int(m["num_shards"]),
                                    n_matches=m.get("n_matches"), seed=int(m["seed"]))
    pos = {mid: i for i, mid in enumerate(full_mine)}
    corpus_pos = np.array([pos.get(str(g), -1) for g in corpus["groups"]], dtype=np.int64)
    out_dir = Path(args.out_dir)
    stem = f"labels_shard_{args.shard:03d}"
    if (corpus_pos < 0).any() or (np.diff(corpus_pos) < 0).any():
        msg = (f"[shard {args.shard}] partition mismatch: {int((corpus_pos < 0).sum())} corpus rows belong to matches outside "
               f"this shard's partition of {CACHE_DIR} or appear out of shard order")
        json.dump({"shard": args.shard, "error": msg}, open(out_dir / f"{stem}.mismatch.json", "w", encoding="utf-8"), indent=1)
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
        return IDENTITY_EXIT
    limit = int(args.limit_matches_per_shard or 0)
    mine = full_mine[:limit] if limit > 0 else full_mine
    print(f"[shard {args.shard}] matches={len(mine)} of {len(full_mine)} variants={len(variants)} stored={stored}", flush=True)

    row_label = str(m["label"]["row_label"])
    row_tie = str(m["label"]["row_tie_policy"])
    extra_tie = str(m["label"]["extra_tie_policy"])

    t0 = time.time()
    # build_corpus_shard.py set --label-type / --tie-policy on cfg before indexing; mirror that process state
    with cfg_override({"LABEL_TYPE": row_label, "LABEL_TIE_POLICY": row_tie}):
        refs = build_fight_index(cache_match_ids=mine)
    t_index = time.time() - t0
    print(f"[shard {args.shard}] refs={len(refs)} ({t_index:.0f}s)", flush=True)

    ref_fields = (_ref_engage_ts, _ref_label_end_ts, _ref_first_kill_ts, _ref_last_kill_ts, _ref_anchor_xy)
    ctx_ms, bin_ms, horizon_ms = _get_context_ms(), _get_bin_ms(), _get_horizon_ms()
    gap_ms = int(getattr(cfg, "PREDICTION_GAP_MS", 0))

    t0 = time.time()
    t_rows = 0.0
    t_variant = {v.name: 0.0 for v in variants}
    used, y_row = [], []
    cols: dict[str, list[int]] = {v.name: [] for v in variants}
    for mid, rs in _refs_by_match(refs):
        pack = load_match_cache(mid)
        if not pack:
            continue
        keep = []
        t1 = time.perf_counter()
        with cfg_override({"LABEL_TYPE": row_label, "LABEL_TIE_POLICY": row_tie}):
            for r in rs:
                y = row_label_for_ref(pack, r, ctx_ms=ctx_ms, bin_ms=bin_ms, horizon_ms=horizon_ms,
                                      prediction_gap_ms=gap_ms, compute_label_targets=compute_label_targets,
                                      ref_fields=ref_fields)
                if y is not None:
                    keep.append((r, y))
        t_rows += time.perf_counter() - t1
        if not keep:
            continue
        tm = bcs.team_map_int(pack)
        with cfg_override({"LABEL_TIE_POLICY": extra_tie}):
            for v in variants:
                t1 = time.perf_counter()
                with variant_cfg(v):
                    cols[v.name].extend(bcs.label_ref(pack, tm, r) for r, _ in keep)
                t_variant[v.name] += time.perf_counter() - t1
        used.extend(r for r, _ in keep)
        y_row.extend(y for _, y in keep)
    t_labels = time.time() - t0

    side = {
        "groups": np.array([r.match_id for r in used], dtype=str),
        "engage_ts": np.array([r.t_start_ts for r in used], dtype=np.int64),
        "patch": np.array([r.patch for r in used], dtype=str),
        "cluster_blue": np.array([r.det_cluster_blue for r in used], dtype=np.int16),
        "cluster_red": np.array([r.det_cluster_red for r in used], dtype=np.int16),
        "present_blue": np.array([r.det_present_blue for r in used], dtype=np.int16),
        "present_red": np.array([r.det_present_red for r in used], dtype=np.int16),
        "y_row": np.array(y_row, dtype=np.int8),
    }
    for v in variants:
        side[f"y_{v.name}"] = np.array(cols[v.name], dtype=np.int8)
    try:
        report = verify_identity(side, corpus, prefix=limit > 0, included_matches=(mine if limit > 0 else None),
                                 stored_variants=stored)
    except SidecarIdentityError as e:
        msg = f"[shard {args.shard}] IDENTITY MISMATCH against {corpus_path}: {e}"
        json.dump({"shard": args.shard, "error": msg, "n_rows": len(used)},
                  open(out_dir / f"{stem}.mismatch.json", "w", encoding="utf-8"), indent=1)
        print("!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
        return IDENTITY_EXIT
    print(f"[shard {args.shard}] identity ok: {report['n_rows']} rows, checked {sorted(report['checked'])}", flush=True)

    ref = side["y_market_event"]
    labels = {}
    for v in variants:
        col = side[f"y_{v.name}"]
        ok = col >= 0
        both = ok & (ref >= 0)
        labels[v.name] = {
            "labelled_share": float(ok.mean()) if len(col) else float("nan"),
            "positive_rate": float((col[ok] == 1).mean()) if ok.any() else float("nan"),
            "agreement_with_market_event": float((col[both] == ref[both]).mean()) if both.any() else float("nan"),
        }
        print(f"[shard {args.shard}] y_{v.name}: labelled {labels[v.name]['labelled_share'] * 100:.1f}% "
              f"positives {labels[v.name]['positive_rate']:.3f} agreement with market_event "
              f"{labels[v.name]['agreement_with_market_event']:.3f}", flush=True)

    tmp = out_dir / f"{stem}.partial.npz"
    np.savez_compressed(tmp, corpus_row=np.arange(len(used), dtype=np.int32), **side)
    os.replace(tmp, out_dir / f"{stem}.npz")
    git_commit, git_dirty = _git_state()
    info = {
        "shard": args.shard, "corpus_shard": str(corpus_path), "cache_dir": str(CACHE_DIR),
        "git_commit": git_commit, "git_dirty": git_dirty, "preset": os.environ.get("LOL_CFG_PRESET"),
        "n_matches_shard": len(full_mine), "n_matches_used": len(mine), "limit_matches_per_shard": limit or None,
        "n_refs": len(refs), "n_rows": len(used), "n_matches_with_rows": int(len(np.unique(side["groups"]))),
        "variants": [v.name for v in variants], "stored_variants_checked": stored,
        "identity": report, "labels": labels, "price_table": check_price_table(m),
        "timing_s": {"fight_index": round(t_index, 1), "rows_and_labels": round(t_labels, 1), "row_guards": round(t_rows, 1),
                     "per_variant": {k: round(v, 2) for k, v in t_variant.items()}, "total": round(time.time() - t_start, 1)},
    }
    json.dump(info, open(out_dir / f"{stem}.json", "w", encoding="utf-8"), indent=1)
    print(f"[shard {args.shard}] wrote {out_dir / (stem + '.npz')} in {time.time() - t_start:.0f}s", flush=True)
    return 0


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "".join(open(path, encoding="utf-8", errors="replace").readlines()[-n:])
    except OSError:
        return ""


def run_parent(args) -> int:
    t_run = time.time()
    m = load_corpus_manifest(args.corpus)
    num_shards = int(m["num_shards"])
    shards = parse_shards(args.shards, num_shards)
    out_dir = Path(args.out_dir)
    if out_dir.resolve() == Path(args.corpus).resolve() or "match_cache" in str(out_dir).lower():
        raise SystemExit(f"refusing to write into {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    from gameplay.labels import LABEL_VARIANTS, get_label_variant
    names = list(LABEL_VARIANTS) if args.variants == "all" else [s.strip() for s in args.variants.split(",") if s.strip()]
    for n in names:
        get_label_variant(n)
    if "market_event" not in names:
        names = ["market_event"] + names
    limit = int(args.limit_matches_per_shard or 0)

    todo = []
    for i in shards:
        stem = f"labels_shard_{i:03d}"
        existing = [out_dir / f"{stem}{s}" for s in (".npz", ".json", ".mismatch.json")]
        if any(p.exists() for p in existing):
            if args.skip_existing and (out_dir / f"{stem}.npz").exists() and (out_dir / f"{stem}.json").exists():
                info = json.load(open(out_dir / f"{stem}.json", encoding="utf-8"))
                if info.get("identity", {}).get("ok") and info.get("variants") == names and (info.get("limit_matches_per_shard") or 0) == limit:
                    print(f"[shard {i}] already built with the same settings; skipping", flush=True)
                    continue
            if not args.overwrite:
                raise SystemExit(f"{out_dir} already holds outputs for shard {i}; pass --overwrite or --skip-existing")
            for p in existing:
                if p.exists():
                    p.unlink()
        todo.append(i)

    env = dict(os.environ)
    env["LOL_CFG_PRESET"] = "v3.3"
    inherited = {}
    if env.get("LOL_CFG_OVERRIDES"):
        try:
            inherited = json.loads(env["LOL_CFG_OVERRIDES"])
        except json.JSONDecodeError as e:
            raise SystemExit(f"inherited LOL_CFG_OVERRIDES is not JSON: {e}")
    effective_build = m.get("effective_overrides") or {}
    conflicts = {k: (inherited[k], effective_build[k]) for k in inherited if k in effective_build and inherited[k] != effective_build[k]}
    extra_keys = sorted(set(inherited) - set(effective_build))
    if conflicts or extra_keys:
        raise SystemExit(f"LOL_CFG_OVERRIDES differs from the v3.3 build: conflicts={conflicts} extra={extra_keys}")
    env["LOL_CFG_OVERRIDES"] = json.dumps(effective_build)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("OMP_NUM_THREADS", "1")

    git_commit, git_dirty = _git_state()
    prices = check_price_table(m)
    deviations = list(DEVIATIONS)
    if not prices["same_file"]:
        deviations.append(
            f"config/game_rules/event_prices.json differs in bytes from the build (sha1 {prices['sha1_now']} vs "
            f"{prices['sha1_build']}); its pooled prices equal those at the build commit {prices.get('build_commit')} "
            "(only descriptive fields changed), and every shard's y_market_event identity check confirms the labels.")
    manifest = {
        "item": "A2-label-family", "script": "scripts/build_label_sidecars_v33.py",
        "run_id": time.strftime("%Y%m%d_%H%M%S"), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit, "git_dirty": git_dirty, "python": sys.executable,
        "preset": "v3.3", "effective_overrides": effective_build, "lol_output_root": os.environ.get("LOL_OUTPUT_ROOT"),
        "corpus": {"dir": str(args.corpus), **{k: m.get(k) for k in ("run_id", "git_commit", "seed", "num_shards", "n_matches", "feature_names_sha1")},
                   "label": m.get("label")},
        "row_label": {"label_type": m["label"]["row_label"], "tie_policy": m["label"]["row_tie_policy"]},
        "variant_tie_policy": m["label"]["extra_tie_policy"], "price_table": prices,
        "seed": int(m["seed"]), "shard_partition": "scripts/build_corpus_shard.py shard_match_ids",
        "limit_matches_per_shard": limit or None, "shards_requested": shards, "index_workers": args.index_workers,
        "parallel": args.parallel,
        "variants": {n: get_label_variant(n).as_dict() for n in names},
        "deviations": deviations, "shards": {}, "complete": False,
    }
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        old = json.load(open(manifest_path, encoding="utf-8"))
        for k, v in (old.get("shards") or {}).items():
            if int(k) not in todo:
                manifest["shards"][k] = v

    def dump():
        json.dump(manifest, open(manifest_path, "w", encoding="utf-8"), indent=1)
    dump()

    running: dict[int, tuple] = {}
    pending = list(todo)
    failed: list[int] = []
    while pending or running:
        while pending and len(running) < args.parallel and not failed:
            i = pending.pop(0)
            log = open(out_dir / f"labels_shard_{i:03d}.log", "w", encoding="utf-8")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--shard", str(i), "--corpus", str(args.corpus),
                   "--out-dir", str(out_dir), "--variants", ",".join(names), "--index-workers", str(args.index_workers)]
            if limit:
                cmd += ["--limit-matches-per-shard", str(limit)]
            running[i] = (subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print(f"[shard {i}] started ({len(running)} running, {len(pending)} pending)", flush=True)
        time.sleep(args.poll_s)
        for i in list(running):
            proc, log, t_i = running[i]
            rc = proc.poll()
            if rc is None:
                continue
            log.close()
            del running[i]
            entry = {"rc": int(rc), "wall_s": round(time.time() - t_i, 1)}
            info_path = out_dir / f"labels_shard_{i:03d}.json"
            if rc == 0 and info_path.exists():
                info = json.load(open(info_path, encoding="utf-8"))
                entry.update({"n_rows": info["n_rows"], "n_matches_with_rows": info.get("n_matches_with_rows"),
                              "n_matches_used": info.get("n_matches_used"), "identity_ok": bool(info["identity"]["ok"]),
                              "file": f"labels_shard_{i:03d}.npz", "timing_s": info["timing_s"]})
            manifest["shards"][str(i)] = entry
            dump()
            print(f"[shard {i}] rc={rc} after {entry['wall_s']:.0f}s" + (f" rows={entry.get('n_rows')}" if rc == 0 else ""), flush=True)
            if rc != 0:
                failed.append(i)
                print("!" * 100, flush=True)
                print(f"[shard {i}] FAILED (rc={rc}{', identity mismatch' if rc == IDENTITY_EXIT else ''}); log tail:", flush=True)
                print(_tail(out_dir / f"labels_shard_{i:03d}.log"), flush=True)
                print("!" * 100, flush=True)
                if running:
                    print(f"terminating {len(running)} running shard(s)", flush=True)
                for j, (p, lg, _) in list(running.items()):
                    p.terminate()
                    try:
                        p.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        p.kill()
                    lg.close()
                    manifest["shards"][str(j)] = {"rc": None, "terminated": True}
                running.clear()
                pending.clear()
                dump()

    ok = [i for i in shards if (manifest["shards"].get(str(i)) or {}).get("identity_ok")]
    manifest["complete"] = len(ok) == len(shards) and not failed
    manifest["full"] = bool(manifest["complete"] and shards == list(range(num_shards)) and not limit)
    manifest["n_rows"] = int(sum((manifest["shards"].get(str(i)) or {}).get("n_rows", 0) for i in shards))
    manifest["n_matches_with_rows"] = int(sum((manifest["shards"].get(str(i)) or {}).get("n_matches_with_rows") or 0 for i in shards))
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["wall_clock_s"] = round(time.time() - t_run, 1)
    dump()
    if failed:
        raise SystemExit(f"label sidecar build ABORTED: shard(s) {failed} failed; see {out_dir}")
    print(f"sidecars done in {(time.time() - t_run) / 60:.1f} min: {len(ok)}/{len(shards)} shards identical to the corpus, "
          f"{manifest['n_rows']} rows, full={manifest['full']}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="corpus_shards_v33 (read only)")
    ap.add_argument("--shards", default="all", help="'all' or e.g. '0,31' / '0-7'")
    ap.add_argument("--variants", default="all", help="'all' registry variants or a comma list (market_event is always added)")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--index-workers", type=int, default=0,
                    help="build_fight_index processes per shard (0 = the build's auto rule, min(cpu, 8))")
    ap.add_argument("--limit-matches-per-shard", type=int, default=0,
                    help="smoke tests: first N matches of each shard's own order (rows stay a corpus prefix)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs of the requested shards")
    ap.add_argument("--skip-existing", action="store_true", help="keep shards already built with the same settings")
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=-1, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return run_worker(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
