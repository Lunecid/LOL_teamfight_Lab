"""T2.2-T2.3 E2 run: prediction-time group tables, Q_DEV evaluation, lock, and the single TEST step.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E2_KO.md (PLAN_E1_E5_KO.md Task 4; SDD Rulings 23, 30).
Stages, run in this order and each once; every output is written once under D:/LOL_Training_20261004/tog_e2_1.

  groups  One pass over the observable-state bank (main process + 3 workers).  For each primary-supported bank
          component the frame F(tau) = bank xy[F_pre_index] is checked as headcount_tau.py:205-213 does (frame index
          inside the match frames, minute_ts[F_pre_index] == F_pre_ms, frame age pre - F_pre_ms equal to the stored
          age and within 0-10 s) and the team map as headcount_tau.py:171-172 does (five players of team 100 and five
          of team 200).  Only a component that passes goes to e2_groups.pre_tau_group, at radius 1,800, 2,000 and
          2,500; one that fails is counted, never dropped silently.  The components are aligned to the q rows of
          q_run_1/Q_ROWS_h90.npz (exactly one per q row); each q row's pre_ms and frame age must equal the bank's and
          its q input frame (q_run_1/inputs/main_h90/XY.npy) must equal the bank frame.  Writes
          E2_GROUPS_r<radius>_<role>.npz (engagement_key, n_small, n_large, n_unknown, stratum; Q_ROWS_h90 order) and
          .json for TRAIN, Q_DEV, Q_CAL and TEST, then E2_GROUPS_BUILD.json.  TEST tables come from bank coordinates
          and row keys only: no label is read.
  qdev    Per radius and stratum on Q_DEV: match-weighted Brier of P1 (LightGBM) and PT_flex from the stored Q_DEV
          predictions (raw p) of the search jobs the two final cells reuse, the difference P1 - PT_flex (negative:
          P1 better) with a 95 % percentile interval from qeval's paired match bootstrap (2,000 draws, PCG64 seed
          20261011, matches in ascending id order, draws shared by both cells), each match weighing equally within
          the stratum (scale_headcount.py:177-205); AUC (secondary) of both cells and their difference on the same
          draws.  Writes E2_QDEV.json: development evidence, never used for the decision.
  lock    E2_LOCK.json: PRESPEC_E2's SHA-256 and the commit holding it, code, inputs, the two TEST_SCORES cells and
          q_run_1/Q_TEST_SEALED_h90.npz (file bytes hashed, never opened), the strata tables of the three radii with
          their hashes, the Q_DEV result, the selected cells, the rules and the UTC time.
  test    Only after the review of the lock (Ruling 30): every hash in the lock is checked, the opening is appended to
          reports/tog_experiments_20261012/OPENINGS.json, the sealed TEST labels are opened once, TEST is evaluated
          like Q_DEV, and E2_REPORT.json records the decision of PRESPEC_E2 section 5.

A BLOCKED condition (PRESPEC_E2 section 7, or a team map that fails, Ruling 23) writes E2_BLOCKED_<stage>_<n>.json and
ends the stage with status 2.  At most 4 CPU processes, no GPU; players are bank slots 1-10, never identifiers.

Usage: python -B -m studies.tog_experiments_20261012.e2_run groups [--workers 3] [--smoke-shards N]
       python -B -m studies.tog_experiments_20261012.e2_run qdev
       python -B -m studies.tog_experiments_20261012.e2_run lock
       python -B -m studies.tog_experiments_20261012.e2_run test --confirm-single-opening
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Iterable, Mapping  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005.engagements import BANK, COMPONENT_COLUMNS, shard_dirs  # noqa: E402
from studies.q_production_20261005 import qeval  # noqa: E402
from studies.q_production_20261005.qtest import load_npz, read_json  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, TOG_E2, document_record, git_head, peak_memory_bytes, run_record, sha256_file, utc, write_npz_once,
    write_once)
from studies.tog_experiments_20261012.e2_groups import pre_tau_group, stratum  # noqa: E402

VERSION = "tog_e2_run_v1_20261012"
RADII = (1800.0, 2000.0, 2500.0)
DECISION_RADIUS, DECISION_STRATUM = 2000.0, "teamfight"
STRATA = ("separated", "pick", "small", "asymmetric", "teamfight")
ROLE_CODES = dict(Q_ROLES)                     # TRAIN 0, Q_DEV 3, Q_CAL 4, TEST 5
ROLES = tuple(ROLE_CODES)
SEED, RESAMPLES, LEVEL = qeval.BOOTSTRAP_SEED, qeval.BOOTSTRAP_RESAMPLES, qeval.EXPLORATORY_LEVEL  # 20261011, 2000, 0.95
RECOMPUTE_ATOL = 1e-10
CELLS = {  # final cell -> the search job it reuses (its REUSE.json reuse_of_key), whose pred_q_dev.npz is its Q_DEV score
    "P1": ("final__module__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__selected~lightgbm",
           "search__auxiliary__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__lgbm_l15_m200"),
    "PT_flex": ("final__reference__PT_FLEX__PT_flex__selected~PT_flex",
                "search__reference__PT_FLEX__PT_flex__ptflex_k8_C0.001"),
}
CONTRAST = "P1_minus_PT_flex"
CONTRAST_TEXT = ("Brier(P1, LightGBM LGBM_S1_A1_M1_E1_C1_R1_D1) - Brier(PT_flex), match-weighted; negative means P1 is "
                 "better (PRESPEC_E2 line 5; the opposite sign of E1's G)")
QDEV_POLICY = ("PRESPEC_E2 section 6: the decision uses the TEST interval only; this Q_DEV result is development "
               "evidence and changes neither the radius nor the strata")
BUILD_NAME, QDEV_NAME, LOCK_NAME, REPORT_NAME = "E2_GROUPS_BUILD.json", "E2_QDEV.json", "E2_LOCK.json", "E2_REPORT.json"
Q_GROUP_FIELDS = ("engagement_key", "canonical_match_key", "component_key", "match_index", "role", "pre_ms", "frame_age_s")
BANK_FIELDS = ("component_values", "component_key", "xy", "minute_ts", "team_ids")
COL = {name: index for index, name in enumerate(COMPONENT_COLUMNS)}
ROW_CHECKS = ("frame_index_inside_match_frames", "frame_time_equals_F_pre_ms",
              "frame_age_equals_stored_and_within_0_10000", "team_map_five_of_100_and_five_of_200")
TEAM_BIT = 1 << 3
CODE_FILES = ("studies/tog_experiments_20261012/e2_run.py", "studies/tog_experiments_20261012/e2_groups.py",
              "studies/tog_experiments_20261012/common.py", "studies/tog_experiments_20261012/__init__.py",
              "studies/q_production_20261005/qeval.py", "studies/q_production_20261005/qtest.py",
              "studies/q_production_20261005/rows.py", "studies/label_production_20261005/engagements.py")
DECISIONS = {  # PRESPEC_E2 section 5, on the TEST interval of the radius-2,000 teamfight stratum
    "reversal": "하한 > 0 (P1이 나쁨): '예측 시점 큰 무리에서도 역전'으로 주장 3의 큰 싸움 부분을 그대로 쓴다.",
    "gain_remains": ("상한 < 0 (P1이 나음): '역전은 킬 기록으로 고른 큰 싸움에서만 나타나고, 예측 시점 큰 무리에서는 "
                     "이득이 남는다'로 한정하고, 결과로 표본을 고른 문제를 한계에 적는다."),
    "undetermined": ("0을 포함: 역전은 킬 기록으로 고른 큰 싸움에서만 나타난다로 한정하고 '예측 시점 큰 무리에서는 "
                     "미결'로 적는다."),
}
DEFINITION = (
    "PRESPEC_E2 section 3. Input frame F(tau) = bank xy[F_pre_index] (minute_ts[F_pre_index] == F_pre_ms, frame age "
    "0-10 s; headcount_tau.py:205-213). A player whose coordinate is not finite or whose x or y is 0 is unknown: left "
    "out of the graph and counted in n_unknown. Two remaining players of any team are linked when their Euclidean "
    "distance is <= radius; a connected component holding players of both teams is a mixed group; the chosen group "
    "maximises (size, smaller-team count, -smallest participant index), participant_order 1-10. n_small <= n_large are "
    "its per-team counts. Strata: pick (n_small <= 1), small (both 2-3), asymmetric (one side 2-3, the other 4 or "
    "more), teamfight (both 4 or more); separated when there is no mixed group. The first-kill position is not used.")
RADIUS_NOTE = ("2,000 = studies/observable_state_20261002/population.py LINK_RADIUS_UNITS (fixed in early October, not "
               "chosen for E2); 1,800 and 2,500 are reported as sensitivity only")


@dataclass(frozen=True)
class Paths:
    """Where the stages read and write; the defaults are the real run (the tests point them at tmp_path)."""

    out: Path = TOG_E2
    bank: Path = BANK
    q_run: Path = Q_RUN
    engagements_manifest: Path = LABELS / "ENGAGEMENTS_MANIFEST.json"
    prespec: Path = ROOT / "reports/tog_experiments_20261012/PRESPEC_E2_KO.md"
    exposure: Path = ROOT / "reports/tog_experiments_20261012/EXPOSURE_KO.md"
    openings: Path = ROOT / "reports/tog_experiments_20261012/OPENINGS.json"

    @property
    def q_rows(self) -> Path:
        return self.q_run / "Q_ROWS_h90.npz"

    @property
    def xy(self) -> Path:
        return self.q_run / "inputs" / "main_h90" / "XY.npy"

    @property
    def sealed(self) -> Path:
        return self.q_run / "Q_TEST_SEALED_h90.npz"

    @property
    def jobs(self) -> Path:
        return self.q_run / "jobs"

    @property
    def scores(self) -> Path:
        return self.q_run / "TEST_SCORES"

    def table(self, radius: float, role: str, suffix: str) -> Path:
        return self.out / f"E2_GROUPS_r{int(radius)}_{role}.{suffix}"


# ---------------------------------------------------------------- checks before the group function

def team_ok(team_ids: Any) -> bool:
    """Five players of team 100 and five of team 200 (headcount_tau.py:171-172)."""

    team = np.asarray(team_ids)
    return team.shape == (10,) and int((team == 100).sum()) == 5 and int((team == 200).sum()) == 5


def frame_failures(fidx: int, n_frames: int, minute_ts: Any, f_pre_ms: int, pre: int, stored_age: int) -> tuple[int, int]:
    """(failed bits, frame age) of headcount_tau.py:205-213: bit 0 frame index outside the match frames (or a clock of
    another length), bit 1 frame time != F_pre_ms, bit 2 age != stored age or outside 0-10,000 ms."""

    clock = np.asarray(minute_ts)
    failed = 0
    frame_ok = 0 <= fidx < n_frames and clock.shape[0] == n_frames
    if not frame_ok:
        failed |= 1 << 0
    frame_ms = int(clock[fidx]) if frame_ok else -1
    if frame_ms != int(f_pre_ms):
        failed |= 1 << 1
    age = int(pre) - frame_ms
    if not (age == int(stored_age) and 0 <= age <= 10_000):
        failed |= 1 << 2
    return failed, age


# ---------------------------------------------------------------- bank pass (worker) and the tables

def scan_shard(directory_text: str) -> dict[str, Any]:
    """Frame, checks and groups (at the three radii) of every primary-supported component of one bank shard.

    Worker task.  Only the five arrays the definition needs (and their offsets) are decompressed; the shard's SHA-256
    comes from the same bytes.  A component whose frame or team map fails gets groups -1 and is not passed to
    pre_tau_group."""

    start = time.perf_counter()
    directory = Path(directory_text)
    raw = (directory / "BANK.npz").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with np.load(io.BytesIO(raw), allow_pickle=False) as bank:
        arrays = {name: bank[name] for base in BANK_FIELDS for name in (base, "offset_" + base)}
    del raw
    with (directory / "MATCH_LEDGER.jsonl").open(encoding="utf-8") as stream:
        ledger = [json.loads(line) for line in stream if line.strip()]
    for base in BANK_FIELDS:
        if len(arrays["offset_" + base]) != len(ledger) + 1:
            raise ValueError(f"{directory.name}: {base} offsets do not cover the match ledger")

    def part(name: str, m: int) -> np.ndarray:
        offsets = arrays["offset_" + name]
        return arrays[name][int(offsets[m]):int(offsets[m + 1])]

    keys: list[bytes] = []
    frames, failed, pre, age, groups = [], [], [], [], []
    no_frame = np.full((10, 2), np.nan, dtype=np.float32)
    for m, entry in enumerate(ledger):
        values = part("component_values", m)
        selected = np.flatnonzero((values[:, COL["primary_candidate_eligible"]] != 0)
                                  & (values[:, COL["supported_candidate_eligible"]] != 0))
        if not selected.size:
            continue
        match_key = str(entry["canonical_match_key"]).encode("ascii")
        component_keys = part("component_key", m)
        xy, clock, team = part("xy", m), part("minute_ts", m), part("team_ids", m)
        good_team = team_ok(team)
        for c in selected.tolist():
            v = values[c]
            fidx, start_ms = int(v[COL["F_pre_index"]]), int(v[COL["pre"]])
            bits, frame_age = frame_failures(fidx, xy.shape[0], clock, int(v[COL["F_pre_ms"]]), start_ms,
                                             int(v[COL["pre_frame_age_ms"]]))
            frame = no_frame if bits & 1 else xy[fidx]           # headcount_tau.py:232
            if not good_team:
                bits |= TEAM_BIT
            found = np.full((len(RADII), 3), -1, dtype=np.int8)
            if not bits:
                for k, radius in enumerate(RADII):
                    found[k] = pre_tau_group(frame, team, radius)
            keys.append(match_key + b":" + bytes(component_keys[c]))
            frames.append(frame)
            failed.append(bits)
            pre.append(start_ms)
            age.append(frame_age)
            groups.append(found)
    count = len(keys)
    return {"key": np.asarray(keys, dtype="S129"), "frame": np.asarray(frames, dtype=np.float32).reshape(count, 10, 2),
            "failed": np.asarray(failed, dtype=np.uint8), "pre": np.asarray(pre, dtype=np.int64),
            "age": np.asarray(age, dtype=np.int64), "groups": np.asarray(groups, dtype=np.int8).reshape(count, len(RADII), 3),
            "shard": directory.name, "sha256": digest, "n_matches": len(ledger),
            "seconds": time.perf_counter() - start, "peak_memory_bytes": peak_memory_bytes()}


def align_to_q_rows(bank_keys: np.ndarray, q_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(target, hits): the q row of each bank component (-1 if it is not a q row) and the components found per q row."""

    n = len(q_keys)
    order = np.argsort(q_keys, kind="stable")
    sorted_keys = q_keys[order]
    pos = np.searchsorted(sorted_keys, bank_keys)
    pos_clip = np.minimum(pos, n - 1)
    in_q = (pos < n) & (sorted_keys[pos_clip] == bank_keys)
    target = np.where(in_q, order[pos_clip], -1)
    return target, np.bincount(target[in_q], minlength=n)


def _stratum_or_blank(n_small: int, n_large: int) -> str:
    try:
        return stratum(n_small, n_large)
    except ValueError:
        return ""


_STRATUM_OF = np.array([[_stratum_or_blank(a, b) for b in range(6)] for a in range(6)], dtype="<U10")


def build_groups(role: str, radius: float, aligned: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The group table of one q role at one radius, in Q_ROWS_h90 row order.

    ``aligned`` holds the q rows' engagement_key, role and groups [row, radius, (n_small, n_large, n_unknown)], with -1
    where a row failed its checks (its stratum is then blank, so the strata no longer sum to the rows)."""

    rows = np.flatnonzero(np.asarray(aligned["role"]) == ROLE_CODES[role])
    found = np.asarray(aligned["groups"])[rows, RADII.index(float(radius))]
    n_small, n_large, n_unknown = (found[:, k].astype(np.int8) for k in range(3))
    labels = np.full(rows.size, "", dtype="<U10")
    valid = (n_small >= 0) & (n_large >= 0)
    labels[valid] = _STRATUM_OF[n_small[valid], n_large[valid]]
    return {"engagement_key": np.asarray(aligned["engagement_key"])[rows], "n_small": n_small, "n_large": n_large,
            "n_unknown": n_unknown, "stratum": labels}


def _table_summary(table: Mapping[str, np.ndarray], match: np.ndarray) -> dict[str, Any]:
    labels = table["stratum"]
    pairs, counts = np.unique(np.stack([table["n_small"], table["n_large"]], axis=1), axis=0, return_counts=True)
    return {"rows": int(labels.size), "matches": int(np.unique(match).size),
            "strata": {s: {"rows": int((labels == s).sum()), "matches": int(np.unique(match[labels == s]).size),
                           "row_share": float((labels == s).mean()) if labels.size else None} for s in STRATA},
            "n_small_n_large_rows": {f"{a}:{b}": int(k) for (a, b), k in zip(pairs.tolist(), counts.tolist())},
            "n_unknown": {"rows_with_unknown": int((table["n_unknown"] > 0).sum()),
                          "unknown_slots": int(table["n_unknown"][table["n_unknown"] > 0].sum())}}


def groups_stage(paths: Paths = Paths(), *, workers: int = 3, smoke_shards: int = 0) -> int:
    """Build, check and write the group tables of every q role at every radius (module docstring, stage groups)."""

    if not 0 <= workers <= 3:
        raise ValueError("workers must be 0-3 (main + workers <= 4 CPU processes; 0 runs in this process)")
    smoke = smoke_shards > 0
    started, tick = utc(), time.perf_counter()
    targets = [paths.table(r, name, s) for r in RADII for name in ROLES for s in ("npz", "json")] + [paths.out / BUILD_NAME]
    if not smoke and any(path.exists() for path in targets):
        raise FileExistsError(f"{paths.out}: group tables exist; they are written once")
    manifest = read_json(paths.bank / "FIELD_MANIFEST.json")
    if (tuple(manifest["component_columns"]) != COMPONENT_COLUMNS or manifest["participant_order"] != list(range(1, 11))
            or list(manifest["team_order"]) != [100, 200]):
        raise ValueError("bank FIELD_MANIFEST layout differs from this code")
    shard_record = read_json(paths.engagements_manifest)["inputs"]["bank_shards_sha256"]
    q = _load_fields(paths.q_rows, Q_GROUP_FIELDS)
    if sha256_file(paths.q_rows) != read_json(paths.q_rows.with_suffix(".json"))["rows_sha256"]:
        raise ValueError("Q_ROWS_h90.npz differs from its manifest rows_sha256")
    if not np.array_equal(np.char.add(np.char.add(q["canonical_match_key"], b":"), q["component_key"]), q["engagement_key"]):
        raise ValueError("Q_ROWS_h90 engagement_key is not canonical_match_key:component_key")
    keys, role, match = q["engagement_key"], np.asarray(q["role"]), np.asarray(q["match_index"])
    n = len(keys)
    directories = shard_dirs(paths.bank)
    if smoke:
        directories = directories[:smoke_shards]
    t_scan = time.perf_counter()
    if workers:
        with Pool(workers) as pool:
            results = list(pool.imap(scan_shard, [str(d) for d in directories], chunksize=1))
    else:
        results = [scan_shard(str(d)) for d in directories]
    scan_seconds = time.perf_counter() - t_scan
    bank = {name: np.concatenate([r[name] for r in results]) for name in ("key", "frame", "failed", "pre", "age", "groups")}

    # ---------------------------------------------- align to the q rows and check every row
    target, hits = align_to_q_rows(bank["key"], keys)
    found = target >= 0
    filled = hits == 1

    def place(values: np.ndarray, fill: Any, dtype: Any) -> np.ndarray:
        column = np.full((n, *values.shape[1:]), fill, dtype=dtype)
        column[target[found]] = values[found]
        return column

    failed = place(bank["failed"], 0, np.uint8)
    groups = place(bank["groups"], -1, np.int8)
    pre_ok = place(bank["pre"], -1, np.int64) == np.asarray(q["pre_ms"])
    age_ok = place(bank["age"], -1, np.int64) == np.rint(np.asarray(q["frame_age_s"]) * 1000.0).astype(np.int64)
    xy = np.load(paths.xy, allow_pickle=False)
    if xy.shape != (n, 10, 2):
        raise ValueError(f"XY.npy has shape {xy.shape}, expected ({n}, 10, 2)")
    frame = place(bank["frame"], np.nan, np.float32).astype(np.float64)
    xy_ok = ((frame == xy) | (np.isnan(frame) & np.isnan(xy))).all(axis=(1, 2))
    del frame, xy
    n_unknown = groups[:, :, 2]
    passed = filled & (failed == 0)

    def by_role(mask: np.ndarray) -> dict[str, int]:
        return {name: int((mask & (role == code)).sum()) for name, code in ROLE_CODES.items() if (mask & (role == code)).any()}

    masks = {name: filled & (((failed >> bit) & 1) == 1) for bit, name in enumerate(ROW_CHECKS)}
    masks.update({"q_rows_with_several_bank_components": hits > 1,
                  "q_row_pre_ms_equals_bank_pre": filled & ~pre_ok,
                  "q_row_frame_age_equals_bank_age": filled & ~age_ok,
                  "frame_equals_q_input_xy": filled & ~xy_ok,
                  "rows_with_unknown_position": passed & (n_unknown[:, 0] > 0),
                  "n_unknown_differs_between_radii": passed & ~(n_unknown == n_unknown[:, :1]).all(axis=1)})
    if not smoke:
        masks["q_rows_without_bank_component"] = hits == 0
    checks: dict[str, Any] = {"q_row_keys_repeat": int(n - np.unique(keys).size),
                              "bank_component_keys_repeat": int(bank["key"].size - np.unique(bank["key"]).size)}
    checks.update({name: by_role(mask) for name, mask in masks.items()})
    shard_hash = {r["shard"]: r["sha256"] for r in results}
    if not smoke:
        checks["bank_shards_missing_or_extra"] = sorted(set(shard_hash) ^ set(shard_record))
        checks["bank_shards_differ_from_label_record"] = sorted(k for k, v in shard_hash.items() if shard_record.get(k) != v)
    aligned = {"engagement_key": keys, "role": role, "groups": groups}
    tables, summaries = {}, {}
    for radius in RADII:
        for name in ROLES:
            rows = np.flatnonzero(role == ROLE_CODES[name])
            table = build_groups(name, radius, aligned)
            summary = _table_summary(table, match[rows])
            label = f"r{int(radius)}:{name}"
            checks[f"strata_sum_differs_from_rows:{label}"] = summary["rows"] - sum(s["rows"] for s in summary["strata"].values())
            checks[f"keys_differ_from_q_rows_order:{label}"] = int(not np.array_equal(table["engagement_key"], keys[rows]))
            tables[(radius, name)], summaries[(radius, name)] = table, summary
    worker_peak = max((r["peak_memory_bytes"] or 0 for r in results), default=None)
    stage_extra = {"workers": workers, "shards": len(results), "scan_wall_seconds": scan_seconds,
                   "worker_cpu_seconds_sum": float(sum(r["seconds"] for r in results)),
                   "peak_memory_bytes_worker_max": worker_peak}
    if smoke:
        print(json.dumps(_jsonable({"smoke_shards": len(results), "q_rows_filled": int(filled.sum()),
                                    "failing_checks_on_filled_rows": {k: v for k, v in checks.items()
                                                                      if _count(v) and not k.startswith("strata_sum")},
                                    "strata_of_filled_rows": {f"r{int(r)}:{name}": dict(zip(*np.unique(
                                        tables[(r, name)]["stratum"][filled[role == ROLE_CODES[name]]], return_counts=True)))
                                        for r in RADII for name in ROLES},
                                    "stage": _stage("groups-smoke", started, tick, **stage_extra)}), indent=1))
        return 0
    reasons = {k: v for k, v in checks.items() if _count(v)}
    if reasons:
        examples = {name: [k.decode("ascii") for k in keys[mask][:10].tolist()] for name, mask in masks.items() if mask.any()}
        return _blocked(paths.out, "groups", reasons, checks=checks, examples=examples,
                        stage=_stage("groups", started, tick, **stage_extra))

    # ---------------------------------------------- write the tables, then the build record
    paths.out.mkdir(parents=True, exist_ok=True)
    rows_by_role = {name: int((role == code).sum()) for name, code in ROLE_CODES.items()}
    matches_by_role = {name: int(np.unique(match[role == code]).size) for name, code in ROLE_CODES.items()}
    record = run_record(rows=rows_by_role, matches=matches_by_role,
                        inputs=[paths.q_rows, paths.q_rows.with_suffix(".json"), paths.xy, paths.bank / "FIELD_MANIFEST.json",
                                paths.bank / "RUN_MANIFEST.json", paths.engagements_manifest],
                        code=[ROOT / name for name in CODE_FILES], started_utc=started)
    bank_record = {"path": str(paths.bank), "shards": len(results), "shard_sha256_digest": _digest(shard_hash),
                   "shards_equal_label_production_record": True, "label_production_record": str(paths.engagements_manifest),
                   "field_manifest_sha256": sha256_file(paths.bank / "FIELD_MANIFEST.json"),
                   "run_manifest_sha256": sha256_file(paths.bank / "RUN_MANIFEST.json")}
    files: dict[str, dict[str, str]] = {}
    for (radius, name), table in tables.items():
        npz_path, json_path = paths.table(radius, name, "npz"), paths.table(radius, name, "json")
        digest = write_npz_once(npz_path, table)
        back = load_npz(npz_path)
        if (set(back) != set(table) or any(back[k].dtype != table[k].dtype or not np.array_equal(back[k], table[k]) for k in table)
                or not np.array_equal(back["engagement_key"], keys[role == ROLE_CODES[name]])):
            raise RuntimeError(f"{npz_path.name}: the written table differs on reload")
        doc = {"version": VERSION, "experiment": "E2", "role": name, "role_code": ROLE_CODES[name], "radius": radius,
               **summaries[(radius, name)],
               "npz": {"path": str(npz_path), "sha256": digest, "fields": list(table),
                       "order": "the Q_ROWS_h90 rows of this role in file order"},
               "checks": {"strata_sum_equals_rows": True, "keys_equal_q_rows_order": True,
                          "rows_failing_frame_or_team_checks": 0, "rows_with_unknown_position": 0},
               "labels_read": "none: bank coordinates and the Q_ROWS_h90 key, role, match, pre-time and frame-age columns",
               "q_row_columns_read": list(Q_GROUP_FIELDS), "definition": DEFINITION, "radius_note": RADIUS_NOTE,
               "bank": bank_record, "build_record": str(paths.out / BUILD_NAME), "seed": None, "run": record}
        write_once(json_path, _jsonable(doc))
        files[f"r{int(radius)}:{name}"] = {"npz": str(npz_path), "npz_sha256": digest, "json": str(json_path),
                                           "json_sha256": sha256_file(json_path)}
    build = {"version": VERSION, "experiment": "E2", "passed": True, "checks": checks, "tables": files,
             "q_rows": n, "rows_by_role": rows_by_role, "matches_by_role": matches_by_role, "radii": list(RADII),
             "strata": list(STRATA), "row_checks": list(ROW_CHECKS), "definition": DEFINITION, "radius_note": RADIUS_NOTE,
             "labels_read": "none", "bank": {**bank_record, "shard_sha256": shard_hash},
             "all_bank_components": {"primary_supported": int(bank["key"].size), "not_q_rows": int((~found).sum()),
                                     "failing": {name: int((((bank["failed"] >> bit) & 1) == 1).sum())
                                                 for bit, name in enumerate(ROW_CHECKS)}},
             "stage": _stage("groups", started, tick, **stage_extra), "run": record}
    write_once(paths.out / BUILD_NAME, _jsonable(build))
    for (radius, name), summary in summaries.items():
        print(f"r{int(radius)} {name} rows={summary['rows']}", {s: v["rows"] for s, v in summary["strata"].items()})
    print(f"groups written: {len(files)} tables, {time.perf_counter() - tick:.1f} s")
    return 0


# ---------------------------------------------------------------- evaluation (Q_DEV and TEST alike)

def _auc_order(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(p, kind="stable")
    values = p[order]
    return order, np.flatnonzero(np.r_[True, values[1:] != values[:-1]]), y[order] == 1.0


def _weighted_auc(prepared: tuple[np.ndarray, np.ndarray, np.ndarray], weights: np.ndarray) -> float | None:
    """qeval._auc (weighted Mann-Whitney, ties 1/2) on a fixed sort order; None when a class has zero weight."""

    order, starts, positive = prepared
    w = weights[order]
    pos, neg = np.where(positive, w, 0.0), np.where(positive, 0.0, w)
    total_pos, total_neg = pos.sum(), neg.sum()
    if not (total_pos > 0.0 and total_neg > 0.0):
        return None
    tied_pos, tied_neg = np.add.reduceat(pos, starts), np.add.reduceat(neg, starts)
    below = np.concatenate(([0.0], np.cumsum(tied_neg)[:-1]))
    return float(np.sum(tied_pos * (below + 0.5 * tied_neg)) / (total_pos * total_neg))


def auc_difference(y: Any, p_a: Any, p_b: Any, match: Any, *, resamples: int = RESAMPLES, seed: int = SEED,
                   level: float = LEVEL, return_draws: bool = False) -> dict[str, Any]:
    """Match-weighted AUC(p_a) - AUC(p_b) with a percentile interval on the paired match draws of the Brier contrast.

    Draw b is qeval.bootstrap_indices over the ascending unique match ids (the order per_match_loss uses); a match drawn
    c times gives each of its m rows weight c / m.  The interval is None when a draw leaves one class only."""

    y = np.asarray(y, dtype=np.float64)
    unique, code, counts = np.unique(np.asarray(match), return_inverse=True, return_counts=True)
    base = 1.0 / counts.astype(np.float64)
    prepared = [_auc_order(y, np.asarray(p, dtype=np.float64)) for p in (p_a, p_b)]

    def pair(weights: np.ndarray) -> float | None:
        a, b = (_weighted_auc(prep, weights) for prep in prepared)
        return None if a is None or b is None else a - b

    point = pair(base[code])
    draws = np.full(int(resamples), np.nan)
    for b, positions in enumerate(qeval.bootstrap_indices(unique.size, resamples, seed)):
        value = pair((np.bincount(positions, minlength=unique.size) * base)[code])
        if value is not None:
            draws[b] = value
    undefined = int(np.isnan(draws).sum())
    quantiles = qeval.bootstrap_quantiles(level)
    interval = None if undefined or point is None else [float(v) for v in np.quantile(draws, quantiles)]
    out = {"point": point, "interval": interval, "level": float(level), "quantiles": list(quantiles),
           "undefined_draws": undefined, "resamples": int(resamples), "seed": int(seed)}
    if return_draws:
        out["draws"] = draws
    return out


def evaluate_pair(y: Any, p_p1: Any, p_pt: Any, match: Any) -> dict[str, Any]:
    """P1 - PT_flex on one set of rows: match-weighted Brier difference with its 95 % paired match bootstrap interval
    (qeval, as scale_headcount.py:203-209), the cells' match-weighted metrics, and the AUC difference on the same draws."""

    y = np.asarray(y, dtype=np.float64)
    cells = {"P1": np.asarray(p_p1, dtype=np.float64), "PT_flex": np.asarray(p_pt, dtype=np.float64)}
    match = np.asarray(match)
    losses = {name: qeval.per_match_loss(y, p, match) for name, p in cells.items()}
    shared, aligned = qeval.align_cells(losses)
    boot = qeval.paired_match_bootstrap(shared, aligned, {CONTRAST: qeval.difference("P1", "PT_flex")}, levels=LEVEL,
                                        resamples=RESAMPLES, seed=SEED)
    contrast = boot["contrasts"][CONTRAST]
    metrics = {name: {k: v for k, v in qeval.evaluate(y, p, match)["match_weighted"].items() if k != "reliability"}
               for name, p in cells.items()}
    auc = auc_difference(y, cells["P1"], cells["PT_flex"], match)
    for name in cells:  # independent route: the qeval match-weighted Brier equals the mean of the per-match losses
        if abs(metrics[name]["brier"] - boot["cells"][name]["point"]) > RECOMPUTE_ATOL:
            raise RuntimeError(f"{name}: match-weighted Brier differs between qeval.evaluate and the per-match losses")
    expected = (None if metrics["P1"]["auc"] is None or metrics["PT_flex"]["auc"] is None
                else metrics["P1"]["auc"] - metrics["PT_flex"]["auc"])
    if (expected is None) != (auc["point"] is None) or (expected is not None and abs(expected - auc["point"]) > RECOMPUTE_ATOL):
        raise RuntimeError("the AUC difference differs from qeval.evaluate")
    return {"rows": int(y.size), "matches": int(shared.size), "positives": int(y.sum()),
            "brier": {name: metrics[name]["brier"] for name in cells},
            "brier_difference": {"point": contrast["point"], "interval": contrast["interval"], "level": contrast["level"],
                                 "quantiles": contrast["quantiles"]},
            "auc": {name: metrics[name]["auc"] for name in cells}, "auc_difference": auc, "metrics": metrics}


def evaluate_strata(y: Any, p_p1: Any, p_pt: Any, match: Any, labels: Any) -> dict[str, Any]:
    """evaluate_pair inside each stratum (its own matches and match weights); an empty stratum reports zero rows."""

    labels = np.asarray(labels).astype(str)
    outside = sorted(set(np.unique(labels).tolist()) - set(STRATA))
    if outside:
        raise ValueError(f"labels outside the strata: {outside}")
    y, p_p1, p_pt, match = (np.asarray(a) for a in (y, p_p1, p_pt, match))
    return {name: (evaluate_pair(y[labels == name], p_p1[labels == name], p_pt[labels == name], match[labels == name])
                   if (labels == name).any() else {"rows": 0, "matches": 0}) for name in STRATA}


def decide(interval: Iterable[float]) -> str:
    """PRESPEC_E2 section 5 on the 95 % TEST interval of P1 - PT_flex: lower > 0 reversal, upper < 0 gain_remains,
    otherwise (0 inside or on a bound) undetermined."""

    low, high = (float(v) for v in interval)
    if not (math.isfinite(low) and math.isfinite(high)) or low > high:
        raise ValueError(f"not an interval: {low}, {high}")
    if low > 0.0:
        return "reversal"
    if high < 0.0:
        return "gain_remains"
    return "undetermined"


def rules() -> dict[str, Any]:
    """The rules every stage applies; the lock binds them and the TEST step refuses others."""

    return {"radii": list(RADII), "radius_note": RADIUS_NOTE, "strata": list(STRATA), "contrast": CONTRAST_TEXT,
            "weights": "within a stratum each match with rows there weighs equally; its rows are averaged "
                       "(scale_headcount.py:177-205)",
            "bootstrap": {"resamples": RESAMPLES, "seed": SEED, "level": LEVEL, "quantiles": list(qeval.bootstrap_quantiles(LEVEL)),
                          "unit": "match; per stratum over the matches with rows in it, ascending match id; the draws are "
                                  "shared by both cells and by the AUC difference",
                          "generator": "qeval.bootstrap_indices: numpy.random.Generator(numpy.random.PCG64(seed))"
                                       ".integers(0, M, size=M) per draw",
                          "interval": "percentile; numpy.quantile method='linear'"},
            "auc": "secondary: match-weighted AUC of both cells (qeval) and the difference P1 - PT_flex on the same draws",
            "probabilities": "raw p of the stored predictions (no calibration, no retraining)",
            "decision": {"split": "TEST only", "radius": DECISION_RADIUS, "stratum": DECISION_STRATUM, "outcomes": DECISIONS,
                         "rule": "95% interval of the TEST difference: lower > 0 -> reversal; upper < 0 -> gain_remains; "
                                 "otherwise undetermined"},
            "cells": {cell: {"final": final, "q_dev_predictions_from": search} for cell, (final, search) in CELLS.items()}}


def qdev_stage(paths: Paths = Paths()) -> int:
    """Q_DEV per radius and stratum (module docstring, stage qdev); writes E2_QDEV.json once."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / QDEV_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    build = read_json(paths.out / BUILD_NAME)
    if not (build and build.get("passed")):
        raise RuntimeError("the groups stage has not passed")
    q = _load_fields(paths.q_rows, ("engagement_key", "match_index", "role", "y"))
    dev = np.flatnonzero(np.asarray(q["role"]) == ROLE_CODES["Q_DEV"])
    y = np.asarray(q["y"])[dev].astype(np.float64)
    if not np.isin(y, (0.0, 1.0)).all():
        raise RuntimeError("Q_DEV labels are not binary")
    keys, match = np.asarray(q["engagement_key"])[dev], np.asarray(q["match_index"])[dev]
    p, inputs = {}, [paths.q_rows]
    for cell, (final_key, search_key) in CELLS.items():
        reuse_path, pred_path = paths.jobs / final_key / "REUSE.json", paths.jobs / search_key / "pred_q_dev.npz"
        if (read_json(reuse_path) or {}).get("reuse_of_key") != search_key:
            raise RuntimeError(f"{final_key}: REUSE.json does not point to {search_key}")
        pred = load_npz(pred_path)
        if not np.array_equal(np.asarray(pred["row"], dtype=np.int64), dev):
            raise RuntimeError(f"{search_key}: stored Q_DEV predictions are not the Q_DEV rows of Q_ROWS_h90 in order")
        p[cell] = np.asarray(pred["p"], dtype=np.float64)
        inputs += [reuse_path, pred_path]
    results = {"ALL": evaluate_pair(y, p["P1"], p["PT_flex"], match)}
    for radius in RADII:
        table = _checked_table(paths, build, radius, "Q_DEV", keys)
        results[f"r{int(radius)}"] = evaluate_strata(y, p["P1"], p["PT_flex"], match, table["stratum"])
        inputs.append(paths.table(radius, "Q_DEV", "npz"))
    record = run_record(rows=int(dev.size), matches=int(np.unique(match).size), inputs=inputs,
                        code=[ROOT / name for name in CODE_FILES], seed=SEED, started_utc=started)
    payload = {"version": VERSION, "experiment": "E2", "split": "Q_DEV", "policy": QDEV_POLICY, "contrast": CONTRAST_TEXT,
               "rules": rules(), "rows": int(dev.size), "matches": int(np.unique(match).size), "results": results,
               "stage": _stage("qdev", started, tick), "run": record}
    write_once(target, _jsonable(payload))
    _print_results("Q_DEV", results)
    return 0


# ---------------------------------------------------------------- lock and the single TEST step

def required_labels() -> list[str]:
    labels = ["prespec", "input:q_rows", "test:sealed", "groups:build", "qdev", *(f"code:{name}" for name in CODE_FILES)]
    labels += [f"test:{cell}:{kind}" for cell in CELLS for kind in ("npz", "json")]
    labels += [f"groups:r{int(r)}:{name}:{kind}" for r in RADII for name in ROLES for kind in ("npz", "json")]
    return labels


def lock_stage(paths: Paths = Paths(), *, require_committed_code: bool = True) -> int:
    """E2_LOCK.json (PRESPEC_E2 section 6).  TEST files are hashed as bytes only; nothing in them is opened."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / LOCK_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; it is written once")
    build, qdev = read_json(paths.out / BUILD_NAME), read_json(paths.out / QDEV_NAME)
    if not (build and build.get("passed")) or qdev is None:
        raise RuntimeError("run the groups and qdev stages first")
    prespec = document_record(paths.prespec)
    if not (prespec["sha256"] and prespec["commit"] and prespec["unchanged_since_commit"]):
        return _blocked(paths.out, "lock", {"prespec_hash_or_commit_missing": prespec}, stage=_stage("lock", started, tick))
    code = {name: document_record(ROOT / name) for name in CODE_FILES}
    current = {name: entry["sha256"] for name, entry in code.items()}
    for stage_name, record in (("groups", build), ("qdev", qdev)):
        if record["run"]["code_sha256"] != current:
            raise RuntimeError(f"the code changed after the {stage_name} stage; the lock must bind the code that ran")
    if require_committed_code and not all(entry["commit"] and entry["unchanged_since_commit"] for entry in code.values()):
        raise RuntimeError("commit the E2 code before writing the lock")

    verify: dict[str, dict[str, str]] = {}

    def add(label: str, path: Path) -> str:
        verify[label] = {"path": str(Path(path)), "sha256": sha256_file(path)}
        return verify[label]["sha256"]

    add("prespec", paths.prespec)       # EXPOSURE_KO.md is recorded (exposure_ledger) but not bound: it is appended to
    for name in CODE_FILES:
        add(f"code:{name}", ROOT / name)
    q_sha = add("input:q_rows", paths.q_rows)
    add("input:q_rows_manifest", paths.q_rows.with_suffix(".json"))
    add("input:xy", paths.xy)
    add("input:bank_field_manifest", paths.bank / "FIELD_MANIFEST.json")
    add("input:bank_run_manifest", paths.bank / "RUN_MANIFEST.json")
    add("input:engagements_manifest", paths.engagements_manifest)
    for cell, (final_key, search_key) in CELLS.items():
        add(f"qdev:{cell}:REUSE.json", paths.jobs / final_key / "REUSE.json")
        for name in ("SPEC.json", "RESULT.json", "model_refit.pkl", "pred_q_dev.npz"):
            add(f"qdev:{cell}:{name}", paths.jobs / search_key / name)
        add(f"test:{cell}:npz", paths.scores / f"{final_key}.npz")
        add(f"test:{cell}:json", paths.scores / f"{final_key}.json")
    sealed_sha = add("test:sealed", paths.sealed)
    add("test:score_lock", paths.q_run / "SCORE_LOCK.json")
    add("groups:build", paths.out / BUILD_NAME)
    add("qdev", paths.out / QDEV_NAME)
    strata_tables: dict[str, dict[str, Any]] = {}
    for radius in RADII:
        for name in ROLES:
            label = f"r{int(radius)}:{name}"
            npz_sha = add(f"groups:{label}:npz", paths.table(radius, name, "npz"))
            json_sha = add(f"groups:{label}:json", paths.table(radius, name, "json"))
            if (npz_sha, json_sha) != (build["tables"][label]["npz_sha256"], build["tables"][label]["json_sha256"]):
                raise RuntimeError(f"{label}: the group table changed after the groups stage")
            doc = read_json(paths.table(radius, name, "json"))
            strata_tables.setdefault(f"r{int(radius)}", {})[name] = {
                "rows": doc["rows"], "matches": doc["matches"], "npz_sha256": npz_sha, "json_sha256": json_sha,
                "strata": {s: {"rows": v["rows"], "matches": v["matches"]} for s, v in doc["strata"].items()}}
    q_manifest, score_lock = read_json(paths.q_rows.with_suffix(".json")), read_json(paths.q_run / "SCORE_LOCK.json")
    cross = {"q_rows_sha256_equals_its_manifest": q_sha == q_manifest.get("rows_sha256"),
             "q_rows_sha256_equals_the_groups_stage_input": q_sha == build["run"]["input_sha256"].get(paths.q_rows.as_posix()),
             "sealed_sha256_equals_q_rows_manifest": sealed_sha == q_manifest.get("test_sealed_sha256"),
             "sealed_sha256_equals_q_run_score_lock": sealed_sha == (score_lock.get("files") or {}).get(paths.sealed.name)}
    if not all(cross.values()):
        raise RuntimeError(f"input cross-checks failed: {cross}")
    payload = {
        "version": VERSION, "experiment": "E2", "locked_utc": utc(),
        "prespec": prespec, "exposure_ledger": document_record(paths.exposure), "code": code, "git_head": git_head(),
        "selected_cells": {cell: {"final": final_key, "reuses_search_job": search_key,
                                  "q_dev_predictions": str(paths.jobs / search_key / "pred_q_dev.npz"),
                                  "test_scores": str(paths.scores / f"{final_key}.npz")}
                           for cell, (final_key, search_key) in CELLS.items()},
        "test_files_policy": ("the two TEST_SCORES cells, their metadata and the sealed TEST label file are hashed as file "
                              "bytes only; nothing in them is opened before the test stage"),
        "rules": rules(), "strata_tables": strata_tables, "row_counts": build["rows_by_role"],
        "match_counts": build["matches_by_role"], "qdev": {"path": str(paths.out / QDEV_NAME), "sha256": verify["qdev"]["sha256"]},
        "cross_checks": cross,
        "bank": {k: build["bank"][k] for k in ("path", "shards", "shard_sha256_digest", "field_manifest_sha256",
                                               "run_manifest_sha256", "shards_equal_label_production_record")},
        "verify": verify,
        "stage": _stage("lock", started, tick),
        "run": run_record(rows=build["rows_by_role"], matches=build["matches_by_role"],
                          inputs=[paths.out / BUILD_NAME, paths.out / QDEV_NAME], code=[ROOT / name for name in CODE_FILES],
                          seed=SEED, started_utc=started),
    }
    write_once(target, _jsonable(payload))
    print(f"lock written: {target} ({len(verify)} hashed files; PRESPEC commit {prespec['commit']})")
    return 0


def verify_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Why the TEST step may not run (empty when the lock holds): no PRESPEC hash (PRESPEC_E2 section 7), a missing
    entry, a file changed since the lock, or rules other than the locked ones."""

    problems: dict[str, Any] = {}
    prespec, verify = lock.get("prespec") or {}, lock.get("verify") or {}
    if not (prespec.get("sha256") and prespec.get("commit")):
        problems["prespec_hash_missing"] = True
    elif (verify.get("prespec") or {}).get("sha256") != prespec["sha256"]:
        problems["prespec_entry_differs"] = True
    missing = [label for label in required_labels() if label not in verify]
    if missing:
        problems["entries_missing"] = missing
    changed = {}
    for label, entry in verify.items():
        path = Path(entry["path"])
        now = sha256_file(path) if path.is_file() else None
        if now != entry["sha256"]:
            changed[label] = {"path": entry["path"], "locked": entry["sha256"], "now": now}
    if changed:
        problems["changed_since_lock"] = changed
    if lock.get("rules") != _jsonable(rules()):
        problems["rules_differ"] = True
    return problems


def append_opening(path: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append one opening to OPENINGS.json (created when missing; EXPOSURE_KO.md rule 1); returns the earlier ones."""

    path = Path(path)
    payload = read_json(path) if path.exists() else {
        "rule": "EXPOSURE_KO.md rule 1: every ToG experiment that opens sealed labels appends experiment, UTC time and "
                "lock SHA-256", "openings": []}
    before = list(payload.get("openings", []))
    if any(o.get("experiment") == entry["experiment"] for o in before):
        raise RuntimeError(f"{entry['experiment']} has already opened its sealed labels")
    payload["openings"] = before + [dict(entry)]
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    pending.write_text(json.dumps(_jsonable(payload), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(pending, path)
    return before


def test_stage(paths: Paths = Paths()) -> int:
    """The single TEST step, run only after the review of the lock (Ruling 30); writes E2_REPORT.json once."""

    started, tick = utc(), time.perf_counter()
    target = paths.out / REPORT_NAME
    if target.exists():
        raise FileExistsError(f"{target} exists; TEST is evaluated once")
    lock_path = paths.out / LOCK_NAME
    lock = read_json(lock_path)
    if lock is None:
        raise RuntimeError("no E2_LOCK.json: run the lock stage first")
    problems = verify_lock(lock)
    if problems:
        return _blocked(paths.out, "test", problems, lock=str(lock_path))
    verify = lock["verify"]

    def path_of(label: str) -> Path:
        return Path(verify[label]["path"])

    q = _load_fields(path_of("input:q_rows"), ("engagement_key", "match_index", "role"))
    rows = np.flatnonzero(np.asarray(q["role"]) == ROLE_CODES["TEST"])
    keys, match = np.asarray(q["engagement_key"])[rows], np.asarray(q["match_index"])[rows]
    p = {}
    for cell in CELLS:
        meta = read_json(path_of(f"test:{cell}:json")) or {}
        if meta.get("pred_sha256") != verify[f"test:{cell}:npz"]["sha256"] or int(meta.get("rows", -1)) != rows.size:
            raise RuntimeError(f"{cell}: the TEST_SCORES metadata does not describe the locked scores")
        scores = _load_fields(path_of(f"test:{cell}:npz"), ("row", "engagement_key", "p"))
        if not (np.array_equal(np.asarray(scores["row"], dtype=np.int64), rows) and np.array_equal(scores["engagement_key"], keys)):
            raise RuntimeError(f"{cell}: the TEST scores are not the TEST rows of Q_ROWS_h90 in order")
        p[cell] = np.asarray(scores["p"], dtype=np.float64)
    strata = {}
    for radius in RADII:
        table = _load_fields(path_of(f"groups:r{int(radius)}:TEST:npz"), ("engagement_key", "stratum"))
        if not np.array_equal(table["engagement_key"], keys):
            raise RuntimeError(f"r{int(radius)}: the TEST group table is not in Q_ROWS_h90 order")
        strata[radius] = table["stratum"]

    # The opening is recorded before the sealed file is read (EXPOSURE_KO.md rule 1).
    entry = {"experiment": "E2", "utc": utc(), "lock": str(lock_path), "lock_sha256": sha256_file(lock_path),
             "prespec_sha256": lock["prespec"]["sha256"], "labels": [verify["test:sealed"]["path"]]}
    before = append_opening(paths.openings, entry)
    sealed = _load_fields(path_of("test:sealed"), ("engagement_key", "y"))
    index = {k: i for i, k in enumerate(sealed["engagement_key"].tolist())}
    absent = sum(k not in index for k in keys.tolist())
    if absent:
        raise RuntimeError(f"{absent} TEST rows have no sealed label")
    y = np.asarray(sealed["y"], dtype=np.float64)[np.asarray([index[k] for k in keys.tolist()], dtype=np.int64)]
    if not np.isin(y, (0.0, 1.0)).all():
        raise RuntimeError("the sealed TEST labels are not binary")
    results = {"ALL": evaluate_pair(y, p["P1"], p["PT_flex"], match)}
    for radius in RADII:
        results[f"r{int(radius)}"] = evaluate_strata(y, p["P1"], p["PT_flex"], match, strata[radius])
    deciding = results[f"r{int(DECISION_RADIUS)}"][DECISION_STRATUM]
    difference = deciding.get("brier_difference") or {}
    outcome = decide(difference["interval"]) if difference.get("interval") is not None else None
    decision = {"radius": DECISION_RADIUS, "stratum": DECISION_STRATUM, "rows": deciding["rows"], "matches": deciding["matches"],
                "point": difference.get("point"), "interval": difference.get("interval"), "level": LEVEL, "outcome": outcome,
                "text": DECISIONS[outcome] if outcome else "the decision stratum has no TEST rows: no decision",
                "rule": "PRESPEC_E2 section 5 (TEST interval only)"}
    counts: dict[str, dict[str, Any]] = {}
    for radius in RADII:
        for name in ROLES:
            doc = read_json(path_of(f"groups:r{int(radius)}:{name}:json"))
            counts.setdefault(f"r{int(radius)}", {})[name] = {
                "rows": doc["rows"], "matches": doc["matches"],
                "strata": {s: {"rows": v["rows"], "matches": v["matches"]} for s, v in doc["strata"].items()}}
    stages = {"groups": read_json(path_of("groups:build"))["stage"], "qdev": read_json(path_of("qdev"))["stage"],
              "lock": lock["stage"], "test": _stage("test", started, tick)}
    inputs = [path_of(label) for label in ("input:q_rows", "test:P1:npz", "test:PT_flex:npz", "test:sealed")]
    inputs += [path_of(f"groups:r{int(r)}:TEST:npz") for r in RADII] + [lock_path]
    report = {"version": VERSION, "experiment": "E2", "split": "TEST", "utc": utc(),
              "lock": {"path": str(lock_path), "sha256": entry["lock_sha256"], "locked_utc": lock["locked_utc"]},
              "openings_before": before, "opening": entry, "labels_opened": entry["labels"], "contrast": CONTRAST_TEXT,
              "rules": rules(), "rows": int(rows.size), "matches": int(np.unique(match).size), "results": results,
              "decision": decision, "counts": counts, "qdev": {"path": str(path_of("qdev")), "sha256": verify["qdev"]["sha256"]},
              "stages": stages,
              "run": run_record(rows=int(rows.size), matches=int(np.unique(match).size), inputs=inputs,
                                code=[ROOT / name for name in CODE_FILES], seed=SEED, started_utc=started)}
    write_once(target, _jsonable(report))
    _print_results("TEST", results)
    print("decision:", outcome, decision["interval"])
    return 0


# ---------------------------------------------------------------- helpers

def _load_fields(path: Path, fields: Iterable[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in fields}


def _checked_table(paths: Paths, build: Mapping[str, Any], radius: float, role: str, keys: np.ndarray) -> dict[str, np.ndarray]:
    path = paths.table(radius, role, "npz")
    if sha256_file(path) != build["tables"][f"r{int(radius)}:{role}"]["npz_sha256"]:
        raise RuntimeError(f"{path.name} changed after the groups stage")
    table = _load_fields(path, ("engagement_key", "stratum"))
    if not np.array_equal(table["engagement_key"], keys):
        raise RuntimeError(f"{path.name}: keys differ from Q_ROWS_h90")
    return table


def _digest(hashes: Mapping[str, str]) -> str:
    return hashlib.sha256("".join(f"{k} {v}\n" for k, v in sorted(hashes.items())).encode("ascii")).hexdigest()


def _count(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return len(value)
    return int(value)


def _stage(name: str, started: str, tick: float, **extra: Any) -> dict[str, Any]:
    return {"stage": name, "started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
            "peak_memory_bytes_main_process": peak_memory_bytes(), **extra}


def _blocked(out: Path, which: str, reasons: Mapping[str, Any], **extra: Any) -> int:
    """Write E2_BLOCKED_<which>_<n>.json once (n = the first free number, so earlier attempts stay) and return 2."""

    out.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out / f"E2_BLOCKED_{which}_{k}.json").exists():
        k += 1
    path = out / f"E2_BLOCKED_{which}_{k}.json"
    write_once(path, _jsonable({"version": VERSION, "experiment": "E2", "blocked_stage": which, "blocked": True,
                                "utc": utc(), "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": which, "file": str(path), "reasons": sorted(reasons)}))
    return 2


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default))


def _print_results(split: str, results: Mapping[str, Any]) -> None:
    def line(label: str, entry: Mapping[str, Any]) -> str:
        if not entry.get("rows"):
            return f"{split} {label}: rows=0"
        d = entry["brier_difference"]
        return (f"{split} {label}: rows={entry['rows']} matches={entry['matches']} P1-PT_flex={d['point']:.5f} "
                f"[{d['interval'][0]:.5f}, {d['interval'][1]:.5f}] auc P1={entry['auc']['P1']} PT_flex={entry['auc']['PT_flex']}")

    print(line("ALL", results["ALL"]))
    for radius in RADII:
        for name in STRATA:
            print(line(f"r{int(radius)} {name}", results[f"r{int(radius)}"][name]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("stage", choices=("groups", "qdev", "lock", "test"))
    parser.add_argument("--workers", type=int, default=3, help="groups: worker processes (main + workers <= 4)")
    parser.add_argument("--smoke-shards", type=int, default=0, help="groups: only the first N shards; nothing is written")
    parser.add_argument("--confirm-single-opening", action="store_true",
                        help="test: required; the step opens the sealed TEST labels once")
    args = parser.parse_args(argv)
    if args.stage == "groups":
        return groups_stage(workers=args.workers, smoke_shards=args.smoke_shards)
    if args.stage == "qdev":
        return qdev_stage()
    if args.stage == "lock":
        return lock_stage()
    if not args.confirm_single_opening:
        parser.error("the test stage opens the sealed TEST labels once; pass --confirm-single-opening")
    return test_stage()


if __name__ == "__main__":
    raise SystemExit(main())
