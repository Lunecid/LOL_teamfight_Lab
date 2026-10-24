"""T4.1 E4 scoring: the per-context LightGBM V (the second measurer) on both queries of every q engagement.

Binding specification: reports/tog_experiments_20261012/PRESPEC_E4_KO.md (sections 2, 3, 6, 7; PLAN_E1_E5_KO.md
Task 10; SDD Ruling 19).  Stages, in PRESPEC_E4 section 6 order (step 1, then step 2); every result is written once
under D:/LOL_Training_20261004/tog_e4_1:

  measurers  Per context, the LightGBM candidate of the V search rule (lowest raw V_DEV log loss, then card order:
             train.select_v_dev within one family) in run_3/<ctx>/SELECTION.json must be PRESPEC_E4's card and value
             (section 2); its candidate_result.json and REFIT_SCORED.json must name that context and card, the pickle's
             SHA-256 must equal REFIT_SCORED.json's, and it must have consumed exactly the FIT, STOP, REFIT and V_DEV
             rows of the context's main V (row-key and game-key hashes) with the context's excluded OOF games, so the
             context membership bits below describe what it saw.
  vdev       Step 1.  Per context, the V_DEV rows of its REFIT cache (membership bit 8; the Blue view, never swapped)
             must equal candidate_result.json's V_DEV consumption record (frozen train._Ledger: rows, batches, target
             sum, row-key and game-key hashes).  They are scored by the same path as the engagement queries, and the
             raw V_DEV log loss (train.raw_logloss_from_probability) must equal candidate_result.json's and PRESPEC_E4's
             value at the fourth decimal in all six contexts (section 7), else BLOCKED and nothing else is scored.
  rows       The q rows are q_run_1/Q_ROWS_h90.npz's rows (identity columns only; no label column is read), aligned one
             to one with the locked labels_1/ENGAGEMENTS.npz (role TRAIN/Q_DEV/Q_CAL/TEST and valid_h90), which gives
             each row its h90 endpoint query.  Scoring context (section 3): a TRAIN row the oof context of its fold,
             a Q_DEV/Q_CAL/TEST row final.  OOF exclusion per row with the match-membership bits as for the main labels
             (assemble.py:109-114): the scoring context never saw the game in fit/stop/refit/dev/cal (bits 1-16) and a
             TRAIN game is its held-out target (bit 32).  One failure, or a row-table key mismatch, is BLOCKED.
  score      Step 2.  Each context's sorted query list in fixed parts (32,768 rows) of canonical 1,024-row batches:
             QuerySurface.iter_query_rows -> queries.transform (the context's REFIT_TRAIN bundle) ->
             queries.family_inputs (lightgbm) -> the pickle's predict_proba: the raw probability, no calibrator.  A part
             is written once with its receipt and a rerun continues from the parts that verify (chunk-level restart).
             At most 4 worker processes, CPU only, LightGBM held to one thread per process.
  merge      PRED_<ctx>.npz (query_index, raw_p) and PRED_<ctx>.json (row counts by role, input and code hashes,
             versions, UTC, stage times, peak memory), then E4_VSCORE.json (the V_DEV table of six contexts, the OOF
             result).

TEST rows' raw probabilities fix the new TEST labels and lie outside any seal (PRESPEC_E4 section 3): nothing here
summarizes them; the only checks on them are validity checks (finite, inside [0, 1]) that stop the run.  No player
identifier is read or written.  A BLOCKED condition writes E4_VSCORE_BLOCKED_<stage>_<n>.json and exits with 2.

Usage: python -B -m studies.tog_experiments_20261012.e4_vscore run [--workers 4] [--parts-per-job 2]
       (it starts its own workers: ... e4_vscore worker --kind vdev|q --context <ctx> --parts ...)
       Smoke check into another folder: ... e4_vscore run --out <scratch> --smoke 2048 --rows-per-part 1024
"""
from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse  # noqa: E402
from dataclasses import dataclass  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path, PureWindowsPath  # noqa: E402
import pickle  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import Any, Mapping, Sequence  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.label_production_20261005 import queries  # noqa: E402
from studies.label_production_20261005.assemble import CONTEXTS as MEMBERSHIP_ROWS, PROJECTION, Q_FINAL_ROLES  # noqa: E402
from studies.q_production_20261005 import rows as q_rows_module  # noqa: E402
from studies.q_production_20261005.rows import Q_ROLES  # noqa: E402
from studies.tog_experiments_20261012.common import (  # noqa: E402
    LABELS, Q_RUN, TRAINING, document_record, git_head, peak_memory_bytes, sha256_file, utc, write_npz_once, write_once)
from studies.v_model_fit_input_20261003.models import family_cards  # noqa: E402
from studies.v_production_20261004.source import MEMBERSHIP_BITS, SPLIT_MASK  # noqa: E402
from studies.v_production_20261004.train import _Ledger, raw_logloss_from_probability  # noqa: E402

VERSION = "tog_e4_vscore_v1_20261012"
PART_VERSION = "tog_e4_vscore_part_v1"   # the scoring path of a part; a part made by another path is never reused
RUN_3 = TRAINING / "run_3"
TOG_E4 = TRAINING / "tog_e4_1"
PRESPEC = ROOT / "reports/tog_experiments_20261012/PRESPEC_E4_KO.md"
CONTEXTS = tuple(queries.CONTEXTS)       # final, oof0..oof4: code = index (CONTEXT_MEMBERSHIP row order, assemble.py:36)
FAMILY = "lightgbm"
MEASURERS = {"final": ("lgbm_l63_m500", 0.487389), "oof0": ("lgbm_l63_m500", 0.488139),   # PRESPEC_E4 section 2
             "oof1": ("lgbm_l63_m500", 0.487628), "oof2": ("lgbm_l63_m500", 0.488013),
             "oof3": ("lgbm_l63_m200", 0.488381), "oof4": ("lgbm_l63_m200", 0.488009)}
Q_ROLE_ROWS = {"TRAIN": 268_101, "Q_DEV": 107_970, "Q_CAL": 54_061, "TEST": 222_154}       # PRESPEC_E4 section 2
DEV_BIT, TARGET_BIT = MEMBERSHIP_BITS["dev"], 32                  # source.py:30-31; bit 32 = the OOF target fold
DECIMALS = 4
PRINTED_DECIMALS = 6
BATCH_ROWS = queries.BATCH_ROWS                                    # 1,024 (canonical scoring batches, queries.py:39)
ROWS_PER_PART = 32 * BATCH_ROWS                                    # 32,768, as score.py:44 and :148-150
MAX_WORKERS = 4
KINDS = {"vdev": "VDEV", "q": "PRED"}
SPLITS = ("fit", "stop", "refit", "dev")
CONSUMED_VIEW = ("rows", "keys_sha256", "game_keys_sha256")
LEDGER_FIELDS = ("rows", "batches", "target_sum", "key_count", "keys_sha256", "game_keys_sha256", "roles")
IDENTITY_FIELDS = ("kind", "context", "card_id", "model_sha256")
ALIGN_COLUMNS = ("canonical_match_key", "component_key", "match_index", "role", "fold", "query_pre")
CACHE_FIELDS = ("context_membership", "role", "swapped", "query_index", "nominal_index", "label_w", "game_key")
PREDICT = "LGBMClassifier.predict_proba(table_csr, num_threads=1)[:, 1] (= the pickle's predict_proba, models.py:939-942)"
CODE_FILES = ("studies/tog_experiments_20261012/e4_vscore.py", "studies/tog_experiments_20261012/common.py",
              "studies/label_production_20261005/queries.py", "studies/label_production_20261005/assemble.py",
              "studies/v_full_preprocess_20261003/surface.py", "studies/v_full_preprocess_20261003/artifacts.py",
              "studies/v_model_fit_input_20261003/common.py", "studies/v_model_input_20261003/view.py",
              "studies/v_model_fit_input_20261003/models.py", "studies/v_production_20261004/train.py",
              "studies/v_production_20261004/source.py", "studies/q_production_20261005/rows.py")
SURFACE_FILES = (PROJECTION / "RUN_MANIFEST.json", PROJECTION / "BANK_METADATA.json", PROJECTION / "MATCH_TABLE.npy",
                 PROJECTION / "QUERY_TABLE.npy", PROJECTION / "MATCH_QUERIES.npy", PROJECTION / "CONTEXT_MEMBERSHIP.npy",
                 queries.SHARED_BANK / "PLAN.json", queries.SHARED_BANK / "ACCEPTANCE.json")
ROW_FILES = (Q_RUN / "Q_ROWS_h90.npz", LABELS / "ENGAGEMENTS.npz", LABELS / "LABEL_LOCK.json")

if tuple(MEMBERSHIP_ROWS) != CONTEXTS:
    raise RuntimeError("the context codes differ from the CONTEXT_MEMBERSHIP row order")


# ----------------------------------------------------------------------------- measurer choice and binding
def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def select_measurer(candidates: Sequence[Mapping[str, Any]], family: str = FAMILY) -> Mapping[str, Any]:
    """The family winner of train.select_v_dev (train.py:1830-1845): among completed candidates with a finite raw
    V_DEV log loss, the lowest loss, then the card order of models.family_cards, then the seed."""

    within = [c for c in candidates if c.get("family") == family and c.get("status") == "completed"
              and _finite((c.get("dev_metrics") or {}).get("raw_logloss"))]
    if not within:
        raise ValueError(f"no completed {family} candidate with a finite V_DEV log loss")
    order = {str(card["id"]): index for index, card in enumerate(family_cards(family))}
    return min(within, key=lambda c: (float(c["dev_metrics"]["raw_logloss"]), order.get(str(c["card_id"]), len(order)),
                                      int(c.get("seed", 0))))


def _view(consumed: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    return None if consumed is None else tuple(consumed.get(name) for name in CONSUMED_VIEW)


def split_consumed(record: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    """A candidate's consumption record of one split.  Table families keep one (``<split>_metrics.consumed``); neural
    families keep one per epoch (``fit_metrics.epochs[*].fit_consumed``, ``refit_metrics.epochs[*].consumed``), which
    must all be equal.  None if absent or inconsistent."""

    metrics = record.get(f"{split}_metrics") or {}
    if isinstance(metrics.get("consumed"), Mapping):
        return metrics["consumed"]
    key = "fit_consumed" if split == "fit" else "consumed"
    views = [epoch.get(key) for epoch in metrics.get("epochs") or [] if isinstance(epoch, Mapping)]
    if views and all(isinstance(v, Mapping) for v in views) and len({_view(v) for v in views}) == 1:
        return views[0]
    return None


def measurer_problems(ctx: str, selection: Mapping[str, Any], result: Mapping[str, Any], scored: Mapping[str, Any],
                      plan: Mapping[str, Any], excluded: Mapping[str, Any],
                      prespec: Mapping[str, tuple[str, float]] = MEASURERS) -> tuple[dict[str, Any], list[str]]:
    """Identity of one context's measurer and every disagreement with PRESPEC_E4 and the run_3 records.

    ``result`` is the chosen card's candidate_result.json, ``scored`` its REFIT_SCORED.json, ``plan`` the context's
    CONTEXT_PLAN.json and ``excluded`` its EXCLUDED_GAME_KEYS.json."""

    problems: list[str] = []
    card, printed = prespec[ctx]
    chosen = select_measurer(selection.get("candidates") or [])
    loss = float(chosen["dev_metrics"]["raw_logloss"])
    if chosen["card_id"] != card:
        problems.append(f"{ctx}: the V search rule picks {chosen['card_id']}, PRESPEC_E4 names {card}")
    if round(loss, PRINTED_DECIMALS) != printed:
        problems.append(f"{ctx}: SELECTION.json V_DEV raw log loss {loss!r} is not PRESPEC_E4's {printed}")
    for name, value in (("context", ctx), ("family", FAMILY), ("card_id", chosen["card_id"]), ("status", "completed")):
        if result.get(name) != value:
            problems.append(f"{ctx}: candidate_result.json {name} is {result.get(name)!r}, not {value!r}")
    if (result.get("dev_metrics") or {}).get("raw_logloss") != chosen["dev_metrics"]["raw_logloss"]:
        problems.append(f"{ctx}: candidate_result.json and SELECTION.json V_DEV log losses differ")
    if PureWindowsPath(str(result.get("checkpoint", ""))).parts[-2:] != (chosen["card_id"], "model_refit.pkl"):
        problems.append(f"{ctx}: candidate_result.json does not point to the card's model_refit.pkl")
    identity = scored.get("identity") or {}
    for name, value in (("context", ctx), ("family", FAMILY), ("card_id", chosen["card_id"])):
        if identity.get(name) != value:
            problems.append(f"{ctx}: REFIT_SCORED.json identity {name} is {identity.get(name)!r}, not {value!r}")
    if not (scored.get("files") or {}).get("model_refit.pkl"):
        problems.append(f"{ctx}: REFIT_SCORED.json records no model_refit.pkl hash")
    keys = list(excluded.get("excluded_game_keys") or [])
    excluded_sha = hashlib.sha256(json.dumps(sorted(keys)).encode("utf-8")).hexdigest()   # train.py:341
    if excluded.get("context") != ctx or identity.get("excluded_game_keys_sha256") != excluded_sha:
        problems.append(f"{ctx}: the measurer's excluded OOF games differ from EXCLUDED_GAME_KEYS.json")
    counts = plan.get("counts") or {}
    expected_rows = [counts.get(split) for split in SPLITS]
    if plan.get("context") != ctx or list(identity.get("expected_rows") or []) != expected_rows:
        problems.append(f"{ctx}: the measurer's FIT/STOP/REFIT/V_DEV row counts differ from CONTEXT_PLAN.json")
    main_v = selection.get("selected") or {}
    binding = {}
    for split in SPLITS:
        mine, theirs = split_consumed(result, split), split_consumed(main_v, split)
        binding[split] = mine is not None and theirs is not None and _view(mine) == _view(theirs)
        if not binding[split]:
            problems.append(f"{ctx}: the measurer's {split} rows differ from the main V's ({main_v.get('card_id')})")
    record = {"context": ctx, "family": FAMILY, "card_id": chosen["card_id"], "v_dev_raw_logloss": loss,
              "prespec": {"card_id": card, "v_dev_raw_logloss": printed},
              "lightgbm_candidates": {str(c["card_id"]): (c.get("dev_metrics") or {}).get("raw_logloss")
                                      for c in selection.get("candidates") or [] if c.get("family") == FAMILY},
              "main_v": {"family": main_v.get("family"), "card_id": main_v.get("card_id")},
              "binding": binding, "excluded_game_keys": len(keys), "expected_rows": dict(zip(SPLITS, expected_rows)),
              "model_sha256_recorded": (scored.get("files") or {}).get("model_refit.pkl")}
    return record, problems


@dataclass
class Measurer:
    """One context's LightGBM V, loaded from its verified pickle."""

    context: str
    card_id: str
    model: Any
    checkpoint: Path
    model_sha256: str
    thread_check: bool | None = None   # set once the first batch agrees with the pickle's own predict_proba


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def candidate_dir(ctx: str, card: str, run_root: Path = RUN_3) -> Path:
    return Path(run_root) / ctx / "candidates" / FAMILY / card


def load_measurer(ctx: str, run_root: Path = RUN_3) -> Measurer:
    """The V-search-rule LightGBM candidate of ``ctx`` (PRESPEC_E4's card), its pickle checked against REFIT_SCORED.json."""

    if ctx not in CONTEXTS:
        raise ValueError(f"unknown context {ctx!r}")
    card = str(select_measurer(_read(Path(run_root) / ctx / "SELECTION.json")["candidates"])["card_id"])
    if card != MEASURERS[ctx][0]:
        raise ValueError(f"{ctx}: the V search rule picks {card}, PRESPEC_E4 names {MEASURERS[ctx][0]}")
    folder = candidate_dir(ctx, card, run_root)
    checkpoint = folder / "model_refit.pkl"
    digest = sha256_file(checkpoint)
    if digest != _read(folder / "REFIT_SCORED.json")["files"]["model_refit.pkl"]:
        raise ValueError(f"{ctx}: {checkpoint} differs from its REFIT_SCORED.json hash")
    with checkpoint.open("rb") as stream:
        model = pickle.load(stream)
    if getattr(model, "family", None) != FAMILY or getattr(model, "estimator", None) is None:
        raise ValueError(f"{ctx}: {checkpoint} is not a LightGBM adapter")
    return Measurer(context=ctx, card_id=card, model=model, checkpoint=checkpoint, model_sha256=digest)


# ----------------------------------------------------------------------------- scoring
def _check_probabilities(values: np.ndarray, what: str) -> None:
    values = np.asarray(values)
    if values.dtype != np.float64 or values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError(f"{what}: probabilities must be finite float64 values inside [0, 1]")


def predict_raw(measurer: Measurer, inputs: Mapping[str, Any]) -> np.ndarray:
    """The pickle's predict_proba (LightGBMAdapter.predict_proba = LGBMClassifier.predict_proba[:, 1]) with LightGBM
    held to one thread.  The first batch of a process is also scored by the adapter itself (its stored thread count)
    and must agree bit for bit; LightGBM's per-row prediction does not depend on the thread count."""

    with warnings.catch_warnings():  # the estimator was fitted on a table with column names; the CSR batch has none
        warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
        raw = np.asarray(measurer.model.estimator.predict_proba(inputs["table_csr"], num_threads=1), dtype=np.float64)[:, 1]
        if measurer.thread_check is None:
            reference = np.asarray(measurer.model.predict_proba(inputs), dtype=np.float64).reshape(-1)
            if not np.array_equal(raw, reference):
                raise ValueError(f"{measurer.context}: one-thread predictions differ from the pickle's predict_proba")
            measurer.thread_check = True
    return raw


def score_context(ctx: str, query_indices: np.ndarray, *, measurer: Measurer | None = None, inputs: tuple[Any, Any] | None = None,
                  transform: Any = None) -> np.ndarray:
    """Raw probabilities (float64) of the context's LightGBM V for strictly increasing QUERY_TABLE indices, in
    canonical 1,024-row batches: QuerySurface.iter_query_rows -> queries.transform -> queries.family_inputs -> pickle
    predict_proba.  ``measurer``, ``inputs`` = (surface, bundle) and ``transform`` are loaded when not given."""

    if ctx not in CONTEXTS:
        raise ValueError(f"unknown context {ctx!r}")
    indices = np.asarray(query_indices, dtype=np.int64)
    measurer = load_measurer(ctx) if measurer is None else measurer
    if measurer.context != ctx:
        raise ValueError(f"the measurer of {measurer.context} cannot score {ctx}")
    transform = queries.transform if transform is None else transform
    own = inputs is None
    surface, bundle = queries.open_inputs(ctx) if own else inputs
    try:
        pieces = []
        for start in range(0, len(indices), BATCH_ROWS):
            batch = indices[start:start + BATCH_ROWS]
            rows, seen = [], []
            for query_index, row, _meta in surface.iter_query_rows(batch):
                rows.append(row)
                seen.append(query_index)
            if not np.array_equal(np.asarray(seen, dtype=np.int64), batch):
                raise ValueError(f"{ctx}: query rows came back out of order")
            pieces.append(predict_raw(measurer, queries.family_inputs(transform(rows, bundle), FAMILY)))
        raw = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
    finally:
        if own:
            surface.close()
    _check_probabilities(raw, f"{ctx} scores")
    return raw


# ----------------------------------------------------------------------------- rows, contexts, OOF exclusion
def align_q_rows(engagements: Mapping[str, np.ndarray], q_rows: Mapping[str, np.ndarray]) -> np.ndarray:
    """Mask of the engagement rows that are the row table's rows, in order: q role and valid at h90 (rows.py:113-116;
    every q-role row of the locked labels is valid otherwise).  Any difference in the identity columns raises."""

    keep = np.isin(np.asarray(engagements["role"]), list(Q_ROLES.values())) & np.asarray(engagements["valid_h90"], dtype=bool)
    for name in ALIGN_COLUMNS:
        left, right = np.asarray(engagements[name])[keep], np.asarray(q_rows[name])
        if left.dtype.kind in "iu":
            left, right = left.astype(np.int64), right.astype(np.int64)
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"the q row table differs from the engagement table in {name}")
    return keep


def row_contexts(role: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Scoring context code per row (index in CONTEXTS): final (0) for Q_DEV/Q_CAL/TEST, oof<fold> (1 + fold) for
    TRAIN, -1 otherwise (assemble.py:82-85)."""

    role, fold = np.asarray(role).astype(np.int64), np.asarray(fold).astype(np.int64)
    codes = np.full(len(role), -1, dtype=np.int8)
    codes[np.isin(role, Q_FINAL_ROLES)] = 0
    train = role == Q_ROLES["TRAIN"]
    if np.any(train & ((fold < 0) | (fold >= len(CONTEXTS) - 1))):
        raise ValueError("a TRAIN row has no OOF fold 0-4")
    codes[train] = 1 + fold[train]
    return codes


def context_query_list(codes: np.ndarray, code: int, query_pre: np.ndarray, query_post: np.ndarray) -> np.ndarray:
    """Sorted unique QUERY_TABLE indices one context scores: both ends of the rows it labels."""

    own = np.asarray(codes) == code
    values = np.concatenate([np.asarray(query_pre)[own], np.asarray(query_post)[own]]).astype(np.int64)
    if np.any(values < 0):
        raise ValueError("a q row lacks a query index")
    return np.unique(values)


def oof_leaks(codes: np.ndarray, role: np.ndarray, match_index: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Per row, whether the scoring context saw the game (assemble.py:109-114): a split bit 1-16 of its membership
    (fit/stop/refit/dev/cal), or a TRAIN row whose game is not the context's held-out target (bit 32)."""

    codes = np.asarray(codes).astype(np.int64)
    if np.any(codes < 0):
        raise ValueError("a q row has no scoring context")
    bits = np.asarray(membership, dtype=np.uint8)[codes, np.asarray(match_index).astype(np.int64)]
    leaked = (bits & SPLIT_MASK) != 0
    leaked |= (np.asarray(role) == Q_ROLES["TRAIN"]) & ((bits & TARGET_BIT) == 0)
    return leaked


# ----------------------------------------------------------------------------- V_DEV reproduction
def dev_rows(metadata: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The V_DEV rows of a REFIT cache in cache order (membership bit 8 selects them, source.py:228-245)."""

    dev = (np.asarray(metadata["context_membership"], dtype=np.uint8) & DEV_BIT) != 0
    role = np.asarray(metadata["role"])[dev]
    if not dev.any():
        raise ValueError("the cache has no V_DEV rows")
    if np.any(role != b"V_DEV"):
        raise ValueError("a V_DEV membership bit is attached to a non-V_DEV row")
    if np.any(np.asarray(metadata["swapped"], dtype=bool)[dev]):
        raise ValueError("a V_DEV row is swapped; V_DEV rows are the Blue view")
    query_index = np.asarray(metadata["query_index"], dtype=np.int64)[dev]
    nominal = np.asarray(metadata["nominal_index"], dtype=np.int64)[dev]
    if np.any(np.diff(query_index) <= 0) or np.any(np.diff(nominal) <= 0):
        raise ValueError("V_DEV rows are not strictly increasing")
    return {"query_index": query_index, "label": np.asarray(metadata["label_w"], dtype=bool)[dev].astype(np.int8),
            "nominal_index": nominal, "game_key": np.asarray(metadata["game_key"])[dev], "role": role}


def dev_consumed(rows: Mapping[str, np.ndarray], excluded_game_keys: Sequence[str] = ()) -> dict[str, Any]:
    """The record the frozen train._Ledger keeps for these rows, fed as 1,024-row batches in cache order as the V
    runner fed them (train.py:724-766, source.py:426-461)."""

    n = len(rows["query_index"])
    ledger = _Ledger(expected_rows=n, excluded_game_keys=excluded_game_keys, split="dev")
    for start in range(0, n, BATCH_ROWS):
        stop = min(start + BATCH_ROWS, n)
        metadata = {"game_keys": [bytes(v).decode("utf-8") for v in rows["game_key"][start:stop]],
                    "roles": [bytes(v).decode("utf-8") for v in rows["role"][start:stop]],
                    "nominal_indices": rows["nominal_index"][start:stop]}
        ledger.update(metadata, np.asarray(rows["label"][start:stop], dtype=np.float32))
    return ledger.finish()


def reproduces(value: float, reference: float, decimals: int = DECIMALS) -> bool:
    """Whether two log losses agree at the given decimal (both rounded to it)."""

    return _finite(value) and _finite(reference) and round(float(value), decimals) == round(float(reference), decimals)


def vdev_record(ctx: str, card: str, raw_p: np.ndarray, label: np.ndarray, consumed: Mapping[str, Any],
                result: Mapping[str, Any], printed: float) -> dict[str, Any]:
    """The V_DEV reproduction of one context: raw log loss of the rescored V_DEV rows against candidate_result.json's
    (and PRESPEC_E4's printed value) at the fourth decimal, and the V_DEV rows against its consumption record."""

    loss = raw_logloss_from_probability(raw_p, label)
    recorded = float(result["dev_metrics"]["raw_logloss"])
    expected = (result.get("dev_metrics") or {}).get("consumed") or {}
    consumed_equal = {name: consumed.get(name) == expected.get(name) for name in LEDGER_FIELDS}
    record = {"context": ctx, "card_id": card, "rows": int(len(raw_p)), "raw_logloss": loss,
              "candidate_result_raw_logloss": recorded, "prespec_printed": printed, "abs_diff": abs(loss - recorded),
              "raw_logloss_4dp": f"{loss:.{DECIMALS}f}",
              "reproduces_candidate_result_4dp": reproduces(loss, recorded), "reproduces_prespec_4dp": reproduces(loss, printed),
              "consumed": dict(consumed), "consumed_equal": consumed_equal}
    record["passed"] = bool(record["reproduces_candidate_result_4dp"] and record["reproduces_prespec_4dp"]
                            and all(consumed_equal.values()) and len(raw_p) == expected.get("rows"))
    return record


# ----------------------------------------------------------------------------- parts (written once, restartable)
def part_ranges(n_rows: int, rows_per_part: int = ROWS_PER_PART) -> list[tuple[int, int]]:
    return [(start, min(start + rows_per_part, n_rows)) for start in range(0, n_rows, rows_per_part)]


def part_path(directory: Path, part: int) -> Path:
    return Path(directory) / f"part_{part:05d}.npz"


def write_part(directory: Path, part: int, query_index: np.ndarray, raw_p: np.ndarray, identity: Mapping[str, Any],
               extra: Mapping[str, Any] | None = None) -> str:
    """Write one part (npz, then its receipt) and return its SHA-256.  A part with a receipt is written once; an npz
    left without a receipt by an interrupted run is replaced."""

    target = part_path(directory, part)
    receipt = target.with_suffix(".json")
    if receipt.exists():
        raise FileExistsError(f"{receipt} exists; a part is written once")
    query_index, raw_p = np.asarray(query_index, dtype=np.int64), np.asarray(raw_p, dtype=np.float64)
    if query_index.shape != raw_p.shape or query_index.ndim != 1:
        raise ValueError("query indices and probabilities must align")
    _check_probabilities(raw_p, f"part {part}")
    Path(directory).mkdir(parents=True, exist_ok=True)
    pending = target.with_name(f".{target.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.savez(stream, query_index=query_index, raw_p=raw_p)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, target)
    digest = sha256_file(target)
    write_once(receipt, {"version": PART_VERSION, **{name: identity[name] for name in IDENTITY_FIELDS}, "part": int(part),
                         "rows": int(len(query_index)), "sha256": digest, "utc": utc(), **dict(extra or {})})
    return digest


def part_status(directory: Path, part: int, expected: np.ndarray, identity: Mapping[str, Any]) -> str:
    """'done' (receipt, bytes, identity and query indices verify), 'missing', or 'incomplete' (an npz without its
    receipt); a part whose receipt does not verify raises, so another run's or a changed part is never reused."""

    target = part_path(directory, part)
    receipt = target.with_suffix(".json")
    if not receipt.is_file():
        return "incomplete" if target.exists() else "missing"
    if not target.is_file():
        raise ValueError(f"{receipt} has no part file")
    record = _read(receipt)
    problems = [f"{name} {record.get(name)!r} != {identity[name]!r}" for name in IDENTITY_FIELDS if record.get(name) != identity[name]]
    if record.get("version") != PART_VERSION:
        problems.append(f"version {record.get('version')!r}")
    if record.get("sha256") != sha256_file(target):
        problems.append("the part file differs from its receipt")
    else:
        with np.load(target, allow_pickle=False) as stored:
            if not np.array_equal(stored["query_index"], np.asarray(expected, dtype=np.int64)):
                problems.append("query indices differ from the query list")
            _check_probabilities(stored["raw_p"], str(target))
    if problems:
        raise ValueError(f"{target}: " + "; ".join(problems))
    return "done"


def merge_parts(directory: Path, ranges: Sequence[tuple[int, int]], query_index: np.ndarray, identity: Mapping[str, Any]) -> np.ndarray:
    """The raw probabilities of a whole query list from its verified parts (every part must be done)."""

    pieces = []
    for part, (start, stop) in enumerate(ranges):
        if part_status(directory, part, query_index[start:stop], identity) != "done":
            raise ValueError(f"{directory}: part {part} is not complete")
        with np.load(part_path(directory, part), allow_pickle=False) as stored:
            pieces.append(np.asarray(stored["raw_p"], dtype=np.float64))
    raw = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)
    if len(raw) != len(query_index):
        raise ValueError(f"{directory}: the parts do not cover the query list")
    return raw


def _receipts(directory: Path, n_parts: int) -> list[dict[str, Any]]:
    return [_read(part_path(directory, part).with_suffix(".json")) for part in range(n_parts)]


def _parts_summary(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    seconds = [float(r.get("seconds") or 0.0) for r in receipts]
    peaks = [int(r["peak_memory_bytes"]) for r in receipts if r.get("peak_memory_bytes")]
    return {"parts": len(receipts), "rows": int(sum(int(r["rows"]) for r in receipts)), "seconds_sum": float(sum(seconds)),
            "seconds_max": max(seconds, default=0.0), "worker_peak_memory_bytes_max": max(peaks, default=None),
            "part_sha256": [r["sha256"] for r in receipts], "code_sha256": sorted({r.get("code_sha256") for r in receipts}),
            "thread_check": sorted({str(r.get("thread_check")) for r in receipts}),
            "first_started_utc": min((r.get("started_utc") or "" for r in receipts), default=None),
            "last_utc": max((r.get("utc") or "" for r in receipts), default=None)}


# ----------------------------------------------------------------------------- worker process
def _identity(kind: str, measurer: Measurer) -> dict[str, Any]:
    return {"kind": kind, "context": measurer.context, "card_id": measurer.card_id, "model_sha256": measurer.model_sha256}


def _list_file(out: Path, kind: str, ctx: str) -> Path:
    return Path(out) / (f"VDEV_{ctx}_ROWS.npz" if kind == "vdev" else f"QUERIES_{ctx}.npy")


def _query_list(out: Path, kind: str, ctx: str) -> np.ndarray:
    path = _list_file(out, kind, ctx)
    if kind == "vdev":
        with np.load(path, allow_pickle=False) as stored:
            return np.asarray(stored["query_index"], dtype=np.int64)
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.int64)


def worker(kind: str, ctx: str, parts: Sequence[int], out: Path, rows_per_part: int) -> int:
    """Score the given parts of one (kind, context) query list; parts that already verify are skipped."""

    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    measurer = load_measurer(ctx)
    index = _query_list(out, kind, ctx)
    list_sha = sha256_file(_list_file(out, kind, ctx))
    ranges = part_ranges(len(index), rows_per_part)
    directory = Path(out) / f"{KINDS[kind]}_{ctx}"
    identity = _identity(kind, measurer)
    todo = [part for part in parts if part_status(directory, part, index[slice(*ranges[part])], identity) != "done"]
    if not todo:
        return 0
    code_sha = sha256_file(Path(__file__))
    surface, bundle = queries.open_inputs(ctx)
    try:
        for part in todo:
            started, tick = utc(), time.perf_counter()
            start, stop = ranges[part]
            raw = score_context(ctx, index[start:stop], measurer=measurer, inputs=(surface, bundle))
            write_part(directory, part, index[start:stop], raw, identity,
                       {"range": [int(start), int(stop)], "queries_file": _list_file(out, kind, ctx).name, "queries_sha256": list_sha,
                        "rows_per_part": int(rows_per_part), "batch_rows": BATCH_ROWS, "predict": PREDICT,
                        "thread_check": measurer.thread_check, "code_sha256": code_sha, "started_utc": started,
                        "seconds": time.perf_counter() - tick, "peak_memory_bytes": peak_memory_bytes(), "pid": os.getpid()})
            print(json.dumps({"kind": kind, "context": ctx, "part": part, "rows": int(stop - start),
                              "seconds": round(time.perf_counter() - tick, 1)}), flush=True)
    finally:
        surface.close()
    return 0


# ----------------------------------------------------------------------------- driver
def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _stage(name: str, started: str, tick: float, **extra: Any) -> dict[str, Any]:
    return {"stage": name, "started_utc": started, "finished_utc": utc(), "wall_seconds": time.perf_counter() - tick,
            "peak_memory_bytes_driver": peak_memory_bytes(), **extra}


def _blocked(out: Path, which: str, reasons: Sequence[str] | Mapping[str, Any], **extra: Any) -> int:
    """Write E4_VSCORE_BLOCKED_<which>_<n>.json once (n = the first free number) and return 2."""

    out.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out / f"E4_VSCORE_BLOCKED_{which}_{k}.json").exists():
        k += 1
    path = out / f"E4_VSCORE_BLOCKED_{which}_{k}.json"
    write_once(path, _json_safe({"version": VERSION, "experiment": "E4", "blocked_stage": which, "blocked": True, "utc": utc(),
                                 "reasons": reasons, **extra}))
    print(json.dumps({"BLOCKED": which, "file": str(path), "reasons": _json_safe(reasons)}, ensure_ascii=False), flush=True)
    return 2


def _hashes(paths: Sequence[Path], cache: dict[str, str]) -> dict[str, str]:
    out = {}
    for path in paths:
        key = Path(path).as_posix()
        if key not in cache:
            cache[key] = sha256_file(path)
        out[key] = cache[key]
    return out


def _versions() -> dict[str, str]:
    import lightgbm
    import scipy
    import sklearn

    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__}


def _code_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in CODE_FILES}


def _write_array_once(path: Path, array: np.ndarray) -> str:
    """np.save once (no pickle); an existing file must hold the same array (a rerun recomputes and compares)."""

    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), array):
            raise ValueError(f"{path} exists and differs from the recomputed list")
        return sha256_file(path)
    pending = path.with_name(f".{path.name}.{os.getpid()}.pending")
    with pending.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    return sha256_file(path)


def _write_npz_or_verify(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """write_npz_once, except that an existing file holding exactly these arrays is kept (restart); another raises."""

    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if set(stored.files) != set(arrays) or any(not np.array_equal(stored[k], v) for k, v in arrays.items()):
                raise ValueError(f"{path} exists and differs from the recomputed rows")
        return sha256_file(path)
    return write_npz_once(path, arrays)


def stage_measurers(hashes: dict[str, str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records, problems = {}, []
    for ctx in CONTEXTS:
        root = RUN_3 / ctx
        selection = _read(root / "SELECTION.json")
        card = str(select_measurer(selection["candidates"])["card_id"])
        folder = candidate_dir(ctx, card)
        result, scored = _read(folder / "candidate_result.json"), _read(folder / "REFIT_SCORED.json")
        record, found = measurer_problems(ctx, selection, result, scored, _read(root / "CONTEXT_PLAN.json"),
                                          _read(root / "EXCLUDED_GAME_KEYS.json"))
        checkpoint = folder / "model_refit.pkl"
        if Path(str(result.get("checkpoint"))) != checkpoint:
            found.append(f"{ctx}: candidate_result.json checkpoint {result.get('checkpoint')!r} is not {checkpoint}")
        files = [checkpoint, folder / "candidate_result.json", folder / "REFIT_SCORED.json", root / "SELECTION.json",
                 root / "CONTEXT_PLAN.json", root / "EXCLUDED_GAME_KEYS.json"]
        record["files"] = _hashes(files, hashes)
        record["model_sha256"] = record["files"][checkpoint.as_posix()]
        if record["model_sha256"] != record["model_sha256_recorded"]:
            found.append(f"{ctx}: {checkpoint} differs from its REFIT_SCORED.json hash")
        record["checkpoint"] = str(checkpoint)
        records[ctx] = record
        problems += found
    return records, problems


def stage_vdev_rows(out: Path, ctx: str, smoke: int | None, hashes: dict[str, str]) -> dict[str, Any]:
    """V_DEV rows of the context's REFIT cache, their consumption record, and the query list file (written once)."""

    root = RUN_3 / ctx
    plan = _read(root / "CONTEXT_PLAN.json")
    cache = RUN_3 / "cache" / plan["refit_cache_name"]
    manifest_path = cache / "MANIFEST.json"
    problems = []
    if sha256_file(manifest_path) != plan["refit_manifest_sha256"]:
        problems.append(f"{ctx}: the REFIT cache manifest differs from CONTEXT_PLAN.json")
    manifest = _read(manifest_path)
    files = [cache / manifest["metadata_files"][name] for name in CACHE_FIELDS]
    metadata = {name: np.load(path, allow_pickle=False) for name, path in zip(CACHE_FIELDS, files)}
    rows = dev_rows(metadata)
    del metadata
    consumed = dev_consumed(rows, _read(root / "EXCLUDED_GAME_KEYS.json")["excluded_game_keys"])
    scored_rows = {name: rows[name] for name in ("query_index", "label", "nominal_index")}
    if smoke:
        scored_rows = {name: value[:smoke] for name, value in scored_rows.items()}
    path = _list_file(out, "vdev", ctx)
    sha = _write_npz_or_verify(path, scored_rows)
    return {"rows": scored_rows, "consumed": consumed, "file": str(path), "sha256": sha, "problems": problems,
            "inputs": _hashes([manifest_path, *files], hashes), "cache": str(cache), "all_rows": int(len(rows["query_index"]))}


def _plan_jobs(kind: str, missing: Mapping[str, Sequence[int]], parts_per_job: int) -> list[tuple[str, str, list[int]]]:
    """Jobs of at most ``parts_per_job`` parts, interleaved across contexts (the largest list first)."""

    per_context = {ctx: [list(parts[i:i + parts_per_job]) for i in range(0, len(parts), parts_per_job)]
                   for ctx, parts in missing.items() if parts}
    order = sorted(per_context, key=lambda ctx: (-len(per_context[ctx]), CONTEXTS.index(ctx)))
    jobs = []
    while any(per_context[ctx] for ctx in order):
        for ctx in order:
            if per_context[ctx]:
                jobs.append((kind, ctx, per_context[ctx].pop(0)))
    return jobs


def _run_jobs(out: Path, jobs: Sequence[tuple[str, str, list[int]]], workers: int, rows_per_part: int) -> list[dict[str, Any]]:
    """Run worker jobs as subprocesses, at most ``workers`` at once; raise after the running ones end if one fails."""

    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="", KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8")
    pending, running, finished, failed = list(jobs), [], [], []
    while pending or running:
        while pending and len(running) < workers and not failed:
            kind, ctx, parts = pending.pop(0)
            stamp = utc().replace(":", "").replace("-", "")[:15]
            log_path = logs / f"{kind}_{ctx}_{parts[0]:05d}-{parts[-1]:05d}_{stamp}.log"
            log = log_path.open("w", encoding="utf-8")
            command = [sys.executable, "-B", "-m", "studies.tog_experiments_20261012.e4_vscore", "worker", "--kind", kind,
                       "--context", ctx, "--out", str(out), "--rows-per-part", str(rows_per_part), "--parts", *map(str, parts)]
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append({"process": process, "log": log, "log_path": str(log_path), "kind": kind, "context": ctx,
                            "parts": parts, "tick": time.perf_counter(), "started_utc": utc()})
            print(json.dumps({"start": kind, "context": ctx, "parts": parts, "pid": process.pid}), flush=True)
        time.sleep(2.0)
        for job in list(running):
            code = job["process"].poll()
            if code is None:
                continue
            running.remove(job)
            job["log"].close()
            record = {"kind": job["kind"], "context": job["context"], "parts": job["parts"], "returncode": code,
                      "seconds": time.perf_counter() - job["tick"], "started_utc": job["started_utc"], "log": job["log_path"]}
            (finished if code == 0 else failed).append(record)
            print(json.dumps({"end": job["kind"], "context": job["context"], "parts": job["parts"], "returncode": code,
                              "seconds": round(record["seconds"], 1)}), flush=True)
    if failed:
        raise RuntimeError(f"{len(failed)} worker job(s) failed; logs: {[job['log'] for job in failed]}")
    return finished


def _missing_parts(out: Path, kind: str, ctx: str, index: np.ndarray, identity: Mapping[str, Any], rows_per_part: int) -> list[int]:
    directory = out / f"{KINDS[kind]}_{ctx}"
    ranges = part_ranges(len(index), rows_per_part)
    return [part for part, (start, stop) in enumerate(ranges) if part_status(directory, part, index[start:stop], identity) != "done"]


def stage_vdev_check(out: Path, ctx: str, measurer: Mapping[str, Any], vdev: Mapping[str, Any], rows_per_part: int,
                     smoke: int | None) -> dict[str, Any]:
    """Merge the V_DEV parts (VDEV_<ctx>.npz) and write VDEV_CHECK_<ctx>.json once; a rerun reads the stored check."""

    target = out / f"VDEV_CHECK_{ctx}.json"
    npz = out / f"VDEV_{ctx}.npz"
    if target.exists():
        record = _read(target)
        if record.get("vdev_sha256") != sha256_file(npz):
            raise ValueError(f"{npz} differs from {target}")
        return record
    identity = {"kind": "vdev", "context": ctx, "card_id": measurer["card_id"], "model_sha256": measurer["model_sha256"]}
    index = vdev["rows"]["query_index"]
    ranges = part_ranges(len(index), rows_per_part)
    directory = out / f"VDEV_{ctx}"
    raw = merge_parts(directory, ranges, index, identity)
    sha = _write_npz_or_verify(npz, {"query_index": index, "raw_p": raw})   # an interrupted run may have written it
    result = _read(candidate_dir(ctx, measurer["card_id"]) / "candidate_result.json")
    record = vdev_record(ctx, measurer["card_id"], raw, vdev["rows"]["label"], vdev["consumed"], result, MEASURERS[ctx][1])
    if smoke:  # the first rows only: the log loss is not comparable and decides nothing
        record["passed"] = None
    record.update({"version": VERSION, "experiment": "E4", "step": "1 V_DEV reproduction", "smoke": smoke,
                   "model_sha256": measurer["model_sha256"], "vdev_file": str(npz), "vdev_sha256": sha,
                   "rows_file": vdev["file"], "rows_sha256": vdev["sha256"], "cache": vdev["cache"], "inputs": vdev["inputs"],
                   "scoring": {**_parts_summary(_receipts(directory, len(ranges))), "rows_per_part": rows_per_part,
                               "batch_rows": BATCH_ROWS, "predict": PREDICT, "loss": "train.raw_logloss_from_probability"},
                   "utc": utc()})
    write_once(target, _json_safe(record))
    return _read(target)


def stage_rows(out: Path, smoke: int | None, hashes: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    """q rows (Q_ROWS_h90 identity columns aligned to the locked engagements), contexts, OOF check, query lists."""

    problems: list[str] = []
    lock = q_rows_module.verify_lock(LABELS, 90)          # LABELS_h90 / ENGAGEMENTS / Q_DEV keys equal the label lock
    with np.load(Q_RUN / "Q_ROWS_h90.npz", allow_pickle=False) as table:      # identity columns only: no label column
        q_rows = {name: table[name] for name in ALIGN_COLUMNS}
    with np.load(LABELS / "ENGAGEMENTS.npz", allow_pickle=False) as table:
        engagements = {name: table[name] for name in (*ALIGN_COLUMNS, "query_h90", "valid_h90")}
    try:
        keep = align_q_rows(engagements, q_rows)
    except ValueError as error:
        return {}, [f"row table key mismatch: {error}"]
    role, fold = q_rows["role"].astype(np.int64), q_rows["fold"].astype(np.int64)
    match_index = q_rows["match_index"].astype(np.int64)
    query_pre, query_post = q_rows["query_pre"].astype(np.int64), engagements["query_h90"][keep].astype(np.int64)
    del engagements
    by_role = {name: int((role == code).sum()) for name, code in Q_ROLES.items()}
    if by_role != Q_ROLE_ROWS:
        problems.append(f"q rows by role {by_role} differ from PRESPEC_E4's {Q_ROLE_ROWS}")
    codes = row_contexts(role, fold)
    membership = np.load(PROJECTION / "CONTEXT_MEMBERSHIP.npy", allow_pickle=False)
    if membership.shape[0] != len(CONTEXTS) or membership.dtype != np.uint8 or match_index.max() >= membership.shape[1]:
        problems.append(f"CONTEXT_MEMBERSHIP.npy has an unexpected shape {membership.shape} or dtype {membership.dtype}")
        return {}, problems
    leaked = oof_leaks(codes, role, match_index, membership)
    oof = {"rule": "assemble.py:109-114: split bits 1-16 of the scoring context's membership are 0 for the row's match; "
                   "a TRAIN row's match carries the held-out target bit 32 of its oof context",
           "rows_checked": int(len(codes)), "leaked": int(leaked.sum()),
           "by_context": {ctx: {"rows": int((codes == c).sum()), "leaked": int((leaked & (codes == c)).sum())} for c, ctx in enumerate(CONTEXTS)},
           "passed": bool(not leaked.any())}
    if leaked.any():
        problems.append(f"OOF exclusion failed on {int(leaked.sum())} rows")
    names = {code: name for name, code in Q_ROLES.items()}
    contexts: dict[str, Any] = {}
    for c, ctx in enumerate(CONTEXTS):
        own = codes == c
        if smoke:      # smoke runs never write TEST rows' probabilities
            own &= role != Q_ROLES["TEST"]
        index = context_query_list(np.where(own, c, -1), c, query_pre, query_post)
        if smoke:
            index = index[:smoke]
        path = _list_file(out, "q", ctx)
        sha = _write_array_once(path, index)
        contexts[ctx] = {"queries": int(len(index)), "queries_file": str(path), "queries_sha256": sha,
                         "q_rows_by_role": {names[r]: int((own & (role == r)).sum()) for r in sorted(set(role[codes == c].tolist()))},
                         "matches_by_role": {names[r]: int(len(np.unique(match_index[own & (role == r)])))
                                             for r in sorted(set(role[codes == c].tolist()))}}
    record = {"q_rows": int(len(role)), "q_rows_by_role": by_role,
              "matches_by_role": {name: int(len(np.unique(match_index[role == code]))) for name, code in Q_ROLES.items()},
              "label_lock_utc": lock.get("locked_utc"), "oof_check": oof, "contexts": contexts,
              "inputs": _hashes([*ROW_FILES, PROJECTION / "CONTEXT_MEMBERSHIP.npy"], hashes)}
    return record, problems


def stage_merge(out: Path, ctx: str, measurer: Mapping[str, Any], vdev_check: Mapping[str, Any], rows: Mapping[str, Any],
                rows_per_part: int, common: Mapping[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    """PRED_<ctx>.npz and PRED_<ctx>.json, written once; a rerun reads the stored record."""

    target, npz = out / f"PRED_{ctx}.json", out / f"PRED_{ctx}.npz"
    if target.exists():
        record = _read(target)
        if record.get("pred_sha256") != sha256_file(npz):
            raise ValueError(f"{npz} differs from {target}")
        return record
    identity = {"kind": "q", "context": ctx, "card_id": measurer["card_id"], "model_sha256": measurer["model_sha256"]}
    index = _query_list(out, "q", ctx)
    ranges = part_ranges(len(index), rows_per_part)
    directory = out / f"PRED_{ctx}"
    started, tick = utc(), time.perf_counter()
    raw = merge_parts(directory, ranges, index, identity)
    sha = _write_npz_or_verify(npz, {"query_index": index, "raw_p": raw})   # an interrupted run may have written it
    bundle = queries.PREPROCESS / f"{ctx}_REFIT_TRAIN"
    inputs = {**measurer["files"], **_hashes(sorted(p for p in bundle.iterdir() if p.is_file()), hashes),
              **_hashes(SURFACE_FILES, hashes), **rows["inputs"], **_hashes([_list_file(out, "q", ctx)], hashes)}
    context_rows = rows["contexts"][ctx]
    record = {"version": VERSION, "experiment": "E4", "step": "2 scoring (raw probability, no calibration)", "context": ctx,
              "measurer": {name: measurer[name] for name in ("card_id", "family", "checkpoint", "model_sha256", "v_dev_raw_logloss",
                                                             "binding", "main_v")},
              "pred_file": str(npz), "pred_sha256": sha, "columns": ["query_index", "raw_p"],
              "queries_file": context_rows["queries_file"], "queries_sha256": context_rows["queries_sha256"],
              "rows": {"queries": context_rows["queries"], "q_rows_by_role": context_rows["q_rows_by_role"],
                       "matches_by_role": context_rows["matches_by_role"],
                       "rule": "TRAIN rows: the oof context of their fold; Q_DEV/Q_CAL/TEST rows: final; both the "
                               "prediction-time query (query_pre) and the h90 endpoint query (ENGAGEMENTS query_h90)"},
              "v_dev_check": {name: vdev_check.get(name) for name in ("raw_logloss", "candidate_result_raw_logloss", "prespec_printed",
                                                                      "abs_diff", "reproduces_candidate_result_4dp",
                                                                      "reproduces_prespec_4dp", "passed", "rows")},
              "v_dev_check_file": str(out / f"VDEV_CHECK_{ctx}.json"),
              "oof_check": {"rows": rows["oof_check"]["by_context"][ctx]["rows"],
                            "leaked": rows["oof_check"]["by_context"][ctx]["leaked"], "rule": rows["oof_check"]["rule"]},
              "scoring": {**_parts_summary(_receipts(directory, len(ranges))), "rows_per_part": rows_per_part, "batch_rows": BATCH_ROWS,
                          "predict": PREDICT, "calibration": None, "bundle": str(bundle)},
              "merge": _stage("merge", started, tick),
              "inputs": inputs, **common, "utc": utc()}
    write_once(target, _json_safe(record))
    return _read(target)


def run(out: Path = TOG_E4, *, workers: int = MAX_WORKERS, parts_per_job: int = 2, rows_per_part: int = ROWS_PER_PART,
        smoke: int | None = None) -> int:
    out = Path(out)
    if smoke and out.resolve() == TOG_E4.resolve():
        raise ValueError("a smoke run never writes into tog_e4_1")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be 1-{MAX_WORKERS}")
    report_path = out / "E4_VSCORE.json"
    if report_path.exists():
        print(json.dumps({"done": str(report_path)}), flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    run_started, stages, hashes = utc(), [], {}
    common = {"prespec": document_record(PRESPEC), "git_head": git_head(), "code_sha256": _code_hashes(), "versions": _versions(),
              "run_started_utc": run_started, "smoke": smoke, "workers": workers, "rows_per_part": rows_per_part}

    # measurers ------------------------------------------------------------------------------------------------
    s, t = utc(), time.perf_counter()
    measurers, problems = stage_measurers(hashes)
    stages.append(_stage("measurers", s, t))
    if problems:
        return _blocked(out, "measurers", problems, measurers=measurers)
    print(json.dumps({"measurers": {ctx: m["card_id"] for ctx, m in measurers.items()}}), flush=True)

    # step 1: V_DEV reproduction ---------------------------------------------------------------------------------
    s, t = utc(), time.perf_counter()
    vdev = {ctx: stage_vdev_rows(out, ctx, smoke, hashes) for ctx in CONTEXTS}
    problems = [p for ctx in CONTEXTS for p in vdev[ctx]["problems"]]
    stages.append(_stage("vdev_rows", s, t))
    if problems:
        return _blocked(out, "vdev", problems)
    s, t = utc(), time.perf_counter()
    missing = {ctx: _missing_parts(out, "vdev", ctx, vdev[ctx]["rows"]["query_index"],
                                   {"kind": "vdev", "context": ctx, "card_id": measurers[ctx]["card_id"],
                                    "model_sha256": measurers[ctx]["model_sha256"]}, rows_per_part) for ctx in CONTEXTS}
    jobs = _run_jobs(out, _plan_jobs("vdev", missing, parts_per_job), workers, rows_per_part)
    stages.append(_stage("vdev_score", s, t, jobs=jobs))
    s, t = utc(), time.perf_counter()
    checks = {ctx: stage_vdev_check(out, ctx, measurers[ctx], vdev[ctx], rows_per_part, smoke) for ctx in CONTEXTS}
    table = [{name: checks[ctx].get(name) for name in ("context", "card_id", "rows", "candidate_result_raw_logloss", "prespec_printed",
                                                       "raw_logloss", "abs_diff", "reproduces_candidate_result_4dp",
                                                       "reproduces_prespec_4dp", "passed")} for ctx in CONTEXTS]
    stages.append(_stage("vdev_check", s, t))
    print(json.dumps({"v_dev_reproduction": table}), flush=True)
    if not smoke and not all(row["passed"] for row in table):
        return _blocked(out, "vdev", [f"{row['context']}: V_DEV raw log loss {row['raw_logloss']!r} vs "
                                      f"{row['candidate_result_raw_logloss']!r}" for row in table if not row["passed"]],
                        v_dev_reproduction=table)

    # step 2: rows, OOF exclusion, scoring -------------------------------------------------------------------------
    s, t = utc(), time.perf_counter()
    rows, problems = stage_rows(out, smoke, hashes)
    stages.append(_stage("rows_and_oof", s, t))
    if problems:
        return _blocked(out, "rows", problems, oof_check=rows.get("oof_check"))
    print(json.dumps({"q_rows_by_role": rows["q_rows_by_role"], "oof_leaked": rows["oof_check"]["leaked"]}), flush=True)
    s, t = utc(), time.perf_counter()
    missing = {ctx: _missing_parts(out, "q", ctx, _query_list(out, "q", ctx),
                                   {"kind": "q", "context": ctx, "card_id": measurers[ctx]["card_id"],
                                    "model_sha256": measurers[ctx]["model_sha256"]}, rows_per_part) for ctx in CONTEXTS}
    jobs = _run_jobs(out, _plan_jobs("q", missing, parts_per_job), workers, rows_per_part)
    stages.append(_stage("score", s, t, jobs=jobs))

    # merge and report -------------------------------------------------------------------------------------------
    s, t = utc(), time.perf_counter()
    common = {**common, "stages_before_merge": stages}
    preds = {ctx: stage_merge(out, ctx, measurers[ctx], checks[ctx], rows, rows_per_part, common, hashes) for ctx in CONTEXTS}
    stages.append(_stage("merge", s, t))
    worker_peaks = [p["scoring"]["worker_peak_memory_bytes_max"] for p in preds.values()]
    worker_peaks += [c["scoring"]["worker_peak_memory_bytes_max"] for c in checks.values()]
    report = {"version": VERSION, "experiment": "E4", "steps": ["1 V_DEV reproduction (six contexts)", "2 scoring with OOF exclusion"],
              "measurers": measurers, "v_dev_reproduction": table, "v_dev_passed": None if smoke else all(r["passed"] for r in table),
              "oof_check": rows["oof_check"], "q_rows": rows["q_rows"], "q_rows_by_role": rows["q_rows_by_role"],
              "matches_by_role": rows["matches_by_role"],
              "contexts": {ctx: {"pred_file": p["pred_file"], "pred_sha256": p["pred_sha256"], "record": str(out / f"PRED_{ctx}.json"),
                                 "record_sha256": sha256_file(out / f"PRED_{ctx}.json"), "queries": p["rows"]["queries"],
                                 "q_rows_by_role": p["rows"]["q_rows_by_role"], "v_dev_check": str(out / f"VDEV_CHECK_{ctx}.json"),
                                 "v_dev_check_sha256": sha256_file(out / f"VDEV_CHECK_{ctx}.json")} for ctx, p in preds.items()},
              "stages": stages, "peak_memory_bytes": {"driver": peak_memory_bytes(),
                                                       "worker_max": max((p for p in worker_peaks if p), default=None)},
              "notes": ["raw probabilities of the LightGBM V (pickle predict_proba), no calibration (PRESPEC_E4 section 3)",
                        "TEST rows' probabilities in PRED_final.npz are label information outside any seal; no summary of "
                        "them was computed (PRESPEC_E4 section 3)",
                        "labels and agreement are Task 11; q retraining and the TEST step are Task 12"],
              **{k: v for k, v in common.items() if k != "stages_before_merge"}, "utc": utc()}
    write_once(report_path, _json_safe(report))
    print(json.dumps({"done": str(report_path), "v_dev_passed": report["v_dev_passed"], "oof_leaked": rows["oof_check"]["leaked"],
                      "pred_sha256": {ctx: p["pred_sha256"] for ctx, p in preds.items()}}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--out", default=str(TOG_E4))
    go.add_argument("--workers", type=int, default=MAX_WORKERS)
    go.add_argument("--parts-per-job", type=int, default=2)
    go.add_argument("--rows-per-part", type=int, default=ROWS_PER_PART)
    go.add_argument("--smoke", type=int, default=None, help="score only the first N rows of each list into another folder")
    one = sub.add_parser("worker")
    one.add_argument("--kind", required=True, choices=sorted(KINDS))
    one.add_argument("--context", required=True, choices=CONTEXTS)
    one.add_argument("--parts", type=int, nargs="+", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--rows-per-part", type=int, required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.command == "worker":
        return worker(args.kind, args.context, args.parts, Path(args.out), args.rows_per_part)
    return run(Path(args.out), workers=args.workers, parts_per_job=args.parts_per_job, rows_per_part=args.rows_per_part,
               smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
