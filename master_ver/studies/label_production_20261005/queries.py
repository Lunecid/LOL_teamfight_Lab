"""L1/L2 input side of label production: per-context query sets and model inputs for arbitrary query rows.

The V input of a query row is built exactly as for a nominal V row
(``BankSurface.iter_nominal_rows``): projection features + shared-history
record -> ``make_view`` -> ``semantic_row`` -> the context's REFIT_TRAIN bundle
(``transform_rows``).  Two differences, both required by the label contract:
rows are read by QUERY_TABLE index (engagement pre/post states), and the
training replacement swap is never applied (labels are the Blue view).

Design: reports/label_production_20261005/LABEL_PRODUCER_DESIGN_KO.md (L1, L2).
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.v_full_preprocess_20261003.artifacts import load_bundle  # noqa: E402
from studies.v_full_preprocess_20261003.surface import BankSurface, _integer  # noqa: E402
from studies.v_model_fit_input_20261003.common import semantic_row, transform_rows  # noqa: E402
from studies.v_model_input_20261003.view import make_view  # noqa: E402

SHARED_BANK = ROOT / "outputs/shared_history_eight_cpu_20261003/full_1"
PROJECTION = ROOT / "outputs/v_input_projection_20261002/full_raw_20261002_1"
PREPROCESS = ROOT / "outputs/v_full_preprocess_20261003/full_1"
CONTEXTS = ("final", "oof0", "oof1", "oof2", "oof3", "oof4")
ROLES = ("TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST")
FINAL_ROLES = ("Q_DEV", "Q_CAL", "TEST")     # q rows labelled with the final V (design §3)
QUERY_COLUMNS = ("query_pre", "query_h60", "query_h90", "query_h120")
STABILITY_COLUMNS = ("query_pre", "query_h90")  # Q_DEV h90 keys scored by all six V (v_stability)
BATCH_ROWS = 1024


def context_queries(engagements: Mapping[str, np.ndarray], context: str) -> np.ndarray:
    """Sorted unique QUERY_TABLE indices one context must score."""

    if context not in CONTEXTS:
        raise ValueError(f"unknown context {context!r}")
    role, fold = np.asarray(engagements["role"]), np.asarray(engagements["fold"])
    parts = []
    if context == "final":
        own = np.isin(role, [ROLES.index(name) for name in FINAL_ROLES])
    else:
        own = (role == ROLES.index("TRAIN")) & (fold == int(context[-1]))
        stability = role == ROLES.index("Q_DEV")
        parts += [np.asarray(engagements[name])[stability] for name in STABILITY_COLUMNS]
    parts += [np.asarray(engagements[name])[own] for name in QUERY_COLUMNS]
    values = np.concatenate(parts).astype(np.int64)
    return np.unique(values[values >= 0])


def batches(query_indices: np.ndarray, batch_rows: int = BATCH_ROWS) -> list[tuple[int, int]]:
    """Fixed [start, stop) positions of the canonical scoring batches (independent of the worker count)."""

    n = int(len(query_indices))
    return [(start, min(start + batch_rows, n)) for start in range(0, n, batch_rows)]


class QuerySurface(BankSurface):
    """``BankSurface`` that yields Blue-view semantic rows for arbitrary QUERY_TABLE indices."""

    def iter_query_rows(self, query_indices: Sequence[int], features_hook: Any = None) -> Iterator[tuple[int, dict[str, Any], dict[str, Any]]]:
        """Rows in the given (strictly increasing) order; ``features_hook(query_index, query_ms, features)``
        may return replaced projection features (used only by the clock-only diagnostic)."""

        indices = np.asarray(query_indices, dtype=np.int64)
        if len(indices) and (np.any(np.diff(indices) <= 0) or indices[0] < 0 or indices[-1] >= len(self.loader.queries)):
            raise ValueError("query indices must be strictly increasing and inside QUERY_TABLE")
        position = 0
        while position < len(indices):
            match_index = int(self.loader.queries[int(indices[position])]["match_index"])
            first_query = _integer(self.loader.match_queries[match_index]["first_query"], "first_query")
            n_queries = _integer(self.loader.match_queries[match_index]["n_queries"], "n_queries")
            stop = position
            while stop < len(indices) and int(indices[stop]) < first_query + n_queries:
                stop += 1
            if int(indices[position]) < first_query:
                raise ValueError(f"query {int(indices[position])} lies before its match's first query")
            _match, key, role, _is_train_patch, _fold, records = self._records_for_match(match_index)
            by_query = {int(record["query_id"]): record for record in records}
            for query_index in (int(value) for value in indices[position:stop]):
                query = self.loader.queries[query_index]
                query_ms = _integer(query["query_ms"], "query_ms")
                if int(query["match_index"]) != match_index:
                    raise ValueError(f"query {query_index} belongs to another match")
                record = by_query.get(query_index)
                if record is None or _integer(record.get("query_ms"), "shard query_ms") != query_ms:
                    raise ValueError(f"shard record differs for query {query_index}")
                supplied = self.loader.query_input(query_index, include_histories=False)
                audit = supplied.get("audit") if isinstance(supplied, Mapping) else None
                if not isinstance(audit, Mapping) or int(audit.get("match_index", -1)) != match_index or int(audit.get("query_ms", -1)) != query_ms:
                    raise ValueError(f"projection query identity differs for query {query_index}")
                nonitem = record.get("nonitem", record.get("nonitem_history"))
                item = record.get("item", record.get("item_history"))
                if nonitem is None or item is None:
                    raise ValueError(f"shard history record is incomplete for query {query_index}")
                features = supplied["features"] if features_hook is None else features_hook(query_index, query_ms, supplied["features"])
                view = make_view(features, nonitem, item_history=item)  # Blue view: no replacement swap
                yield query_index, semantic_row(view), {"match_index": match_index, "query_ms": query_ms, "game_key": key, "role": role}
            position = stop


def family_inputs(structured: Any, family: str) -> Any:
    """The model input of one family, as ``CacheSource._family_inputs`` builds it in training."""

    if family == "plain_mlp":
        return replace(structured, dense_features=np.asarray(structured.table_csr.toarray(), dtype=np.float32))
    if family in {"logistic", "lightgbm"}:
        return {"table_csr": structured.table_csr}
    return structured


def open_inputs(context: str) -> tuple[QuerySurface, Any]:
    """Bank reader and the context's REFIT_TRAIN preprocessing bundle (the scope of the selected checkpoint)."""

    surface = QuerySurface(SHARED_BANK, PROJECTION, seed=20261001, verify_projection_hashes=False)
    return surface, load_bundle(PREPROCESS / f"{context}_REFIT_TRAIN")


def transform(rows: Sequence[Mapping[str, Any]], bundle: Any) -> Any:
    if not 0 < len(rows) <= BATCH_ROWS:
        raise ValueError("transform batches hold 1..1024 rows")
    return transform_rows(rows, bundle)
