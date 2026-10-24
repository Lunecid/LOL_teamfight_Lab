"""One declared semantic surface for all five V families.

Raw views contain no fit state. The reader below is for bounded integration
checks; full training must use a verified preaggregated history bank rather
than replaying the source at every epoch.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
import json

import numpy as np
from scipy import sparse

from studies.observable_state_20261002.kernel import STATIC_NAMES
from studies.item_node_observed_20261002.ledger import ITEM_EVENT_TYPES, ITEM_FIELD_ROLES
from studies.v_model_input_20261003.history import HistoryAccumulator
from studies.v_model_input_20261003.view import make_view
from studies.v_preparation_20261002.partition import assigned_role, context_membership
from .preprocess import (NumericFieldSpec, CategoricalFieldSpec, FitPreprocessor,
                         fit_numeric, fit_ragged_vocabulary, make_fit_evidence)

VERSION = "v_common_fit_surface_20261003_v1"
ITEM_GROUPS = tuple(f"history:{event}:{field}" for event in ITEM_EVENT_TYPES for field in ITEM_FIELD_ROLES)
CATEGORY_LAYOUT = {"node": tuple(STATIC_NAMES), "team": tuple(f"ban{i}" for i in range(5)), "global": ()}
BAG_LAYOUT = {"node": ("current_item", *ITEM_GROUPS, "nonitem"),
              "team": ("nonitem",), "global": (*ITEM_GROUPS, "nonitem")}
CATEGORY_NAMESPACES = {**{name: f"static:{name}" for name in STATIC_NAMES},
                       **{f"ban{i}": "static:champion_id" for i in range(5)}}
BAG_NAMESPACES = {**{name: "item" for name in ("current_item", *ITEM_GROUPS)}, "nonitem": "nonitem"}
NUMERIC_SPECS = {"node": NumericFieldSpec("node", "player"),
                 "team": NumericFieldSpec("team", "team"),
                 "global": NumericFieldSpec("global", "row")}
NUMERIC_DIMS = {"node": 102, "team": 8, "global": 75}
MAX_BATCH_ROWS = 1024


def _array(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "biuf":
        raise ValueError(f"wrong numeric shape/type for {name}")
    if np.any(np.isinf(result)):
        raise ValueError(f"infinite input in {name}")
    return result.astype(np.float64)


def _bag(values, counts=None):
    values = list(values)
    if counts is None:
        return dict(Counter(int(value) for value in values))
    counts = list(counts)
    if len(values) != len(counts):
        raise ValueError("bag IDs/counts differ")
    result = Counter()
    for value, count in zip(values, counts):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("bag counts must be positive integers")
        result[int(value)] += int(count)
    return dict(result)


def _context(context):
    if tuple(context["event_types"]) != tuple(ITEM_EVENT_TYPES) or tuple(context["field_roles"]) != tuple(ITEM_FIELD_ROLES):
        raise ValueError("item history type/field order differs from accepted v3")
    shapes = (("event_counts", (4,)), ("event_presence", (4,)),
              ("field_presence_counts", (4, 3)), ("field_validity_counts", (4, 3)),
              ("field_malformed_counts", (4, 3)), ("event_malformed_counts", (4,)))
    parts = [_array(context[name], shape, name).reshape(-1) for name, shape in shapes]
    parts.append(np.array([float(bool(context["history_observed"]))]))
    numeric = np.concatenate(parts)
    vectors = context["item_id_vectors"]
    if len(vectors) != 4 or any(len(fields) != 3 for fields in vectors):
        raise ValueError("item history vectors need four event types and three roles")
    bags = {name: _bag(vectors[i // 3][i % 3]) for i, name in enumerate(ITEM_GROUPS)}
    if numeric.shape != (49,):
        raise ValueError("item history numeric width differs")
    return numeric, bags


def semantic_row(view):
    """Convert an accepted P2a view, preserving every declared model field."""
    if set(view) != {"version", "features", "nonitem_history", "item_history", "focal_is_blue"}:
        raise ValueError("view field allowlist violated")
    if view["item_history"] is None:
        raise ValueError("complete item history is required for every family")
    features, history, items = view["features"], view["nonitem_history"], view["item_history"]
    current = features["items"]
    item_numeric, item_bags = zip(*(_context(context) for context in items["node"]))
    unknown_numeric, unknown_bags = _context(items["unassigned"])
    offsets = np.asarray(current["offsets"])
    current_bags = [_bag(current["item_ids"][offsets[i]:offsets[i + 1]],
                         current["counts"][offsets[i]:offsets[i + 1]]) for i in range(10)]
    support = np.asarray(current["recorded_ledger_supported"], dtype=bool)
    if support.shape != (10,) or any(bag for bag, known in zip(current_bags, support) if not known):
        raise ValueError("unsupported current ledgers must have masked empty bags")
    current_numeric = np.column_stack(([sum(bag.values()) for bag in current_bags],
                                      current["recorded_ledger_supported"], np.zeros(10),
                                      np.full(10, float(current["unassigned_observed"]))))
    numeric = {
        "node": np.concatenate((_array(features["node_frame"], (10, 46), "node_frame"),
                                _array(features["player_kda"], (10, 3), "player_kda"),
                                current_numeric, np.stack(item_numeric)), axis=1),
        "team": np.column_stack((_array(features["team_growth"], (2, 6), "team_growth"),
                                 _array(features["cached_gold"], (2,), "cached_gold"),
                                 _array(features["team_kills"], (2,), "team_kills"))),
        "global": np.concatenate(([float(features["elapsed_minutes"]), float(features["frame_age_seconds"])],
                                  _array(features["team_growth_delta"], (6,), "team_growth_delta"),
                                  _array(features["objective_counts"], (9,), "objective_counts"),
                                  _array(features["objective_age_seconds"], (9,), "objective_age_seconds"), unknown_numeric)),
    }
    for lane, values in numeric.items():
        _array(values, {"node": (10, 102), "team": (2, 8), "global": (75,)}[lane], lane)
    static = np.asarray(features["static_ids"])
    bans = np.asarray(features["bans"])
    if static.shape != (10, 14) or bans.shape != (2, 5) or static.dtype.kind not in "iu" or bans.dtype.kind not in "iu":
        raise ValueError("static and ban categories must be the original integer arrays")
    categories = {"node": {name: static[:, i].copy() for i, name in enumerate(STATIC_NAMES)},
                  "team": {f"ban{i}": bans[:, i].copy() for i in range(5)}, "global": {}}
    bags = {"node": {"current_item": current_bags,
                      **{name: [bag[name] for bag in item_bags] for name in ITEM_GROUPS},
                      "nonitem": list(history["node"])},
            "team": {"nonitem": list(history["team"])},
            "global": {**{name: [unknown_bags[name]] for name in ITEM_GROUPS}, "nonitem": [history["global"]]}}
    teams = np.asarray(features["team_ids"])
    if teams.shape != (10,) or not np.all(np.isin(teams, (100, 200))) or np.count_nonzero(teams == 100) != 5:
        raise ValueError("actual participant team routing is required")
    return {"numeric": numeric, "categories": categories, "bags": bags,
            "team_ids": teams.copy(), "focal_is_blue": bool(view["focal_is_blue"]), "current_item_supported": support.copy()}


def query_view(loader, query_index):
    """Bounded integration reader; no fit state or target enters the view."""
    supplied = loader.query_input(query_index, include_histories=False)
    match_index = supplied["audit"]["match_index"]
    cutoff = supplied["audit"]["query_ms"]
    accumulator = HistoryAccumulator()
    for record in loader._history(match_index).iter_prefix(cutoff):
        accumulator.add(record)
    local = int(loader.queries[query_index]["local_query"])
    observed = loader._item_loader().node_input(match_index, local)
    contexts = {"node": observed["assigned_history_by_participant"],
                "unassigned": observed["unassigned_history_context"]}
    return make_view(supplied["features"], accumulator.snapshot_at(cutoff), item_history=contexts)


def namespace_rows(rows):
    namespaces = sorted(set(CATEGORY_NAMESPACES.values()) | set(BAG_NAMESPACES.values()))
    result = {name: [] for name in namespaces}
    for row in rows:
        values = {name: [] for name in namespaces}
        for lane in CATEGORY_LAYOUT:
            for group, array in row["categories"][lane].items():
                values[CATEGORY_NAMESPACES[group]].extend(int(value) for value in array)
            for group, bags in row["bags"][lane].items():
                for bag in bags:
                    values[BAG_NAMESPACES[group]].extend(bag)
        for name in namespaces:
            result[name].append(values[name])
    return result


def fit_common(rows, fit_indices, **scope):
    """Fit a bounded integration scope; full-corpus fitting needs streaming."""
    if not 0 < len(rows) <= MAX_BATCH_ROWS:
        raise ValueError("bounded integration fit requires 1..1024 rows; use the full-corpus streaming fitter")
    if any(scope.get(name) is None for name in ("game_keys", "roles", "membership", "source_hash")):
        raise ValueError("scope evidence must be supplied before fitting")
    if set(scope["membership"]) != {"fit", "stop", "refit", "dev", "cal", "target"}:
        raise ValueError("all six protocol membership flags are required")
    if len(scope["game_keys"]) != len(rows) or len(scope["roles"]) != len(rows):
        raise ValueError("source keys/roles must identify every source row")
    if any(not isinstance(key, str) or len(key) != 64 for key in scope["game_keys"]):
        raise ValueError("canonical source game keys are required")
    if any(role not in {"TRAIN", "V_DEV", "V_CAL", "Q_DEV", "Q_CAL", "TEST"} for role in scope["roles"]):
        raise ValueError("source patch roles are required")
    if scope.get("context", "final").startswith("oof") and not scope.get("excluded_game_keys"):
        raise ValueError("whole-game OOF exclusions are required")
    context = scope.get("context", "final")
    excluded_fold = None if context == "final" else int(context.removeprefix("oof"))
    for index, (key, role) in enumerate(zip(scope["game_keys"], scope["roles"])):
        patch = "15.14" if role == "TRAIN" else "15.16" if role == "TEST" else "15.15"
        if assigned_role(patch, key) != role:
            raise ValueError("declared role differs from the canonical game partition")
        flags = context_membership(patch, key, excluded_fold)
        if any(bool(scope["membership"][name][index]) != flag for name, flag in flags.items()):
            raise ValueError("declared membership differs from the canonical whole-game partition")
    evidence = make_fit_evidence(fit_indices, n_rows=len(rows), **scope)
    numeric = tuple((lane, fit_numeric(np.stack([row["numeric"][lane] for row in rows]),
                                      fit_indices=fit_indices, spec=NUMERIC_SPECS[lane], **scope))
                    for lane in sorted(NUMERIC_SPECS))
    categorical = tuple((name, fit_ragged_vocabulary(values, fit_indices=fit_indices,
                                                    spec=CategoricalFieldSpec(name), **scope))
                        for name, values in namespace_rows(rows).items())
    return FitPreprocessor(context=evidence.context, scope=evidence.scope, numeric=numeric,
                           categorical=categorical, evidence=evidence)


def table_layout(preprocessor):
    """Stable CSR columns; categorical indices are never continuous IDs."""
    blocks, cursor = [], 0
    def add(key, width):
        nonlocal cursor
        blocks.append((key, cursor, int(width)))
        cursor += int(width)
    for lane, size in (("node", 10), ("team", 2), ("global", 1)):
        add(("numeric", lane), NUMERIC_DIMS[lane] * size)
    add(("routing", "node"), 10)
    add(("side", "global"), 1)
    vocabularies = preprocessor.categorical_map
    for lane, size in (("node", 10), ("team", 2), ("global", 1)):
        for kind, layout, namespaces in (("category", CATEGORY_LAYOUT, CATEGORY_NAMESPACES), ("bag", BAG_LAYOUT, BAG_NAMESPACES)):
            for group in layout[lane]:
                vocab = vocabularies[namespaces[group]]
                add((kind, lane, group), size * (len(vocab.categories) + 2))
                if kind == "bag":
                    add(("bag_count", lane, group), size)
    return blocks, cursor


def transform_rows(rows, preprocessor):
    """Build shared structured and table views from the same fitted artifact."""
    from .models import StructuredBatch, BagInput
    if not 0 < len(rows) <= MAX_BATCH_ROWS:
        raise ValueError("model input must be transformed in batches of at most1024 rows")
    numeric = {lane: preprocessor.numeric_map[lane].transform(np.stack([row["numeric"][lane] for row in rows])) for lane in NUMERIC_SPECS}
    categories, bags = {}, {}
    for lane, size in (("node", 10), ("team", 2), ("global", 1)):
        categories[lane] = {group: preprocessor.categorical_map[CATEGORY_NAMESPACES[group]].transform(
            np.stack([row["categories"][lane][group] for row in rows])) for group in CATEGORY_LAYOUT[lane]}
        bags[lane] = {}
        for group in BAG_LAYOUT[lane]:
            vocab = preprocessor.categorical_map[BAG_NAMESPACES[group]]
            ids, counts, offsets = [], [], []
            observed = []
            for row in rows:
                bounds, totals = [len(ids)], []
                for bag in row["bags"][lane][group]:
                    keys = list(bag)
                    ids.extend(vocab.transform(np.asarray(keys, dtype=object)).tolist())
                    counts.extend(bag[key] for key in keys)
                    totals.append(sum(bag.values()))
                    bounds.append(len(ids))
                offsets.append(bounds)
                observed.append(totals)
            bags[lane][group] = BagInput(np.asarray(ids, dtype=np.int64), np.asarray(counts, dtype=np.float32),
                                        np.asarray(offsets, dtype=np.int64), np.asarray(observed, dtype=np.float32))
    teams = np.stack([row["team_ids"] for row in rows])
    sides = np.array([float(row["focal_is_blue"]) for row in rows], dtype=np.float32)
    blocks, width = table_layout(preprocessor)
    rr, cc, vv = [], [], []
    def add_dense(array, offset):
        array = np.asarray(array).reshape(len(rows), -1)
        x, y = np.nonzero(array)
        rr.extend(x.tolist()); cc.extend((y + offset).tolist()); vv.extend(array[x, y].tolist())
    for key, start, _ in blocks:
        kind, lane, *tail = key
        if kind == "numeric":
            add_dense(numeric[lane], start)
        elif kind == "routing":
            add_dense((teams == 200).astype(np.float32), start)
        elif kind == "side":
            add_dense(sides, start)
        elif kind == "bag_count":
            add_dense(bags[lane][tail[0]].observed_count, start)
        else:
            group, size = tail[0], {"node": 10, "team": 2, "global": 1}[lane]
            vocab = preprocessor.categorical_map[(CATEGORY_NAMESPACES if kind == "category" else BAG_NAMESPACES)[group]]
            ncat = len(vocab.categories) + 2
            for r in range(len(rows)):
                for slot in range(size):
                    if kind == "category":
                        indices, weights = [categories[lane][group][r, slot]], [1.0]
                    else:
                        bag = bags[lane][group]
                        a, b = bag.offsets[r, slot:slot + 2]
                        indices, weights = bag.indices[a:b], bag.counts[a:b]
                    for index, count in zip(indices, weights):
                        rr.append(r); cc.append(start + slot * ncat + int(index)); vv.append(float(count))
    csr = sparse.csr_matrix((np.asarray(vv, dtype=np.float32), (rr, cc)), shape=(len(rows), width), dtype=np.float32)
    csr.sum_duplicates(); csr.sort_indices()
    return StructuredBatch(node_numeric=numeric["node"], team_numeric=numeric["team"],
                           global_numeric=np.column_stack((numeric["global"], sides)), player_team_ids=teams,
                           current_item_supported=np.stack([row["current_item_supported"] for row in rows]),
                           node_categories=categories["node"], team_categories=categories["team"], global_categories=categories["global"],
                           node_bags=bags["node"], team_bags=bags["team"], global_bags=bags["global"],
                           category_namespaces=CATEGORY_NAMESPACES, bag_namespaces=BAG_NAMESPACES, table_csr=csr)


def semantic_fingerprint():
    payload = {"version": VERSION, "numeric": NUMERIC_DIMS, "categories": CATEGORY_LAYOUT,
               "bags": BAG_LAYOUT, "category_namespaces": CATEGORY_NAMESPACES,
               "bag_namespaces": BAG_NAMESPACES, "routing": "actual teams 100/200 -> indicator",
               "side": "focal_is_blue appended unscaled to global input"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
