"""E2a UNK census: share of model-input tokens that the frozen vocabularies map to UNK (diagnostic; no label or score).

README
  Vocabularies: the frozen V final REFIT_TRAIN bundle (outputs/v_full_preprocess_20261003/full_1/final_REFIT_TRAIN,
  loaded with the frozen v_full_preprocess_20261003.artifacts.load_bundle; q_run_1 main_h90 REFIT_TRAIN carries the same
  categories for all 16 fields).  Tokens follow the frozen input path (v_model_fit_input_20261003.common.semantic_row and
  preprocess.CategoricalVocabulary.transform):
    node categories   projection STATIC_IDS, 10 participants x 14 fields -> namespace 'static:<field>'
    team categories   projection BANS (2 x 5)                         -> 'static:champion_id'
    current-item bags item bank ITEM_IDS / ITEM_COUNTS per participant -> 'item'
    item-history bags shared-history 'item' contexts (10 participants + unassigned; 4 event types x 3 roles) -> 'item'
    nonitem bags      shared-history 'nonitem' bags (10 participants, 2 teams, 1 global) -> 'nonitem'
  A token is preprocess._category_token(value) (an int >= 0 or a non-empty string); a value without a token (e.g. a
  negative ban id) and a token outside the vocabulary both map to UNK index 1.  A bag contributes one entry per distinct
  key, weighted by its count.  Reported per namespace group: entries / UNK entries / share and count-weighted tokens /
  UNK / share, separately for (a) every bank query state and (b) the supported nominal V rows; static fields also per
  match (participant slots, as in the inventory's raw estimate).  'unk_nonpositive' counts the values without a token
  (UNK by the encoder rule in 2025 as well); 'unk_unseen' counts tokens absent from the vocabulary.  The most frequent
  unseen tokens are listed.  parity_15_16 (500 accepted 15.16 TEST matches) is the in-distribution reference.

Output: reports/external_validation_20261009/E2a_UNK_CENSUS.json
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402

V_FINAL_BUNDLE = C.ROOT / "outputs" / "v_full_preprocess_20261003" / "full_1" / "final_REFIT_TRAIN"
UNK_CENSUS = C.REPORTS / "E2a_UNK_CENSUS.json"
TOP = 20


def _vocab_sets() -> dict:
    from studies.v_full_preprocess_20261003.artifacts import load_bundle

    pre = load_bundle(V_FINAL_BUNDLE)
    return {name: frozenset(vocab.mapping) for name, vocab in pre.categorical_map.items()}


class Tally:
    def __init__(self):
        self.c = Counter()

    def add(self, group: str, entries: int, unk_unseen: int, unk_nonpositive: int, weight: int, w_unk_unseen: int, w_unk_nonpositive: int):
        self.c[(group, "entries")] += entries
        self.c[(group, "unk_unseen")] += unk_unseen
        self.c[(group, "unk_nonpositive")] += unk_nonpositive
        self.c[(group, "weighted")] += weight
        self.c[(group, "w_unk_unseen")] += w_unk_unseen
        self.c[(group, "w_unk_nonpositive")] += w_unk_nonpositive

    def merge(self, other: "Tally"):
        self.c.update(other.c)

    def result(self) -> dict:
        out = {}
        for group in sorted({g for g, _ in self.c}):
            e, w = self.c[(group, "entries")], self.c[(group, "weighted")]
            ue = self.c[(group, "unk_unseen")] + self.c[(group, "unk_nonpositive")]
            uw = self.c[(group, "w_unk_unseen")] + self.c[(group, "w_unk_nonpositive")]
            out[group] = {"entries": e, "unk_entries": ue, "unk_share": round(ue / e, 6) if e else None,
                          "unk_unseen_entries": self.c[(group, "unk_unseen")],
                          "unk_nonpositive_entries": self.c[(group, "unk_nonpositive")],
                          "weighted_tokens": w, "unk_weighted": uw, "unk_weighted_share": round(uw / w, 6) if w else None}
        return out


def _history_task(task: tuple) -> dict:
    """Count nonitem and item-history tokens of one history shard, split by the nominal-row mask."""
    C.runtime_hygiene()
    import json
    from studies.v_model_fit_input_20261003.preprocess import _category_token
    from studies.v_shared_history_production_20261003.store import ShardReader

    directory, plan_path, first, last, nominal_query_ids, vocab_item, vocab_nonitem = task
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    reader = ShardReader(Path(directory), identity=plan)
    nominal = set(nominal_query_ids)
    tallies = {"all_queries": Tally(), "nominal_rows": Tally()}
    unseen = {"item": Counter(), "nonitem": Counter()}
    n_records = 0
    for match in range(first, last):
        for record in reader.match_records(match):
            n_records += 1
            scopes = ("all_queries", "nominal_rows") if int(record["query_id"]) in nominal else ("all_queries",)
            local = Tally()
            history = record["nonitem"]
            for lane, bags in (("node", history["node"]), ("team", history["team"]), ("global", [history["global"]])):
                for bag in bags:
                    e = uu = un = w = wu = wn = 0
                    for key, count in bag.items():
                        token = _category_token(key)
                        e += 1
                        w += int(count)
                        if token is None:
                            un += 1
                            wn += int(count)
                        elif token not in vocab_nonitem:
                            uu += 1
                            wu += int(count)
                            unseen["nonitem"][str(token)] += int(count)
                    local.add(f"nonitem:{lane}", e, uu, un, w, wu, wn)
            items = record["item"]
            for lane, contexts in (("node", items["node"]), ("unassigned", [items["unassigned"]])):
                for context in contexts:
                    for event_vectors in context["item_id_vectors"]:
                        for vector in event_vectors:
                            bag = Counter(int(value) for value in vector)
                            e = uu = un = w = wu = wn = 0
                            for key, count in bag.items():
                                token = _category_token(key)
                                e += 1
                                w += count
                                if token is None:
                                    un += 1
                                    wn += count
                                elif token not in vocab_item:
                                    uu += 1
                                    wu += count
                                    unseen["item"][str(token)] += count
                            if e:
                                local.add(f"item:history_{lane}", e, uu, un, w, wu, wn)
            for scope in scopes:
                tallies[scope].merge(local)
    return {"tallies": {k: dict(v.c) for k, v in tallies.items()}, "unseen": {k: dict(v) for k, v in unseen.items()},
            "n_records": n_records}


def census_dataset(dataset: str, vocab: dict, workers: int, log) -> dict:
    import json
    import numpy as np

    started = time.perf_counter()
    projection, items, history = (C.stage_dir(dataset, s) for s in ("projection", "items", "history"))
    acceptance = C.read_json(history / "ACCEPTANCE.json")
    if acceptance.get("complete") is not True:
        raise ValueError(f"{dataset}: shared history incomplete")
    mq = np.load(projection / "MATCH_QUERIES.npy")
    vrows = np.load(projection / "V_ROWS.npy")
    first_query = mq["first_query"].astype(np.int64)
    n_queries = mq["n_queries"].astype(np.int64)
    supported = vrows["supported"].astype(bool)
    nominal_q = first_query[vrows["match_index"][supported].astype(np.int64)] + vrows["query_ref"][supported].astype(np.int64)
    total_q = int(n_queries.sum())
    nominal_mask = np.zeros(total_q, dtype=bool)
    nominal_mask[nominal_q] = True
    nominal_per_match = np.bincount(vrows["match_index"][supported].astype(np.int64), minlength=len(mq))
    out = {"dataset": dataset, "n_matches": int(len(mq)), "n_queries": total_q, "n_nominal_rows": int(supported.sum())}

    # static categories (node) and bans (team)
    from studies.observable_state_20261002.schema import STATIC_NAMES

    static = np.load(projection / "STATIC_IDS.npy")
    bans = np.load(projection / "BANS.npy")
    tallies = {"per_match": Tally(), "all_queries": Tally(), "nominal_rows": Tally()}
    unseen_static = {}
    weights = {"per_match": np.ones(len(mq), dtype=np.int64), "all_queries": n_queries, "nominal_rows": nominal_per_match}
    blocks = [(f"static:{name}", f"static:{name}", static[:, :, index]) for index, name in enumerate(STATIC_NAMES)]
    blocks.append(("static:bans", "static:champion_id", bans.reshape(len(mq), -1)))
    for group, namespace, values in blocks:
        known = np.isin(values, np.asarray(sorted(v for v in vocab[namespace] if isinstance(v, int)), dtype=np.int64))
        nonpositive = values < 0
        unseen = ~known & ~nonpositive
        top = Counter()
        for value, count in zip(*np.unique(values[unseen], return_counts=True)):
            top[str(int(value))] = int(count)
        unseen_static[group] = dict(top.most_common(TOP))
        for scope, weight in weights.items():
            per_match_unseen = unseen.sum(axis=1)
            per_match_nonpos = nonpositive.sum(axis=1)
            entries = int(values.shape[1] * weight.sum())
            tallies[scope].add(group, entries, int((per_match_unseen * weight).sum()), int((per_match_nonpos * weight).sum()),
                               entries, int((per_match_unseen * weight).sum()), int((per_match_nonpos * weight).sum()))

    # current-item bags (item bank)
    offsets = np.load(items / "NODE_OFFSETS.npy")
    ids = np.fromfile(items / "ITEM_IDS.bin", dtype="<i8")
    counts = np.fromfile(items / "ITEM_COUNTS.bin", dtype="<i8")
    item_query = np.load(projection / "QUERY_TABLE.npy")["item_query"].astype(np.int64)
    if len(offsets) != total_q or not np.array_equal(item_query, np.arange(total_q)):
        raise ValueError(f"{dataset}: item/projection query order differs")
    if len(ids) != int(offsets[-1, -1]) or int(offsets[0, 0]) != 0 or np.any(offsets[1:, 0] != offsets[:-1, -1]):
        raise ValueError(f"{dataset}: item offsets are not contiguous")
    # element j lies in flat node k = query * 10 + participant with ends[k-1] <= j < ends[k]
    node_of = np.searchsorted(offsets[:, 1:].reshape(-1), np.arange(len(ids), dtype=np.int64), side="right")
    element_query, node_index = node_of // 10, node_of % 10
    keys = (element_query * 16 + node_index) * (1 << 32) + (ids & 0xFFFFFFFF)
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundary = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]] if len(keys) else np.zeros(0, dtype=bool)
    entry_of_sorted = np.cumsum(boundary) - 1
    entry_counts = np.bincount(entry_of_sorted, weights=counts[order]).astype(np.int64) if len(keys) else np.zeros(0, dtype=np.int64)
    entry_ids = ids[order][boundary]
    entry_query = element_query[order][boundary]
    item_vocab = np.asarray(sorted(v for v in vocab["item"] if isinstance(v, int)), dtype=np.int64)
    entry_nonpos = entry_ids < 0
    entry_unseen = ~np.isin(entry_ids, item_vocab) & ~entry_nonpos
    top_items = Counter()
    for value, count in zip(*np.unique(entry_ids[entry_unseen], return_counts=True)):
        top_items[str(int(value))] = int(count)
    for scope, mask in (("all_queries", np.ones(len(entry_ids), dtype=bool)), ("nominal_rows", nominal_mask[entry_query])):
        tallies[scope].add("item:current", int(mask.sum()), int((entry_unseen & mask).sum()), int((entry_nonpos & mask).sum()),
                           int(entry_counts[mask].sum()), int(entry_counts[entry_unseen & mask].sum()), int(entry_counts[entry_nonpos & mask].sum()))
    del ids, counts, keys, order, sorted_keys, element_query, node_of

    # history bags (shared-history store)
    nominal_by_shard = []
    tasks = []
    vocab_item = frozenset(v for v in vocab["item"])
    vocab_nonitem = frozenset(v for v in vocab["nonitem"])
    for receipt in acceptance["shards"]:
        a, b = (int(x) for x in receipt["match_range"])
        lo, hi = int(first_query[a]), int(first_query[b - 1] + n_queries[b - 1])
        q_ids = (np.flatnonzero(nominal_mask[lo:hi]) + lo).tolist()
        tasks.append((str(history / receipt["directory"]), str(history / "PLAN.json"), a, b, q_ids, vocab_item, vocab_nonitem))
        nominal_by_shard.append(len(q_ids))
    unseen_hist = {"item": Counter(), "nonitem": Counter()}
    n_records = 0
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(tasks)))) as pool:
        for result in pool.map(_history_task, tasks):
            n_records += result["n_records"]
            for scope, counter in result["tallies"].items():
                tally = Tally()
                tally.c.update({tuple(k) if isinstance(k, list) else k: v for k, v in counter.items()})
                tallies[scope].merge(tally)
            for name, counter in result["unseen"].items():
                unseen_hist[name].update(counter)
    if n_records != total_q:
        raise ValueError(f"{dataset}: history records {n_records} != queries {total_q}")

    def grouped(result: dict) -> dict:
        merged = {}
        for prefix in ("item", "nonitem"):
            rows = [v for k, v in result.items() if k.startswith(prefix + ":")]
            if rows:
                e = sum(r["entries"] for r in rows)
                ue = sum(r["unk_entries"] for r in rows)
                w = sum(r["weighted_tokens"] for r in rows)
                uw = sum(r["unk_weighted"] for r in rows)
                merged[prefix] = {"entries": e, "unk_entries": ue, "unk_share": round(ue / e, 6) if e else None,
                                  "weighted_tokens": w, "unk_weighted": uw, "unk_weighted_share": round(uw / w, 6) if w else None}
        return merged

    for scope, tally in tallies.items():
        result = tally.result()
        out[scope] = {"groups": result, "namespaces": grouped(result) if scope != "per_match" else None}
    out["top_unseen"] = {"static": unseen_static, "item_current_entries": dict(top_items.most_common(TOP)),
                         "item_history_weighted": dict(unseen_hist["item"].most_common(TOP)),
                         "nonitem_weighted": dict(unseen_hist["nonitem"].most_common(TOP))}
    out["seconds"] = round(time.perf_counter() - started, 1)
    log(f"[unk] {dataset}: {total_q} queries in {out['seconds']} s; item={out['all_queries']['namespaces'].get('item')} "
        f"nonitem={out['all_queries']['namespaces'].get('nonitem')}")
    return out


def run_unk(datasets, workers: int, log) -> dict:
    C.runtime_hygiene()
    vocab = _vocab_sets()
    report = {"version": C.VERSION, "kind": "E2a_UNK_CENSUS", "created_utc": C.utc_now(),
              "vocabulary_bundle": str(V_FINAL_BUNDLE),
              "vocabulary_files_sha256": {name: C.sha256_file(V_FINAL_BUNDLE / name) for name in ("MANIFEST.json", "vocab.json")},
              "vocabulary_sizes": {name: len(values) for name, values in vocab.items()},
              "rules": __doc__.split("README", 1)[1].split("Output:", 1)[0].strip(),
              "labels_or_scores_read": False, "datasets": {}}
    for dataset in datasets:
        report["datasets"][dataset] = census_dataset(dataset, vocab, workers, log)
    C.write_json(UNK_CENSUS, report)
    log(f"[unk] wrote {UNK_CENSUS}")
    return report
