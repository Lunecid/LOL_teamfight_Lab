"""q inputs: the raw store and vectorized transform reproduce common.transform_rows exactly; group masks."""
import hashlib
import json

import numpy as np
import pytest

from studies.observable_state_20261002.kernel import STATIC_NAMES
from studies.q_production_20261005 import features as qf
from studies.v_model_fit_input_20261003.common import ITEM_GROUPS, transform_rows


def _token(event, field="event", measure="count", value=1):
    return json.dumps([event, field, measure, ["int", value]], ensure_ascii=False, separators=(",", ":"))


NODE_TOKENS = [_token("LEVEL_UP", "level", "value", 3), _token("CHAMPION_KILL", "killer", "credit"),
               _token("SKILL_LEVEL_UP", "skillSlot", "value", 2), _token("BUILDING_KILL"), _token("LEVEL_UP", "level", "value", 4)]
TEAM_TOKENS = [_token("CHAMPION_KILL"), _token("ELITE_MONSTER_KILL"), _token("TURRET_PLATE_DESTROYED")]


def _row(seed):
    rng = np.random.default_rng(seed)
    numeric = {"node": rng.normal(size=(10, 102)) * 3, "team": rng.normal(size=(2, 8)), "global": rng.normal(size=75)}
    numeric["node"][seed % 10, 3] = np.nan
    numeric["node"][:, 7] = 0.0
    categories = {"node": {name: rng.integers(-1, 6, size=10) for name in STATIC_NAMES},
                  "team": {f"ban{k}": rng.integers(-1, 6, size=2) for k in range(5)}, "global": {}}

    def items(slots, empty_rate=0.4):
        return [{} if rng.random() < empty_rate else {int(i): int(rng.integers(1, 3)) for i in rng.choice(np.arange(1000, 1012), size=rng.integers(1, 4), replace=False)}
                for _ in range(slots)]

    def tokens(pool, slots):
        return [{t: int(rng.integers(1, 5)) for t in rng.choice(pool, size=rng.integers(0, len(pool) + 1), replace=False)} for _ in range(slots)]

    bags = {"node": {"current_item": items(10), **{group: items(10) for group in ITEM_GROUPS}, "nonitem": tokens(NODE_TOKENS, 10)},
            "team": {"nonitem": tokens(TEAM_TOKENS, 2)},
            "global": {**{group: items(1) for group in ITEM_GROUPS}, "nonitem": tokens(TEAM_TOKENS + [_token("DRAGON_SOUL_GIVEN")], 1)}}
    teams = np.asarray([100, 200] * 5)
    return {"numeric": numeric, "categories": categories, "bags": bags, "team_ids": teams, "focal_is_blue": True,
            "current_item_supported": rng.random(10) < 0.7}


def _store(tmp_path, rows, name="store"):
    writer = qf.RawStoreWriter(tmp_path / name, np.arange(len(rows), dtype=np.int64) * 3 + 5)
    for i, row in enumerate(rows):
        writer.add(i * 3 + 5, row)
    writer.close()
    return qf.RawStore(tmp_path / name)


def _keys(n):
    return [hashlib.sha256(f"g{i}".encode()).hexdigest() for i in range(n)]


def _assert_batches_equal(a, b):
    for lane in ("node_numeric", "team_numeric", "global_numeric"):
        np.testing.assert_array_equal(getattr(a, lane), getattr(b, lane))
        assert getattr(a, lane).dtype == getattr(b, lane).dtype
    np.testing.assert_array_equal(a.player_team_ids, b.player_team_ids)
    np.testing.assert_array_equal(a.current_item_supported, b.current_item_supported)
    for lane in ("node_categories", "team_categories"):
        assert set(getattr(a, lane)) == set(getattr(b, lane))
        for group in getattr(a, lane):
            np.testing.assert_array_equal(getattr(a, lane)[group], getattr(b, lane)[group])
    for lane in ("node_bags", "team_bags", "global_bags"):
        assert set(getattr(a, lane)) == set(getattr(b, lane))
        for group, bag in getattr(a, lane).items():
            other = getattr(b, lane)[group]
            for field in ("indices", "counts", "offsets", "observed_count"):
                np.testing.assert_array_equal(np.asarray(getattr(bag, field)), np.asarray(getattr(other, field)), err_msg=f"{lane}.{group}.{field}")
    assert a.table_csr.shape == b.table_csr.shape and a.table_csr.dtype == b.table_csr.dtype
    np.testing.assert_array_equal(a.table_csr.indptr, b.table_csr.indptr)
    np.testing.assert_array_equal(a.table_csr.indices, b.table_csr.indices)
    np.testing.assert_array_equal(a.table_csr.data, b.table_csr.data)


def test_vectorized_transform_equals_common_transform_rows(tmp_path):
    rows = [_row(seed) for seed in range(8)]
    store = _store(tmp_path, rows)
    pre = qf.fit_scope(store, np.arange(6), "REFIT_TRAIN", _keys(6))
    _assert_batches_equal(qf.Transformer(store, pre)(np.arange(8)), transform_rows(rows, pre))
    _assert_batches_equal(qf.Transformer(store, pre)(np.asarray([7, 2, 5])), transform_rows([rows[7], rows[2], rows[5]], pre))


def test_fit_scope_uses_only_scope_rows_and_both_scopes_are_allowed(tmp_path):
    rows = [_row(seed) for seed in range(8)]
    store = _store(tmp_path, rows)
    fit = qf.fit_scope(store, np.asarray([0, 1, 2]), "FIT", _keys(3))
    refit = qf.fit_scope(store, np.arange(8), "REFIT_TRAIN", _keys(8))
    assert fit.evidence.row_indices == (0, 1, 2) and refit.evidence.scope == "REFIT_TRAIN"
    seen = set()
    for row in rows[:3]:
        for lane in ("node", "global"):
            for group in ITEM_GROUPS:
                for bag in row["bags"][lane][group]:
                    seen.update(bag)
        for bag in row["bags"]["node"]["current_item"]:
            seen.update(bag)
    assert set(fit.categorical_map["item"].categories) == seen
    numeric = np.stack([row["numeric"]["global"] for row in rows[:3]])
    np.testing.assert_allclose(fit.numeric_map["global"].mean, np.nanmean(np.where(np.isnan(numeric), np.nanmedian(numeric, axis=0), numeric), axis=0))


def test_merged_chunks_equal_one_store(tmp_path):
    rows = [_row(seed) for seed in range(8)]
    whole = _store(tmp_path, rows, "whole")
    for name, part in (("a", range(0, 5)), ("b", range(5, 8))):
        writer = qf.RawStoreWriter(tmp_path / name, np.asarray([i * 3 + 5 for i in part], dtype=np.int64))
        for i in part:
            writer.add(i * 3 + 5, rows[i])
        writer.close()
    qf.merge_chunks([tmp_path / "b", tmp_path / "a"], tmp_path / "merged")
    merged = qf.RawStore(tmp_path / "merged")
    pre = qf.fit_scope(whole, np.arange(6), "REFIT_TRAIN", _keys(6))
    pre_merged = qf.fit_scope(merged, np.arange(6), "REFIT_TRAIN", _keys(6))
    assert pre.categorical_map["nonitem"].categories == pre_merged.categorical_map["nonitem"].categories
    _assert_batches_equal(qf.Transformer(merged, pre_merged)(np.arange(8)), qf.Transformer(whole, pre)(np.arange(8)))


def test_group_masks_zero_or_remove_inactive_inputs_only(tmp_path):
    rows = [_row(seed) for seed in range(6)]
    store = _store(tmp_path, rows)
    pre = qf.fit_scope(store, np.arange(6), "REFIT_TRAIN", _keys(6))
    transform = qf.Transformer(store, pre)
    full = transform(np.arange(6))
    no_e = transform(np.arange(6), qf.GroupMask(E=False))
    assert not no_e.node_numeric[..., 0:6].any() and np.array_equal(no_e.node_numeric[..., 6:], full.node_numeric[..., 6:])
    level_up = pre.categorical_map["nonitem"].mapping
    level_codes = {level_up[t] for t in NODE_TOKENS if json.loads(t)[0] == "LEVEL_UP" and t in level_up}
    assert not set(np.asarray(no_e.node_bags["nonitem"].indices).tolist()) & level_codes
    no_c = transform(np.arange(6), qf.GroupMask(C=False))
    assert all(not v.any() for v in no_c.node_categories.values()) and not no_c.current_item_supported.any()
    assert all(len(np.asarray(no_c.node_bags[g].indices)) == 0 for g in ("current_item", *ITEM_GROUPS))
    assert not no_c.global_numeric[:, 26:75].any()
    no_map = transform(np.arange(6), qf.GroupMask(drop_map=True))
    assert not no_map.global_numeric[:, 8:26].any()
    objective = {pre.categorical_map["nonitem"].mapping[t] for t in TEAM_TOKENS if json.loads(t)[0] != "CHAMPION_KILL" and t in pre.categorical_map["nonitem"].mapping}
    assert not set(np.asarray(no_map.team_bags["nonitem"].indices).tolist()) & objective


def test_masked_inputs_do_not_depend_on_the_disabled_group(tmp_path):
    rows = [_row(seed) for seed in range(6)]
    changed = list(rows)
    changed[2] = {**rows[2], "numeric": {**rows[2]["numeric"], "node": rows[2]["numeric"]["node"].copy()}}
    changed[2]["numeric"]["node"][:, 0:6] += 5.0      # E channels change
    a, b = _store(tmp_path, rows, "a"), _store(tmp_path, changed, "b")
    pre = qf.fit_scope(a, np.arange(6), "REFIT_TRAIN", _keys(6))
    mask = qf.GroupMask(E=False)
    _assert_batches_equal(qf.Transformer(a, pre)(np.arange(6), mask), qf.Transformer(b, pre)(np.arange(6), mask))


def test_real_bank_rows_route_and_transform_like_v(tmp_path):
    from pathlib import Path
    engagements = Path(r"D:\LOL_Training_20261004\labels_1\ENGAGEMENTS.npz")
    if not engagements.is_file() or not (qf.ROOT / "outputs/shared_history_eight_cpu_20261003/full_1").is_dir():
        pytest.skip("label run or shared bank not present")
    with np.load(engagements, allow_pickle=False) as eng:
        train = np.flatnonzero(eng["role"] == 0)
        queries = np.sort(eng["query_pre"][train[:: max(1, len(train) // 40)][:40]].astype(np.int64))
    from studies.label_production_20261005.queries import PROJECTION, SHARED_BANK, QuerySurface
    surface = QuerySurface(SHARED_BANK, PROJECTION, seed=20261001, verify_projection_hashes=False)
    rows = [row for _q, row, _m in surface.iter_query_rows(queries)]
    qf.build_chunk(queries, tmp_path / "real", surface=surface)
    store = qf.RawStore(tmp_path / "real")
    pre = qf.fit_scope(store, np.arange(len(queries)), "REFIT_TRAIN", _keys(len(queries)))
    _assert_batches_equal(qf.Transformer(store, pre)(np.arange(len(queries))), transform_rows(rows, pre))
    for lane in ("node", "team", "global"):
        assert set(store.token_groups(lane)) <= set(qf.GROUPS)


def test_unrouted_nonitem_event_fails_closed():
    with pytest.raises(ValueError, match="unrouted"):
        qf.nonitem_group(_token("WARD_PLACED"), "node")
