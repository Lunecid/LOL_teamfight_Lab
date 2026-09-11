"""Label-variant registry (R2 label family): overrides, restoration, price perturbations, helpers."""
import dataclasses
import importlib.util
import json
import random
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from core.config import CFG, cfg
from core.contract import NODE_IDX
from gameplay import labels as L

ROOT = Path(__file__).resolve().parents[1]
TM = {i: (100 if i <= 5 else 200) for i in range(1, 11)}
ANCHOR = (5000.0, 5000.0)
TABLE = {"kills": 20.0, "assists": 45.0, "plates": 120.0, "tower_inner_turret": 730.0, "dragon": 0.0}

REQUIRED = ("micro_win", "kill_survival", "market_event_dz0", "market_event_dz150", "market_event_dz600",
            "market_event_dz900", "market_event_kills_only", "market_event_prices_x0.5", "market_event_prices_x2",
            "market_event_dragon_rule", "market_event_attr_radius_half")
# the y_<type> columns of corpus_shards_v33 (scripts/build_corpus_v3.py --extra-labels default)
STORED = ("market_event", "market_event@window", "market_lex", "market_lex@window", "attention_value_win")


def _script(name):
    spec = importlib.util.spec_from_file_location(f"_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cache(events):
    T = 3
    node = np.zeros((T, 10, len(NODE_IDX)), dtype=np.float32)
    node[:, :, NODE_IDX["alive"]] = 1.0
    return {"minute_ts": np.array([0, 60000, 120000], dtype=np.int64), "node_minute": node,
            "global_minute": np.zeros((T, 26), np.float32), "gold_team_minute": np.zeros((T, 2), np.float32),
            "events": events, "events_ts": np.array([e["timestamp"] for e in events], dtype=np.int64)}


def _interp(cache, ms):
    return cache["node_minute"][0], cache["global_minute"][0]


def _kill(ts, killer, victim, bounty, shutdown=0, pos=ANCHOR):
    return {"type": "CHAMPION_KILL", "timestamp": ts, "killerId": killer, "victimId": victim, "bounty": bounty,
            "shutdownBounty": shutdown, "assistingParticipantIds": [], "position": {"x": pos[0], "y": pos[1]}}


# A: blue kill (300 + 20) against red taking blue's inner turret (730): swing -410
SCENARIO_TOWER = [_kill(20000, 1, 6, 300),
                  {"type": "BUILDING_KILL", "timestamp": 30000, "teamId": 100, "buildingType": "TOWER_BUILDING",
                   "towerType": "INNER_TURRET", "laneType": "MID_LANE", "position": {"x": 5500, "y": 5000}}]
# B: kills traded (330 vs 20), red dragon: +310 with the fitted 0 g, +285 with the 25 g rule payout
SCENARIO_DRAGON = [_kill(20000, 1, 6, 310), _kill(25000, 6, 1, 0),
                   {"type": "ELITE_MONSTER_KILL", "timestamp": 30000, "killerTeamId": 200, "monsterType": "DRAGON",
                    "monsterSubType": "FIRE_DRAGON", "position": {"x": 5200, "y": 5000}}]
# C: blue kill at the centre (+320), red shutdown kill 3,000 u away (-1,020): inside D, outside D/2
SCENARIO_RADIUS = [_kill(20000, 1, 6, 300), _kill(22000, 7, 2, 300, shutdown=700, pos=(8000.0, 5000.0))]


@pytest.fixture
def v33_labels(monkeypatch, tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"pooled": TABLE}), encoding="utf-8")
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_TABLE", str(path))
    monkeypatch.setattr(cfg, "LABEL_TIE_POLICY", "drop", raising=False)
    monkeypatch.setattr(cfg, "LABEL_TYPE", "market_event")
    monkeypatch.setattr(cfg, "LABEL_EVENT_ATTRIBUTION", "engagement")
    monkeypatch.setattr(cfg, "LABEL_GOLD_DEADZONE", 300.0)
    monkeypatch.setattr(cfg, "LABEL_ATTRIBUTION_RADIUS_U", 0.0)
    monkeypatch.setattr(cfg, "CLUSTER_MAX_DIAMETER", 4264.0)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_NONKILL_SCALE", 1.0)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", {})

    def run(variant, events):
        return L.compute_label_variant(variant, _cache(events), TM, -1, engage_ts=10000, label_end_ts=40000,
                                       horizon_ms=30000, first_kill_ts=20000, last_kill_ts=30000,
                                       interp_node_global=_interp, anchor_xy=ANCHOR)
    return run


def test_registry_holds_every_required_variant_with_real_cfg_fields():
    names = {f.name for f in dataclasses.fields(CFG)}
    for n in ("market_event",) + REQUIRED:
        assert n in L.LABEL_VARIANTS, n
    for v in L.LABEL_VARIANTS.values():
        v.validate()
        for k in v.overrides_dict():
            assert k in names, (v.name, k)
    assert set(L.REQUIRED_LABEL_VARIANTS) == {"market_event", *REQUIRED}
    assert L.REQUIRED_LABEL_VARIANTS[0] == "market_event"
    assert L.get_label_variant("market_event").overrides == ()
    assert L.get_label_variant("market_event_dz0").overrides_dict() == {"LABEL_GOLD_DEADZONE": 0.0}
    assert L.get_label_variant("market_event_kills_only").overrides_dict() == {"LABEL_EVENT_PRICE_TABLE": ""}
    assert L.get_label_variant("market_event_attr_radius_half").overrides_dict() == {"LABEL_ATTRIBUTION_RADIUS_U": 2132.0}
    assert L.get_label_variant("market_event_prices_x0.5").overrides_dict() == {"LABEL_EVENT_PRICE_NONKILL_SCALE": 0.5}
    assert L.get_label_variant("market_event_prices_x2").overrides_dict() == {"LABEL_EVENT_PRICE_NONKILL_SCALE": 2.0}
    dragon = L.get_label_variant("market_event_dragon_rule")
    assert dragon.overrides_dict() == {"LABEL_EVENT_PRICE_OVERRIDES": {"dragon": L.DRAGON_RULE_TEAM_GOLD}}
    assert "UNVERIFIED" in dragon.verification and "wiki.leagueoflegends.com" in dragon.source
    per_member = L.get_label_variant("market_event_dragon_rule_per_member")
    assert per_member.overrides_dict() == {"LABEL_EVENT_PRICE_OVERRIDES": {"dragon": 5 * L.DRAGON_RULE_TEAM_GOLD}}
    assert not per_member.required and "UNVERIFIED" in per_member.verification
    with pytest.raises(KeyError):
        L.get_label_variant("no_such_variant")


def test_stored_variants_are_the_corpus_columns_without_overrides():
    assert set(L.STORED_LABEL_VARIANTS) == set(STORED)
    for n in STORED:
        v = L.get_label_variant(n)
        assert v.overrides == ()
        base, attr = L._split_label_type(v.cfg_label_type)
        assert base + ("@window" if attr == "window" else "") == n


def test_variant_definitions_round_trip_through_json():
    for v in L.LABEL_VARIANTS.values():
        d = json.loads(json.dumps(v.as_dict()))
        for k in ("label_type", "attribution", "overrides"):
            assert d[k] == v.as_dict()[k], (v.name, k)


def test_variant_cfg_rejects_unknown_fields_label_type_overrides_and_schemes():
    for bad in (L.LabelVariant(name="bad", label_type="market_event", overrides=(("NO_SUCH_FIELD", 1),)),
                L.LabelVariant(name="bad", label_type="market_event", overrides=(("LABEL_TYPE", "micro_win"),))):
        with pytest.raises(KeyError):
            with L.variant_cfg(bad):
                pass
    for bad in (L.LabelVariant(name="bad", label_type="nope"),
                L.LabelVariant(name="bad", label_type="market_event", attribution="map")):
        with pytest.raises(ValueError):
            with L.variant_cfg(bad):
                pass


def test_price_table_defaults_are_the_identity(monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_NONKILL_SCALE", 1.0)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", {})
    table = dict(TABLE)
    assert L._apply_price_table_variant(table) is table


def test_nonkill_scale_leaves_kill_prices_and_the_loaded_table_alone(monkeypatch):
    table = dict(TABLE)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_NONKILL_SCALE", 0.5)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", {"dragon": 25.0})
    out = L._apply_price_table_variant(table)
    assert out["kills"] == 20.0 and out["assists"] == 45.0
    assert out["plates"] == 60.0 and out["tower_inner_turret"] == 365.0
    assert out["dragon"] == 25.0                                     # override replaces verbatim, unscaled
    assert table == TABLE                                            # never mutates the cached table
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_NONKILL_SCALE", 1.0)
    monkeypatch.setattr(cfg, "LABEL_EVENT_PRICE_OVERRIDES", '{"dragon": 25}')
    assert L._apply_price_table_variant(table)["dragon"] == 25.0      # JSON string (LOL_CFG_OVERRIDES) accepted


def test_cfg_override_restores_values_and_removes_new_attributes(monkeypatch):
    monkeypatch.delattr(cfg, "LABEL_TIE_POLICY", raising=False)
    monkeypatch.setattr(cfg, "LABEL_GOLD_DEADZONE", 300.0)
    with pytest.raises(RuntimeError):
        with L.cfg_override({"LABEL_TIE_POLICY": "random", "LABEL_GOLD_DEADZONE": 900.0}):
            assert cfg.LABEL_TIE_POLICY == "random" and cfg.LABEL_GOLD_DEADZONE == 900.0
            raise RuntimeError("boom")
    assert not hasattr(cfg, "LABEL_TIE_POLICY")
    assert cfg.LABEL_GOLD_DEADZONE == 300.0


def test_market_event_through_the_registry_equals_plain_compute_label(v33_labels):
    for events in (SCENARIO_TOWER, SCENARIO_DRAGON, SCENARIO_RADIUS):
        plain = L.compute_label(_cache(events), TM, -1, engage_ts=10000, label_end_ts=40000, horizon_ms=30000,
                                first_kill_ts=20000, last_kill_ts=30000, interp_node_global=_interp, anchor_xy=ANCHOR)
        assert v33_labels("market_event", events) == plain


def test_dead_zone_and_price_variants_move_the_verdict_as_priced(v33_labels):
    ev = SCENARIO_TOWER
    assert v33_labels("market_event", ev) == 0                    # -410 beyond 300
    assert v33_labels("market_event_dz0", ev) == 0
    assert v33_labels("market_event_dz150", ev) == 0
    assert v33_labels("market_event_dz600", ev) == 1              # inside 600 -> refinement: blue has the kill
    assert v33_labels("market_event_dz900", ev) == 1
    assert v33_labels("market_event_kills_only", ev) == 1         # +300 not beyond 300 -> refinement
    assert v33_labels("market_event_prices_x0.5", ev) == 1        # 320 - 365 = -45 -> refinement
    assert v33_labels("market_event_prices_x2", ev) == 0          # 320 - 1,460
    assert v33_labels("micro_win", ev) == 1
    assert v33_labels("kill_survival", ev) == 1
    assert cfg.LABEL_GOLD_DEADZONE == 300.0 and cfg.LABEL_TYPE == "market_event"
    assert cfg.LABEL_EVENT_PRICE_NONKILL_SCALE == 1.0 and cfg.LABEL_EVENT_PRICE_OVERRIDES == {}


def test_dragon_rule_and_attribution_radius_variants(v33_labels):
    assert v33_labels("market_event", SCENARIO_DRAGON) == 1            # +310 with the fitted 0 g dragon
    assert v33_labels("market_event_dragon_rule", SCENARIO_DRAGON) == 0  # +285 -> kills even -> red dragon
    assert v33_labels("market_event_dragon_rule_per_member", SCENARIO_DRAGON) == 0  # +185 -> refinement
    assert v33_labels("market_event", SCENARIO_RADIUS) == 0            # far shutdown inside D
    assert v33_labels("market_event_attr_radius_half", SCENARIO_RADIUS) == 1
    assert v33_labels("market_event@window", SCENARIO_RADIUS) == 0
    assert v33_labels("micro_win", SCENARIO_RADIUS) is None            # one kill each inside D
    assert cfg.LABEL_ATTRIBUTION_RADIUS_U == 0.0 and cfg.LABEL_EVENT_PRICE_OVERRIDES == {}
    for n in L.LABEL_VARIANTS:                                         # every scheme runs and leaves cfg alone
        v33_labels(n, SCENARIO_TOWER)
    assert cfg.LABEL_TYPE == "market_event" and cfg.LABEL_EVENT_PRICE_TABLE.endswith("prices.json")


def test_weighted_bootstrap_auc_equals_the_concatenated_resample():
    fam = _script("run_label_family_v33")
    rng = np.random.default_rng(3)
    n = 900
    groups = np.array([f"M{int(i)}" for i in rng.integers(0, 120, n)])
    y = rng.integers(0, 2, n).astype(np.int8)
    pred = np.round(rng.random(n) * 20) / 20 + 0.1 * y               # heavy ties
    mask = rng.random(n) < 0.4
    scores = fam.SortedScores(pred)
    unique, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))
    draws = np.random.default_rng(fam.SEED)
    for w in fam.match_bootstrap_weights(groups, n_boot=5):
        sample = draws.integers(0, len(unique), size=len(unique))      # run_scale_decomposition.cluster_bootstrap
        idx = np.concatenate([order[starts[g]:ends[g]] for g in sample])
        assert fam.weighted_auc(scores, y, w) == pytest.approx(roc_auc_score(y[idx], pred[idx]), abs=1e-12)
        sub = idx[mask[idx]]
        assert fam.weighted_auc(scores, y, w * mask) == pytest.approx(roc_auc_score(y[sub], pred[sub]), abs=1e-12)
        assert float((w * mask).sum()) == len(sub)


def test_identity_check_accepts_equal_rows_and_rejects_any_difference():
    side_mod = _script("build_label_sidecars_v33")
    corpus = {"groups": np.array(["A", "A", "B", "C"]), "engage_ts": np.array([1, 2, 3, 4], np.int64),
              "patch": np.array(["15.14"] * 4), "y": np.array([1, 0, 1, 1], np.int8),
              "y_market_event": np.array([1, -1, 1, 0], np.int8)}
    for k in ("cluster_blue", "cluster_red", "present_blue", "present_red"):
        corpus[k] = np.array([2, 3, 4, 5], np.int16)
    side = {k: v.copy() for k, v in corpus.items() if k != "y"}
    side["y_row"] = corpus["y"].copy()
    assert side_mod.verify_identity(side, corpus, prefix=False, stored_variants=["market_event"])["ok"]
    head = {k: v[:3] for k, v in side.items()}
    assert side_mod.verify_identity(head, corpus, prefix=True, included_matches=["A", "B"], stored_variants=["market_event"])["ok"]
    with pytest.raises(side_mod.SidecarIdentityError):
        side_mod.verify_identity({k: v[:1] for k, v in side.items()}, corpus, prefix=True, included_matches=["A"],
                                 stored_variants=["market_event"])       # match A continues after the prefix
    for key in ("engage_ts", "y_market_event", "y_row", "groups"):
        bad = {k: v.copy() for k, v in side.items()}
        bad[key][2] = bad[key][1]
        with pytest.raises(side_mod.SidecarIdentityError):
            side_mod.verify_identity(bad, corpus, prefix=False, stored_variants=["market_event"])
    with pytest.raises(side_mod.SidecarIdentityError):
        side_mod.verify_identity(side, corpus, prefix=False, stored_variants=[])


def test_price_table_check_compares_prices_not_bytes(monkeypatch):
    side_mod = _script("build_label_sidecars_v33")
    now = json.load(open(ROOT / "config/game_rules/event_prices.json", encoding="utf-8"))["pooled"]
    manifest = {"git_commit": "0" * 40, "label": {"price_table_sha1": "0" * 40}}
    monkeypatch.setattr(side_mod, "_pooled_prices_at", lambda commit: dict(now))
    report = side_mod.check_price_table(manifest)
    assert report["pooled_equal"] and not report["same_file"]
    monkeypatch.setattr(side_mod, "_pooled_prices_at", lambda commit: {**now, "dragon": 25.0})
    assert not side_mod.check_price_table(manifest)["pooled_equal"]
    monkeypatch.setattr(side_mod, "_pooled_prices_at", lambda commit: None)
    assert not side_mod.check_price_table(manifest)["pooled_equal"]


def test_shard_partition_matches_the_original_inline_rule():
    bcs = _script("build_corpus_shard")
    mids = [f"KR_{random.Random(1).randrange(10**9) + i}" for i in range(257)]
    random.Random(2).shuffle(mids)
    for n_matches in (None, 100):
        expect = sorted(mids)
        if n_matches and n_matches < len(expect):
            expect = sorted(random.Random(7).sample(expect, n_matches))
        shards = [bcs.shard_match_ids(mids, s, 8, n_matches=n_matches, seed=7) for s in range(8)]
        for s in range(8):
            assert shards[s] == [m for i, m in enumerate(expect) if i % 8 == s]
        assert sorted(sum(shards, [])) == expect
