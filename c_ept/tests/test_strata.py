"""Synthetic CPU-only tests of the evaluation strata and the price sensitivity (S4b preparation M13; decisions D18, D19, B17a). No match data: the champion strata read the pinned Data Dragon
rule tables through ``traits.trait_signature`` and are checked against the JSON of the tables themselves, the other tests use hand-made arrays and the toy worlds."""
import json

import numpy as np
import pytest

from c_ept import config as C
from c_ept.evaluation import strata
from c_ept.preprocessing import traits

REFERENCE = '15.20'
TAGS = ('Assassin', 'Fighter', 'Mage', 'Marksman', 'Support', 'Tank')
# Decision D18, evidence of the rule tables only (versus 15.20): 15.21 has 8 changed champions, 15.22 has 13, 16.13 has 170 of 173 plus 2 that 15.20 does not have.
D18_CHANGED = {'15.21': 8, '15.22': 13, '16.13': 170}
D18_UNSEEN_16_13 = 2


def table_of(patch: str) -> dict:
    """The champions of the Data Dragon table of a patch as the JSON file holds them (``{key: entry}``), without the mode ids (keys from 10,000)."""
    path = C.LEGACY_ROOT / 'config' / 'game_rules' / 'datadragon_v2' / f'champions_{patch}.json'
    return {int(k): v for k, v in json.loads(path.read_text(encoding='utf-8'))['champions'].items() if int(k) < 10_000}


def raw_signature(entry: dict) -> tuple:
    """What identifies a champion in a patch without its level, read from the JSON entry with no help of ``traits``: the six tag flags, ranged or not, the range, the resource class, the six base stats
    and the five growth stats."""
    s = entry['stats']
    reach = entry.get('attackrange', s['attackrange'])
    partype = (entry.get('partype') or '').strip()
    resource = partype if partype in ('Mana', 'Energy') else 'none' if partype in ('None', '') else 'other'
    return (tuple(float(t in entry['tags']) for t in TAGS), float(reach > 300), float(reach), resource,
            (float(s['hp']), float(s['armor']), float(s['spellblock']), float(s['attackdamage']), float(s['attackspeed']), float(s['movespeed'])),
            (float(s['hpperlevel']), float(s['armorperlevel']), float(s['spellblockperlevel']), float(s['attackdamageperlevel']), float(s['attackspeedperlevel'])))


@pytest.fixture(scope='module')
def tables():
    return {p: table_of(p) for p in ('15.20', '15.21', '15.22', '16.13')}


@pytest.fixture(scope='module')
def changed_sets(tables):
    """Per patch, the champions that both 15.20 and the patch have and whose signature differs, and those that only the patch has: derived from the JSON."""
    ref = tables[REFERENCE]
    return {p: ({k for k in t if k in ref and raw_signature(t[k]) != raw_signature(ref[k])}, {k for k in t if k not in ref}) for p, t in tables.items() if p != REFERENCE}


def one_champion_rows(champs) -> np.ndarray:
    """One row of ten copies of the champion for every champion: the row is changed (or not, or unseen) exactly when the champion is."""
    return np.repeat(np.asarray(sorted(champs), dtype=np.int32)[:, None], 10, axis=1)


# ---- D18: the changed-champion stratum ------------------------------------------------------------------------------------------------------
def test_changed_rows_flag_any_changed_champion(tables, changed_sets):
    """The premise (the numbers of the evidence of D18, from the JSON of the rule tables) and then the rule: a row is changed when a champion of it has another signature in its patch than in 15.20. One row
    of ten copies for each champion of the table of the patch; also a row of nine unchanged champions and one changed one is changed (any), and the same row without the changed one is not."""
    for patch, n_changed in D18_CHANGED.items():
        changed, _ = changed_sets[patch]
        assert len(changed) == n_changed, patch                                  # 8, 13 and 170: the evidence of D18
        champs = sorted(k for k in tables[patch] if k in tables[REFERENCE])
        rows = one_champion_rows(champs)
        flags = strata.changed_champion_rows(rows, np.full(len(rows), patch))
        assert flags.dtype == bool and flags.shape == (len(rows),)
        assert [c for c, f in zip(champs, flags) if f] == sorted(changed), patch
    changed, _ = changed_sets['15.22']
    unchanged = sorted(set(tables['15.22']) - changed)[:9]
    one = sorted(changed)[0]
    rows = np.array([unchanged + [one], unchanged + [unchanged[0]], [one] + unchanged], dtype=np.int32)
    assert strata.changed_champion_rows(rows, np.full(3, '15.22')).tolist() == [True, False, True]


def test_the_rows_of_the_reference_patch_are_never_changed_and_the_patch_may_be_given_as_a_number(tables, changed_sets):
    """Against itself nothing changed. ``matches/patch`` of the S1 store holds the number of the patch in ``config.PATCHES``: the same rows given that way are the same flags."""
    champs = sorted(tables[REFERENCE])
    rows = one_champion_rows(champs)
    assert not strata.changed_champion_rows(rows, np.full(len(rows), REFERENCE)).any()
    changed, _ = changed_sets['15.22']
    champs = sorted(k for k in tables['15.22'] if k in tables[REFERENCE])
    rows = one_champion_rows(champs)
    by_text = strata.changed_champion_rows(rows, np.full(len(rows), '15.22'))
    by_number = strata.changed_champion_rows(rows, np.full(len(rows), C.PATCHES.index('15.22'), dtype=np.int8))
    assert np.array_equal(by_text, by_number) and by_text.sum() == len(changed)
    mixed = strata.changed_champion_rows(rows[:4], np.array(['15.22', '15.20', '15.22', '15.21']))             # a patch for each row
    assert mixed.shape == (4,)
    assert strata.changed_champion_rows(rows[:0], np.array([], dtype='U5')).shape == (0,)                       # no rows, no flag


def test_unseen_champion_is_its_own_flag(tables, changed_sets):
    """A champion that 15.20 does not have (the two new champions of 16.13) or that the table of its own patch does not hold (the key 0 of an empty slot, a mode id) has no signature to compare: the row is
    ``unseen`` and it is neither ``changed`` nor ``unchanged``, whatever the other nine champions are. The three flags partition the rows."""
    changed, unseen = changed_sets['16.13']
    assert len(unseen) == D18_UNSEEN_16_13
    ref_known = sorted(k for k in tables['16.13'] if k in tables[REFERENCE])
    still = [k for k in ref_known if k not in changed]                       # the one champion of 16.13 that did not change since 15.20
    assert still, 'the premise: 16.13 keeps at least one champion unchanged'
    new = sorted(unseen)[0]
    rows = np.array([[new] * 10, [new] + sorted(changed)[:9], [new] + still * 9, [0] + still * 9, [9999] + sorted(changed)[:9], still * 10, sorted(changed)[:10]], dtype=np.int32)
    patches = np.full(len(rows), '16.13')
    flags = strata.champion_strata(rows, patches)
    assert flags.unseen.tolist() == [True, True, True, True, True, False, False]
    assert flags.changed.tolist() == [False, False, False, False, False, False, True]       # an unseen champion is never folded into 'changed' ...
    assert flags.unchanged.tolist() == [False, False, False, False, False, True, False]     # ... or into 'unchanged'
    assert (flags.changed.astype(int) + flags.unchanged + flags.unseen == 1).all()
    assert strata.changed_champion_rows(rows, patches).tolist() == flags.changed.tolist()
    assert strata.unseen_champion_rows(rows, patches).tolist() == flags.unseen.tolist()
    every = one_champion_rows(sorted(tables['16.13']))
    parts = strata.champion_strata(every, np.full(len(every), '16.13'))
    assert (parts.changed.sum(), parts.unseen.sum(), parts.unchanged.sum()) == (170, 2, 1) and len(every) == 173


def test_a_champion_that_the_table_of_its_own_patch_lacks_is_unseen_even_when_the_reference_has_it(monkeypatch):
    """Unseen is "no signature on either side": a champion that the reference patch has and its own patch has not (a removed champion; no real table has one, so the signature is faked here) cannot be compared either,
    and its row is neither changed nor unchanged."""
    real = traits.trait_signature
    monkeypatch.setattr(traits, 'trait_signature', lambda patch, champ: None if (patch, champ) == ('15.22', 2) else real(patch, champ))
    rows = np.array([[1] * 10, [2] + [1] * 9, [2] * 10], dtype=np.int32)
    flags = strata.champion_strata(rows, np.full(3, '15.22'))
    assert flags.unseen.tolist() == [False, True, True] and flags.changed.tolist() == [False, False, False] and flags.unchanged.tolist() == [True, False, False]
    assert strata.champion_strata(rows, np.full(3, '15.20')).unseen.tolist() == [False, False, False]                # the reference patch itself still has it


def test_the_strata_refuse_what_they_cannot_read():
    rows, patches = one_champion_rows([1, 2]), np.array(['15.22', '15.22'])
    with pytest.raises(ValueError, match='match_champ'):
        strata.changed_champion_rows(rows[:, :9], patches)
    with pytest.raises(ValueError, match='match_patch'):
        strata.changed_champion_rows(rows, patches[:1])
    with pytest.raises(TypeError, match='match_champ'):
        strata.changed_champion_rows(rows.astype(float), patches)
    with pytest.raises(ValueError, match='patch'):
        strata.changed_champion_rows(rows, np.array(['15.22', '99.99']))
    with pytest.raises(ValueError, match='patch'):
        strata.changed_champion_rows(rows, np.array([4, 99]))                    # a number outside config.PATCHES
    with pytest.raises(ValueError, match='reference'):
        strata.changed_champion_rows(rows, patches, reference='99.99')
    with pytest.raises(TypeError, match='reference'):
        strata.changed_champion_rows(rows, patches, reference=1520)


# ---- B17a: the rows whose t0 truth holds objectives from [tau, t0] ----------------------------------------------------------------------------
def test_pre_t0_objective_rows_are_the_rows_with_an_onset_whose_tau_is_at_or_before_t0():
    """B17a: the truth of a t0 row contains the objectives of [tau, t0] when an engagement of its onsets has tau <= t0 < f (tau = f - 15 s). By hand, in whole ms:
        row | t0      | onsets (unit: f, tau)                          | tau <= t0 < f
         0  |  60,000 | u0: 70,000, 55,000                             | yes
         1  | 120,000 | u1: 135,000, 120,000 (tau = t0)  u2: 150,000, 135,000 | yes (the boundary counts)
         2  | 180,000 | u3: 200,000, 185,000                           | no (tau after t0)
         3  | 240,000 | u4: 255,001, 240,001                           | no (one ms too late)
         4  | 300,000 | none                                           | no
         5  | 360,000 | u5: 360,000, 345,000                           | no (f = t0: not after t0)"""
    t0 = np.array([60_000, 120_000, 180_000, 240_000, 300_000, 360_000], dtype=np.int64)
    onset_row, onset_unit = np.array([0, 1, 1, 2, 3, 5]), np.array([0, 1, 2, 3, 4, 5])
    f = np.array([70_000, 135_000, 150_000, 200_000, 255_001, 360_000], dtype=np.int64)
    tau = f - 15_000
    got = strata.pre_t0_objective_rows(t0, onset_row, onset_unit, tau, f)
    assert got.dtype == bool and got.tolist() == [True, True, False, False, False, False]
    # a unit that two rows share is read for each row on its own t0; an onset order that is not sorted changes nothing
    order = np.array([5, 3, 0, 4, 2, 1])
    assert strata.pre_t0_objective_rows(t0, onset_row[order], onset_unit[order], tau, f).tolist() == got.tolist()
    assert strata.pre_t0_objective_rows(t0, onset_row[:0], onset_unit[:0], tau, f).tolist() == [False] * 6        # no onsets, no pre-t0 objectives


def test_pre_t0_objective_rows_refuse_what_they_cannot_read():
    t0, row, unit = np.array([60_000], dtype=np.int64), np.array([0]), np.array([0])
    tau, f = np.array([55_000], dtype=np.int64), np.array([70_000], dtype=np.int64)
    assert strata.pre_t0_objective_rows(t0, row, unit, tau, f).tolist() == [True]
    for bad in (t0.astype(float), t0.astype(bool)):
        with pytest.raises(TypeError, match='t0_ms'):
            strata.pre_t0_objective_rows(bad, row, unit, tau, f)
    with pytest.raises(TypeError, match='unit_tau_ms'):
        strata.pre_t0_objective_rows(t0, row, unit, tau.astype(float), f)
    with pytest.raises(ValueError, match='onset_row'):
        strata.pre_t0_objective_rows(t0, np.array([1]), unit, tau, f)            # a row that the table does not have
    with pytest.raises(ValueError, match='onset_unit'):
        strata.pre_t0_objective_rows(t0, row, np.array([1]), tau, f)             # a unit that the table does not have
    with pytest.raises(ValueError, match='onset'):
        strata.pre_t0_objective_rows(t0, np.array([0, 0]), unit, tau, f)         # row and unit of different lengths
    with pytest.raises(ValueError, match='unit_f_ms'):
        strata.pre_t0_objective_rows(t0, row, unit, tau, np.array([70_000, 1], dtype=np.int64))


# ---- D19: the sensitivity "all flow prices" -------------------------------------------------------------------------------------------------
def test_flow_sensitivity_pairs_draws_with_the_primary():
    """The second composer has the seed, the number of simulations, the rows, the tier law, the gold representatives and the settings of the primary one and only other prices, so every random draw of the
    chain (counts, marks, zones, sizes, kills, gold bins, objects, conversions) is the same draw and only the value of an onset moves: by hand, a dragon that is worth 50 more is worth 50 more in every
    onset for each dragon that blue took and 50 less for each dragon that red took, objects and conversions together. The primary composer is not changed by the making of the second."""
    from c_ept import config as C
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, ToyHeads, toy_composer
    primary = toy_composer(S=300, seed=3)
    n = len(primary.rows)
    before = [primary.simulate_onsets(i) for i in range(n)]
    flow = {**TOY_COMPOSE_PRICES, 'dragon': TOY_COMPOSE_PRICES['dragon'] + 50.0}
    second = strata.flow_composer(primary, flow, None, ToyHeads())
    assert second is not primary and second.seed == primary.seed == 3 and second.S == primary.S == 300 and second.rows is primary.rows
    assert second.tier_model is primary.tier_model and np.array_equal(second.gold_reps, primary.gold_reps) and second.prices['dragon'] == primary.prices['dragon'] + 50.0
    assert (second.kill_law, second.zone_mark, second.exposure) == (primary.kill_law, primary.zone_mark, primary.exposure)
    dragon = 2 * C.O_KINDS.index('dragon')
    moved = 0
    for i in range(n):
        a, b = before[i], second.simulate_onsets(i)
        for field in ('sim', 'q', 'm', 'z', 's', 'k', 'gold_bin', 'gold', 'O', 'conv'):
            assert np.array_equal(getattr(a, field), getattr(b, field)), (i, field)           # the draws are paired
        taken = a.O + a.conv
        assert np.allclose(b.value - a.value, 50.0 * (taken[:, dragon] - taken[:, dragon + 1])), i
        moved += int((b.value != a.value).sum())
        assert np.array_equal(primary.simulate_onsets(i).value, a.value)                        # the primary one gives what it gave
    assert moved > 0, 'the control: some onset holds a dragon, so the prices did move a value'
    with pytest.raises(ValueError, match='plate'):
        strata.flow_composer(primary, {k: v for k, v in flow.items() if k != 'plate'}, None, ToyHeads())


def test_flow_sensitivity_takes_its_own_chain_of_heads():
    """A chain of heads is bound to the rows of one composer. The second composer is given a chain of its own: the primary chain is not bound again (its probe would notice), and the same chain is refused."""
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, ToyHeads, toy_composer
    primary = toy_composer(S=50, seed=4)
    bound = primary.heads.n_bind
    strata.flow_composer(primary, TOY_COMPOSE_PRICES, None, ToyHeads())
    assert primary.heads.n_bind == bound
    with pytest.raises(ValueError, match='heads'):
        strata.flow_composer(primary, TOY_COMPOSE_PRICES, None, primary.heads)


def test_the_agreement_of_the_winner_signs_and_the_rank_correlation_of_v_start_by_hand():
    """D19 reports the share of units whose winner sign agrees (the sign of f0 rounded to a millionth of a gold, as ``winner_of``; a draw is a sign too) and the rank correlation of the realised V_start
    (Spearman; ties get their average rank). Rows that are NaN on either side are left out of both.
    W: f0 100 / -50 / 0 / 1e-9 / 200 against 10 / 5 / 0 / -1e-9 / -3 agree on 3 of 5 units (signs 1,-1,0,0,1 against 1,1,0,0,-1); a sixth unit is NaN.
    rho: (1, 2, 3, 4) against (1, 3, 2, 4) is 1 - 6 * 2 / (4 * 15) = 0.8; with the tie (1, 1, 2) against (1, 2, 3) the ranks (1.5, 1.5, 3) and (1, 2, 3) give 1.5 / sqrt(1.5 * 2); with the tie
    (1, 1, 2, 3) against (1, 2, 3, 4) the ranks (1.5, 1.5, 3, 4) and (1, 2, 3, 4) give 4.5 / sqrt(4.5 * 5) = sqrt(0.9) (the first rank of a tie would give 0.9467)."""
    a = np.array([100.0, -50.0, 0.0, 1e-9, 200.0, np.nan])
    b = np.array([10.0, 5.0, 0.0, -1e-9, -3.0, 7.0])
    assert strata.winner_sign_agreement(a, b) == pytest.approx(3 / 5)
    assert strata.winner_sign_agreement(a, a) == 1.0 and strata.winner_sign_agreement(a, b) == strata.winner_sign_agreement(b, a)
    assert np.isnan(strata.winner_sign_agreement(np.array([np.nan]), np.array([1.0]))) and np.isnan(strata.winner_sign_agreement(np.empty(0), np.empty(0)))
    assert strata.v_start_rank_correlation(np.array([1., 2., 3., 4.]), np.array([1., 3., 2., 4.])) == pytest.approx(0.8)
    assert strata.v_start_rank_correlation(np.array([1., 1., 2.]), np.array([1., 2., 3.])) == pytest.approx(1.5 / np.sqrt(3.0))
    assert strata.v_start_rank_correlation(np.array([1., 1., 2., 3.]), np.array([1., 2., 3., 4.])) == pytest.approx(np.sqrt(0.9))
    x, y = np.array([1., 2., 3., 4., np.nan, 9.]), np.array([1., 3., 2., 4., 5., np.nan])
    assert strata.v_start_rank_correlation(x, y) == pytest.approx(0.8) and strata.v_start_rank_correlation(x, -y) == pytest.approx(-0.8)
    assert np.isnan(strata.v_start_rank_correlation(np.array([1., 1., 1.]), np.array([1., 2., 3.]))) and np.isnan(strata.v_start_rank_correlation(np.array([1.]), np.array([2.])))
    with pytest.raises(ValueError, match='one length'):
        strata.winner_sign_agreement(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match='one length'):
        strata.v_start_rank_correlation(np.zeros(3), np.zeros(2))


def test_flow_sensitivity_refuses_a_chain_of_heads_with_another_chunk_views_than_the_primary():
    """M14 review I1: the chunk size moves the head outputs at the last digits of float32 and a Composer compares uniforms with cumulative probabilities, so a flow chain that encodes in other chunks than the
    primary chain could flip a draw and mix sampling noise into the paired difference. The same chunk_views is accepted (None and None too), another one is refused before the chain is bound, and heads
    that have no chunk_views (B0, B1, the toy heads) are not affected."""
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, ToyHeads, toy_composer

    def chain(chunk_views='absent'):
        h = ToyHeads()
        if chunk_views != 'absent':
            h.chunk_views = chunk_views
        return h
    for primary_views, flow_views in ((256, 256), (None, None), (1, 1)):
        primary = toy_composer(S=20, seed=4)
        primary.heads.chunk_views = primary_views
        assert strata.flow_composer(primary, TOY_COMPOSE_PRICES, None, chain(flow_views)).seed == 4
    for primary_views, flow_views in ((256, 128), (256, None), (None, 256), (256, 'absent'), ('absent', 256)):
        primary = toy_composer(S=20, seed=4)
        if primary_views != 'absent':
            primary.heads.chunk_views = primary_views
        flow = chain(flow_views)
        bound = flow.n_bind
        with pytest.raises(ValueError, match='chunk_views'):
            strata.flow_composer(primary, TOY_COMPOSE_PRICES, None, flow)
        assert flow.n_bind == bound                                                                  # refused before the chain was bound
    primary = toy_composer(S=20, seed=4)
    assert not hasattr(primary.heads, 'chunk_views') and strata.flow_composer(primary, TOY_COMPOSE_PRICES, None, chain()).S == 20


def test_flow_sensitivity_with_the_real_ept_heads_needs_the_same_chunk_views(tmp_path):
    """The same refusal with the real EPT composer heads: the same chunk size gives a flow composer whose draws are those of the primary one, another size (or none) is refused before anything is encoded."""
    import torch
    from c_ept.ept import data
    from c_ept.ept.compose_heads import EPTComposerHeads
    from c_ept.tests.s3_toys import TOY_COMPOSE_PRICES, TOY_GOLD_REPS, toy_tier_model
    from c_ept.tests.test_ept_compose import build_compose_world, model
    from c_ept.valuation import compose
    world = build_compose_world(tmp_path)
    stats = data.fit_stats(world.s1, world.s2)
    rows = compose.row_table(world.s1, world.s2, ('FIT', 'STOP', 'DEV'))
    network = model(seed=17)

    def chain(chunk_views):
        return EPTComposerHeads(network, stats, chunk_views=chunk_views)
    primary = compose.Composer(chain(3), TOY_COMPOSE_PRICES, toy_tier_model(), TOY_GOLD_REPS, rows, world.s2, S=10, seed=3)
    second = strata.flow_composer(primary, TOY_COMPOSE_PRICES, world.s2, chain(3))
    assert second.heads.chunk_views == primary.heads.chunk_views == 3
    for other in (chain(4), chain(None)):
        with pytest.raises(ValueError, match='chunk_views'):
            strata.flow_composer(primary, TOY_COMPOSE_PRICES, world.s2, other)
        assert other._enc is None