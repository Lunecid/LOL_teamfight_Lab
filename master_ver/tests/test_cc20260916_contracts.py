"""Contract tests for the 2026-09-16 champion-class run (synthetic and TRAIN-only fixtures; no VALIDATION/TEST rows)."""
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import iq20260915_common as Q  # noqa: E402
import cc20260916_common as CC  # noqa: E402

SCRIPTS = sorted((ROOT / 'scripts').glob('cc20260916_*.py'))


def _names():
    nm = []
    for i in range(10):
        nm += [f'participant_slot{i}_{s}' for s in CC.SLOT_STATS] + [f'participant_slot{i}_champion_id', f'participant_slot{i}_other']
    return nm + ['time_minutes', 'p_pre_V']


def _table():
    rows = {'1': dict(id='A', name='A', primary='Assassin', secondary='Mage'), '2': dict(id='B', name='B', primary='Marksman', secondary=None),
            '3': dict(id='C', name='C', primary='Tank', secondary='Support'), '4': dict(id='D', name='D', primary='Fighter', secondary='Tank')}
    return {'table': {'15.14.1': rows, '16.13.1': dict(rows, **{'5': dict(id='E', name='E', primary='Mage', secondary=None)})}}


def _synthetic(n_matches=400, rows_per=(1, 6), seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(rows_per[0], rows_per[1], size=n_matches)
    g = np.repeat(np.array([f'KR_{i:07d}' for i in range(n_matches)]), counts)
    n = len(g)
    names = _names()
    X = rng.random((n, len(names)))
    ch, _ = CC.slot_columns(names)
    X[:, ch] = rng.integers(1, 6, size=(n, 10))
    for i in range(10):
        X[:, names.index(f'participant_slot{i}_alive')] = rng.integers(0, 2, n)
    X[:, names.index('time_minutes')] = rng.uniform(2, 40, n)
    logit = 2.0 * (X[:, names.index('participant_slot0_totalGold_norm')] - X[:, names.index('participant_slot5_totalGold_norm')])
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y, g, names


def test_registry_equals_iq_winners_arms_and_contrasts_fixed():
    reg, src = CC.registry()
    iq_fz = C.read_json(Q.frozen_path(CC.IQ))
    for coh in CC.COHORTS:
        for f in CC.FAMILIES:
            assert reg[coh][f] == iq_fz['family_winners'][f'{f}_{coh}']['config'] == Q.split_candidate(iq_fz['family_winners'][f'{f}_{coh}']['chosen'])[0]
    assert src['iq_frozen_manifest_sha256'] == C.sha256_file(Q.frozen_path(CC.IQ))
    assert CC.ARMS == ('base', 'tags', 'class_state', 'class_pairs', 'identity', 'class_pairs_identity', 'draft_class', 'draft_identity')
    assert len(CC.NAMED) == 16 and len(CC.CONTRASTS) == 20 and sum(c[2].startswith('PRIMARY') for c in CC.CONTRASTS) == 1
    assert CC.PRIMARY[:2] == ('lgbm_class_pairs', 'lgbm_base') and all(a in CC.NAMED and b in CC.NAMED for a, b, _ in CC.CONTRASTS)
    names = CC.candidate_names('lgbm_L15_M100')
    assert names == sorted(names) and len(names) == 3 and all(CC.split_candidate(n)[1] in CC.CALS for n in names)
    m = {'x__raw': (0.2, 0.6), 'x__isotonic': (0.2, 0.6), 'x__sigmoid': (0.2, 0.5999)}
    assert CC.select_rule(m)[0] == 'x__sigmoid'
    m['x__sigmoid'] = (0.2, 0.6)
    assert CC.select_rule(m)[0] == 'x__isotonic'


def test_extra_blocks_widths_names_and_hand_computed_values():
    names = _names()
    tab = _table()
    X = np.zeros((2, len(names)))
    ch, st = CC.slot_columns(names)
    X[0, ch] = [1, 2, 3, 4, 1, 2, 2, 2, 2, 2]
    X[1, ch] = [0, 99, 99, 99, 99, 99, 99, 99, 99, 99]
    X[0, st['totalGold_norm']] = [0.5, 0.2, 0.1, 0.4, 0.3, 1, 1, 1, 1, 1]
    X[0, st['hp_pct']] = [0.8, 0.6, 0.4, 0.2, 0.0, 1, 1, 1, 1, 1]
    X[0, st['alive']] = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
    vocab = [1, 2, 3, 4]
    ex = CC.extra_blocks(X, names, '15.14.1', vocab, table=tab)
    for b in CC.BLOCKS:
        nm, M = ex[b]
        assert len(nm) == M.shape[1] == (40 if b == 'identity' else CC.BLOCK_WIDTH[b]) and nm == CC.block_names(b, vocab)
    nm, M = ex['tags']
    assert M[0, nm.index('cc_slot0_primary_Assassin')] == 1 and M[0, nm.index('cc_slot0_secondary_Mage')] == 1 and M[0, nm.index('cc_slot1_secondary_Mage')] == 0
    assert M[0].sum() == 2 + 1 + 2 + 2 + 2 + 5 and M[1].sum() == 0
    nm, M = ex['class_count']
    assert M[0, nm.index('cc_blue_count_Assassin')] == 2 and M[0, nm.index('cc_red_count_Marksman')] == 5 and M[0, nm.index('cc_diff_count_Marksman')] == 1 - 5
    nm, M = ex['class_agg']
    assert np.isclose(M[0, nm.index('cc_blue_Assassin_totalGold_norm_sum')], 0.8) and np.isclose(M[0, nm.index('cc_blue_Assassin_hp_pct_mean')], 0.4)
    assert M[0, nm.index('cc_blue_Assassin_alive_sum')] == 2 and M[0, nm.index('cc_blue_Mage_hp_pct_mean')] == 0
    assert np.isclose(M[0, nm.index('cc_diff_Marksman_totalGold_norm_sum')], 0.2 - 5.0) and M[1].sum() == 0
    nm, M = ex['pairs']
    assert M[0].sum() == 11 and M[1].sum() == 0
    assert M[0, nm.index('cc_blue_pair_JG_MID_MarksmanxTank')] == 1 and M[0, nm.index('cc_matchup_TOP_AssassinxMarksman')] == 1
    assert M[0, nm.index('cc_red_pair_BOT_SUP_MarksmanxMarksman')] == 1 and M[0, nm.index('cc_blue_pair_BOT_SUP_FighterxAssassin')] == 1
    nm, M = ex['identity']
    assert M[0].sum() == 10 and M[1].sum() == 0 and M[0, nm.index('cc_slot0_champ_1')] == 1 and M[0, nm.index('cc_slot9_champ_2')] == 1
    X2 = X.copy()
    X2[1, ch] = 5
    assert CC.extra_blocks(X2, names, '15.14.1', vocab, tab)['tags'][1][1].sum() == 0
    assert CC.extra_blocks(X2, names, '16.13.1', vocab, tab)['tags'][1][1].sum() == 10
    assert CC.extra_blocks(X2, names, '16.13.1', vocab, tab)['identity'][1][1].sum() == 0


def test_arm_matrix_widths_draft_arms_multi_version_rows_and_determinism():
    X, y, g, names = _synthetic(60, seed=5)
    ridge = [n for n in names if not n.endswith('champion_id')]
    vocab = [1, 2, 3, 4]
    versions = np.where(np.arange(len(X)) % 2 == 0, '15.14.1', '16.13.1')
    for arm in CC.ARMS:
        M, nm = CC.arm_matrix(X, names, arm, ridge, versions, vocab, table=_table())
        assert M.shape == (len(X), len(nm)) and nm == CC.arm_names(arm, ridge, vocab) and len(nm) == CC.arm_width(arm, 4, len(ridge))
        if arm in CC.DRAFT_ARMS:
            assert not any(n in ridge for n in nm)
        else:
            assert np.array_equal(M[:, :len(ridge)], X[:, [names.index(r) for r in ridge]])
        assert CC.matrix_sha(M) == CC.matrix_sha(CC.arm_matrix(X, names, arm, ridge, versions, vocab, table=_table())[0])
    assert CC.arm_width('class_pairs_identity', 171) == 2704 and CC.arm_width('draft_class', 171) == 534 and CC.arm_width('base', 171) == 352
    M, nm = CC.arm_matrix(X, names, 'tags', ridge, versions, vocab, table=_table())
    ch, _ = CC.slot_columns(names)
    five = X[:, ch[0]] == 5
    col = nm.index('cc_slot0_primary_Mage')
    assert np.all(M[five & (versions == '16.13.1'), col] == 1) and np.all(M[five & (versions == '15.14.1'), col] == 0)
    with pytest.raises(ValueError):
        CC.arm_matrix(X, names, 'tags', ridge, versions[:-1], vocab, table=_table())


def test_fit_family_and_bundle_reload_identity(tmp_path):
    import joblib
    X, y, g, names = _synthetic(400, seed=2)
    ridge = [n for n in names if not n.endswith('champion_id')]
    vocab = [1, 2, 3, 4]
    versions = np.full(len(X), '15.14.1')
    w = CC.weights(g)
    for fam, cfg, arm in (('logit', 'logit_C0.1', 'class_state'), ('lgbm', 'lgbm_L15_M50', 'draft_class')):
        M, nm = CC.arm_matrix(X, names, arm, ridge, versions, vocab, table=_table())
        base, elig, rec, stop = CC.fit_family(fam, cfg, nm, M, y, g, w)
        if fam == 'logit':
            assert isinstance(base, Q.LinearQBase) and stop is None and rec['params']['C'] == 0.1
        else:
            assert isinstance(base, Q.LgbmQBase) and stop['fit90_stop10_match_overlap'] == 0 and len(base.models) == 3
        raw = base.raw(M)
        b = CC.make_bundle(fam, cfg, arm, base, K.fit_calibrators(raw, y, w), nm, {})
        assert b['input_columns'] == list(range(len(nm))) and b['arm'] == arm and b['version'] == CC.VERSION and b['blocks'] == list(CC.ARM_BLOCKS[arm])
        joblib.dump(b, tmp_path / f'{fam}.joblib')
        b2 = joblib.load(tmp_path / f'{fam}.joblib')
        Mf = np.asfortranarray(M)  # column fancy indexing yields F-ordered arrays; predictions must not depend on the layout
        assert not Mf.flags['C_CONTIGUOUS'] and np.array_equal(Mf, M)
        for cal in CC.CALS:
            assert np.array_equal(CC.bundle_predict(b, cal, M, nm), CC.bundle_predict(b2, cal, M, nm))
            assert np.array_equal(CC.bundle_predict(b2, cal, Mf, nm), CC.calibrate(b, cal, base.raw(M)))
    with pytest.raises(ValueError):
        CC.fit_family('pt', 'pt_C1', nm, M, y, g, w)


def test_tag_table_pinned_versions_and_train_vocab_covered(tmp_path):
    tab = CC.load_tag_table()
    assert set(CC.SET_VERSION.values()) <= set(tab['table']) and tab['tags'] == list(CC.TAGS)
    for v, rows in tab['table'].items():
        assert all(r['primary'] in CC.TAGS and (r['secondary'] is None or r['secondary'] in CC.TAGS) for r in rows.values())
        assert all(int(k) < 60000 for k in rows)
    assert len(tab['table']['15.14.1']) == 171
    vocab = CC.identity_vocab_from_train(tmp_path, 'contract test tag coverage')
    assert len(vocab) == 171 and vocab == sorted(vocab) and all(str(v) in tab['table']['15.14.1'] for v in vocab)
    lk = CC.tag_lookup(tab, '15.14.1')
    assert all(0 <= lk[v][0] < 6 for v in vocab)
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_train_fixture_counts_oof_provenance_and_smoke_roles(tmp_path):
    vman = C.read_json(CC.FC / 'v_models_manifest.json')
    F, Lb, Co, checks = CC.load_parent_set('MAIN_TRAIN', tmp_path, 'contract test')
    assert all(checks.values())
    valid = Lb['valid_h90'] == 1
    for coh in CC.COHORTS:
        assert int((valid & (Co['cohort'] == CC.COHORT_CODE[coh])).sum()) == CC.EXPECTED_COUNTS['TRAIN'][coh]
    prov = CC.oof_provenance(F['match'].astype(str), F['sub_role'], Lb['adapter_id'], Lb['adapter_sha256'], vman)
    assert prov['rows'] == prov['adapter_id_equals_own_heldout_fold'] == prov['adapter_sha_equals_manifest_oof_hash']
    F = Lb = Co = None
    D = CC.load_trainval(tmp_path, smoke=True, cohort='T')
    assert set(np.unique(D['role']).tolist()) <= {'TRAIN', 'Q_CAL', 'Q_SELECT'} and np.all(D['version'] == '15.14.1')
    assert 0 < len(D['X']) < CC.EXPECTED_COUNTS['TRAIN']['T'] / 4 and len(D['names']) == 362
    log = [json.loads(line) for line in (tmp_path / 'access_log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert log and not any(r['sealed'] for r in log)


def test_sealed_sets_refuse_until_this_run_freezes(tmp_path, monkeypatch):
    assert Q.frozen_path(CC.IQ).exists()
    monkeypatch.setattr(CC, 'OUT', tmp_path / 'run')
    for name in CC.SEALED_SETS:
        with pytest.raises(PermissionError):
            CC.load_parent_set(name, tmp_path, 'contract test gate')
    with pytest.raises(PermissionError):
        CC.log_access(tmp_path, 'parent predictions', 'MAIN_TEST_h90_T.npz', 'parent predictions', True)
    assert not (tmp_path / 'access_log.jsonl').exists()
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'frozen_manifest.json').write_text('{}', encoding='utf-8')
    CC.log_access(tmp_path, 'after freeze', 'MAIN_TEST', 'probe', True)
    assert json.loads((tmp_path / 'access_log.jsonl').read_text(encoding='utf-8'))['sealed']


def test_fit_stage_uses_q_cal_weights_for_calibrators_and_q_select_for_selection():
    src = (ROOT / 'scripts' / 'cc20260916_fit.py').read_text(encoding='utf-8')
    assert "W = {k: CC.weights(g[m]) for k, m in M.items()}" in src
    assert "K.fit_calibrators(raw['Q_CAL'], y[M['Q_CAL']], W['Q_CAL'])" in src
    assert "(met[c]['Q_SELECT']['brier'], met[c]['Q_SELECT']['logloss'])" in src
    assert "Xa[M['TRAIN']], y[M['TRAIN']], g[M['TRAIN']], W['TRAIN']" in src
    fr = (ROOT / 'scripts' / 'cc20260916_freeze.py').read_text(encoding='utf-8')
    assert "ws = CC.weights(D['g'][ms])" in fr


def test_scripts_never_call_parent_writers_and_write_only_to_this_root():
    forbidden = ('K.Status(', 'K.outcome_gate(', 'KD.', 'import cr20260915_data', 'K.log_command(', 'C.Status(', 'C.aggregate_status(',
                 'Q.Status(', 'Q.log_command(', 'Q.log_failure(', 'Q.log_access(', 'Q.load_parent_set(', 'Q.load_trainval(', 'Q.OUT /',
                 'HS.', 'T.Status(', 'T.load_trainval(')
    import re
    for p in SCRIPTS:
        src = p.read_text(encoding='utf-8')
        assert not [f for f in forbidden if re.search(r'(?<![A-Za-z0-9_])' + re.escape(f), src)], p.name  # CC.Status( is this run's own status writer
    assert CC.OUT == CC.ROOT / 'outputs' / 'champion_class_20260916' and CC.SMOKE.parent == CC.OUT and CC.TAG_TABLE.parent.parent == CC.OUT
    assert 'frozen_path(OUT)' in inspect.getsource(CC.log_access)


def test_parent_read_targets_unchanged_since_before_snapshot():
    snap = CC.OUT / 'integrity' / 'snapshot_before.json'
    if not snap.exists():
        pytest.skip('before snapshot not taken yet')
    before = C.read_json(snap)
    for key in ('sha256_full_corpus_read_targets', 'sha256_incremental_q'):
        for rel, sha in before[key].items():
            assert C.sha256_file(CC.ROOT / rel) == sha, rel
