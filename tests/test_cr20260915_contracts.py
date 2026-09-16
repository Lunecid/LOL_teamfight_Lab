"""Contract tests for the 2026-09-15 cohort / role-aware q experiment (synthetic and TRAIN-only fixtures; no TEST rows).

Covers: scale rule and cohort partition with exact key joins, detector provenance records, no cohort / postgame-role
columns among predictors, role posterior normalization / uniformity / permutation equivariance, weak role annotation
parsing, role inference invariance to postgame roles / future events / outcome, OOF self-match exclusion, static draft
availability, save-load identity, unchanged frozen V labels, TRAIN-only smoke loader and the TEST label gate.
"""
import hashlib
from itertools import permutations
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
os.environ.setdefault('LOL_OUTPUT_ROOT', str(ROOT / 'outputs/cohort_role_training_20260915/runtime_tests'))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import cr20260915_data as KD  # noqa: E402
import cr20260915_draft as KDR  # noqa: E402
import cr20260915_role_models as KRM  # noqa: E402

COH = K.OUT / 'cohorts'
DRAFT = K.OUT / 'draft' / 'MAIN_draft.npz'


def _schema_names():
    return C.read_json(K.FC / 'q_pre_only_schema.json')['input_names_all']


# ------------------------------------------------------------------------------------------ scale / cohorts
def test_scale_rule_classes_and_negative_convention():
    b = np.array([5, 4, 3, -1, 4, -1, 2, 1, 0])
    r = np.array([4, 4, 5, 4, -1, -1, 1, 5, 5])
    proven = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0], dtype=bool)
    n_min, known, cohort, fine = K.scale_classes(b, r, proven)
    assert n_min.tolist() == [4, 4, 3, 0, -1, 0, 1, 1, 0]
    assert known.tolist() == [1, 1, 1, 1, 0, 1, 1, 1, 1]
    assert cohort.tolist() == [1, 1, 0, 0, -1, 0, 0, 0, 0]
    assert fine.tolist() == [2, 2, 1, 0, -1, 0, 0, 0, 0]
    assert K.TEAMFIGHT_MIN == 4 and K.PICK_MAX == 1
    assert C.read_json(K.SHARDS / 'manifest.json')['scale'] == {'pick_max': 1, 'skirmish_min': 2, 'teamfight_min': 4}


@pytest.mark.skipif(not (COH / 'cohort_manifest.json').exists(), reason='cohort stage not run')
def test_train_cohort_exact_key_join_and_partition():
    with np.load(COH / 'MAIN_TRAIN_cohort.npz', allow_pickle=False) as z:
        Co = {k: z[k] for k in z.files}
    with np.load(K.FC / 'labels' / 'MAIN_TRAIN_labels.npz', allow_pickle=False) as z:
        keys = {k: z[k] for k in ('match', 's', 'L', 'sub_role')}            # keys only
    assert np.array_equal(Co['match'].astype(str), keys['match'].astype(str))
    assert np.array_equal(Co['s'], keys['s']) and np.array_equal(Co['L'], keys['L'])
    T, N, known = Co['cohort'] == 1, Co['cohort'] == 0, Co['scale_known'] == 1
    assert not (T & N).any() and np.array_equal(T | N, known)
    nz = Co['negative_read_as_zero'] | ((Co['cluster_blue'] >= 0) & (Co['cluster_red'] >= 0))
    n_min, kn, cohort, fine = K.scale_classes(Co['cluster_blue'], Co['cluster_red'], Co['negative_read_as_zero'])
    assert np.array_equal(cohort, Co['cohort']) and np.array_equal(fine, Co['fine']) and np.array_equal(n_min, Co['n_min'])
    assert np.array_equal(fine == 2, T) and nz.all()
    assert set(Co['source'].tolist()) == {'v33_shard'}
    man = C.read_json(COH / 'cohort_manifest.json')
    assert man['join']['ok'] and man['join']['shard_rows_used_once'] == 566452 and man['join']['valid_rows']['h90'] == 566104
    for s, v in man['sets'].items():
        assert v['T_and_N_intersection'] == 0 and v['T_union_N_equals_known'], s
    assert man['sets']['MAIN_TRAIN']['T'] == int(T.sum())
    fx = C.read_json(COH / 'detector_train_fixture.json')
    assert fx['pass_'] and fx['counts']['counts_equal_shard'] == fx['counts']['engagements'] > 0
    neg = C.read_json(COH / 'negative_count_provenance.json')
    assert neg['rows_total'] == len(neg['rows']) == neg['proven_rows'] + neg['unknown_rows']
    assert all((not r['proven_minus1_is_int_zero']) or (r['redetected'] == 1 and r['L_redetected'] == r['L_label']) for r in neg['rows'])
    assert int(Co['negative_read_as_zero'].sum()) == sum(1 for r in neg['rows'] if str(r['split_role']).startswith('fold') and r['proven_minus1_is_int_zero'])
    assert all(v['pass_'] for v in man['external_checks'].values())


# ------------------------------------------------------------------------------------------ predictors
def test_predictors_exclude_cohort_scale_and_postgame_role_columns():
    names = _schema_names()
    sets = C.q_feature_sets(names)
    assert C.sha256_json(sets) == C.read_json(K.FC / 'q_pre_only_schema.json')['predictor_sets_sha256']
    arm_names = {'participant': K.ArmFeaturizer('participant', names).feature_names,
                 'role': K.ArmFeaturizer('role', names).feature_names}
    bad_tokens = ('cohort', 'cluster', 'present', 'n_min', 'scale', 'teamPosition', 'individualPosition', 'role_slot', 'weak_role',
                  'champion_id', 'snapshot_age', 'delta', 'endpoint', 'winner', 'fold', 'match')
    for arm, nn in list(arm_names.items()) + [('ridge', sets['ridge']), ('economic', sets['economic'])]:
        bad = [n for n in nn if any(t in n for t in bad_tokens) and n != C.P_PRE]
        assert not bad, (arm, bad[:5])
    role = arm_names['role']
    assert not any(n.startswith('participant_slot') for n in role)
    assert len([n for n in role if n.startswith('role_blue_') or n.startswith('role_red_')]) == 2 * 5 * 16
    assert len([n for n in role if n.startswith('role_diff_') and not n.endswith('_x_time_minutes')]) == 5 * 16
    assert len([n for n in role if n.endswith('_x_time_minutes')]) == 5 * 3
    assert sorted(set(K.role_group(n) for n in role)) == sorted(K.ROLE_GROUPS)
    assert K.RoleModel.FIELDS == ('champion_id', 'spell_a', 'spell_b')
    with pytest.raises(ValueError):
        K.ArmFeaturizer('participant', names, drop_groups=('role_TOP',))


# ------------------------------------------------------------------------------------------ role posterior
def _brute(P):
    W = np.zeros((5, 5))
    wsum = 0.0
    for pi in permutations(range(5)):
        w = np.prod([max(P[i, pi[i]], 1e-12) for i in range(5)])
        wsum += w
        for i in range(5):
            W[i, pi[i]] += w
    return W / wsum


def test_team_role_posterior_matches_bruteforce_and_is_doubly_stochastic():
    rng = np.random.default_rng(7)
    P = rng.dirichlet(np.ones(5), size=(9, 5))
    P[3, 2] = [1, 0, 0, 0, 0]
    P[3, 4] = [1, 0, 0, 0, 0]                                           # two certain TOPs -> clipped logs, still finite
    W, Hn, Z = K.team_role_posterior(P)
    for t in range(len(P)):
        assert np.allclose(W[t], _brute(P[t]), atol=1e-12)
    assert np.allclose(W.sum(axis=2), 1, atol=1e-12) and np.allclose(W.sum(axis=1), 1, atol=1e-12)
    assert np.isfinite(W).all() and np.isfinite(Hn).all() and (Hn >= -1e-12).all() and (Hn <= np.log(120) + 1e-12).all()
    same = np.tile(rng.dirichlet(np.ones(5)), (1, 5, 1))
    Wu, Hu, _ = K.team_role_posterior(same)
    assert np.allclose(Wu, 0.2, atol=1e-12) and np.isclose(Hu[0], np.log(120))


def test_permutation_equivariance():
    rng = np.random.default_rng(11)
    P = rng.dirichlet(np.ones(5), size=(4, 5))
    W, Hn, _ = K.team_role_posterior(P)
    for perm in ([4, 3, 2, 1, 0], [1, 0, 3, 2, 4], [2, 4, 0, 1, 3]):
        Wp, Hp, _ = K.team_role_posterior(P[:, perm, :])
        assert np.allclose(Wp, W[:, perm, :], atol=1e-12) and np.allclose(Hp, Hn, atol=1e-12)


def test_unseen_symmetric_drafts_give_uniform_roles_and_unordered_spells():
    D = np.array([[1, 4, 12], [2, 11, 4], [3, 4, 14], [4, 4, 7], [5, 7, 3]] * 40)
    y = np.array([0, 1, 2, 3, 4] * 40)
    m = K.RoleModel(max_iter=2000).fit(D, y, np.ones(len(y)), dict(model_id='fixture'))
    unseen = np.array([[999, 777, 778]] * 5)
    out = KRM.role_outputs(m, unseen[None, :, 0].repeat(2, 0).reshape(1, 10), np.full((1, 10), 777), np.full((1, 10), 778))
    assert np.allclose(out['W'], 0.2, atol=1e-12)
    assert out['flag_champion_unseen'].all() and out['flag_spell_unseen'].all()
    assert np.allclose(m.predict_proba(np.array([[1, 4, 12]])), m.predict_proba(np.array([[1, 12, 4]])))
    assert m.encoder.transform(np.array([[0, 0, 0]])).nnz == 0


# ------------------------------------------------------------------------------------------ draft / weak labels
def _meta(role_slots, tm=None):
    tm = tm or {str(p): (100 if p <= 5 else 200) for p in range(1, 11)}
    return dict(patch='15.14', team_map=tm, role_slots=role_slots,
                static_meta=dict(champion_by_pid={str(p): 10 + p for p in range(1, 11)},
                                 summoner_spells_by_pid={str(p): dict(summoner_spell_1_id=4, summoner_spell_2_id=12) for p in range(1, 11)}))


def test_weak_role_parsing_excludes_detectable_id_fill(tmp_path):
    ident = {str(p): p - 1 for p in range(1, 11)}
    p = tmp_path / 'a.meta.json'
    p.write_text(json.dumps(_meta(ident)), encoding='utf-8')
    r = KDR.parse_meta(p, True)
    assert r['team_supervision_ok'] == [1, 1] and r['role_label'] == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    assert r['champ'] == [11 + i for i in range(10)] and r['spell_a'] == [4] * 10
    fill = {'1': 0, '2': 1, '4': 3, '5': 4, '3': 2, '6': 5, '7': 6, '9': 8, '10': 9, '8': 7}   # pids 3 / 8 filled after role matches
    p.write_text(json.dumps(_meta(fill)), encoding='utf-8')
    r = KDR.parse_meta(p, True)
    assert r['team_supervision_ok'] == [0, 0] and r['team_reason'] == ['detectable_participant_id_fill'] * 2
    assert r['role_label'] == [-1] * 10
    swap = {'2': 0, '1': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, '10': 9}   # role order differs from pid order
    p.write_text(json.dumps(_meta(swap)), encoding='utf-8')
    r = KDR.parse_meta(p, True)
    assert r['team_supervision_ok'] == [1, 1] and r['role_label'][:2] == [1, 0]
    dup = dict(ident, **{'2': 0})
    p.write_text(json.dumps(_meta(dup)), encoding='utf-8')
    assert KDR.parse_meta(p, True)['team_reason'][0] == 'invalid_slot_block_or_duplicate'
    assert KDR.parse_meta(p, False)['team_reason'] == ['not_read', 'not_read']


def _train_fixture_matches(n):
    import csv
    rows = [r['match_id'] for r in csv.DictReader(open(C.MAIN_MANIFEST, encoding='utf-8')) if r['role'] == 'TRAIN']
    return sorted(rows, key=lambda m: hashlib.sha256(f'cr20260915-contract:{m}'.encode()).hexdigest())[:n]


def test_role_inference_invariant_to_postgame_roles_future_events_and_outcome(tmp_path):
    mids = _train_fixture_matches(40)
    recs = [KDR.parse_meta(C.CACHE_MAIN / f'{m}.meta.json', False) for m in mids]
    D = np.stack([np.array([r['champ'] for r in recs]).ravel(), np.array([r['spell_a'] for r in recs]).ravel(),
                  np.array([r['spell_b'] for r in recs]).ravel()], axis=1)
    y = np.tile(np.arange(5), 2 * len(mids))                               # synthetic targets (fixture only)
    model = K.RoleModel().fit(D, y, np.ones(len(y)), dict(model_id='fixture'))
    mid = mids[0]
    for suffix in ('.meta.json', '.events.json'):
        shutil.copy(C.CACHE_MAIN / f'{mid}{suffix}', tmp_path / f'{mid}{suffix}')
    base = KDR.parse_meta(tmp_path / f'{mid}.meta.json', False)
    out0 = KRM.role_outputs(model, np.array([base['champ']]), np.array([base['spell_a']]), np.array([base['spell_b']]))
    meta = json.loads((tmp_path / f'{mid}.meta.json').read_text(encoding='utf-8'))
    events = json.loads((tmp_path / f'{mid}.events.json').read_text(encoding='utf-8'))
    variants = []
    for rs in (None, {}, {str(p): (10 - p) for p in range(1, 11)}, {'invalid': 'future'}):
        m2 = dict(meta)
        if rs is None:
            m2.pop('role_slots', None)
        else:
            m2['role_slots'] = rs
        variants.append(m2)
    for m2 in variants:
        ev2 = [e for e in events if int(e.get('timestamp', 0)) < 600000]      # delete future events
        for e in ev2:
            if e.get('type') == 'GAME_END':
                e['winningTeam'] = 300 - int(e.get('winningTeam', 100) or 100)   # flip outcome W
        (tmp_path / f'{mid}.meta.json').write_text(json.dumps(m2), encoding='utf-8')
        (tmp_path / f'{mid}.events.json').write_text(json.dumps(ev2), encoding='utf-8')
        r = KDR.parse_meta(tmp_path / f'{mid}.meta.json', False)
        out = KRM.role_outputs(model, np.array([r['champ']]), np.array([r['spell_a']]), np.array([r['spell_b']]))
        for k in ('P', 'W', 'U'):
            assert np.array_equal(out[k], out0[k])


def test_role_oof_self_match_exclusion_fixture():
    rng = np.random.default_rng(3)
    mids = np.array([f'KR_{1000 + i}' for i in range(60)])
    A = dict(match=mids, champion_id=rng.integers(1, 30, size=(60, 10)), spell_a=np.full((60, 10), 4), spell_b=np.full((60, 10), 12),
             weak_role_label=np.tile(np.arange(5), (60, 2)).astype(np.int8))
    A['weak_role_label'][5, :5] = -1                                    # excluded team
    fold = np.array([C.train_fold(m) for m in mids])
    for k in range(5):
        Dk, yk, gk = KRM.participant_rows(A, fold != k)
        assert not set(gk.tolist()) & set(mids[fold == k].tolist())
        assert len(yk) == 10 * int((fold != k).sum()) - (5 if fold[5] != k else 0)
    Dall, yall, gall = KRM.participant_rows(A, np.ones(60, dtype=bool))
    w = C.weights(gall)
    tot = np.bincount(np.unique(gall, return_inverse=True)[1], weights=w)
    assert np.allclose(tot, tot[0])                                     # equal total weight per match


@pytest.mark.skipif(not DRAFT.exists(), reason='draft stage not run')
def test_static_draft_availability_and_state_champion_identity_train():
    with np.load(DRAFT, allow_pickle=False) as z:
        A = {k: z[k] for k in ('match', 'split_role', 'roster_ok', 'champion_id', 'spell_a', 'spell_b', 'weak_role_label', 'team_reason')}
    tr = A['split_role'] == 'TRAIN'
    assert A['roster_ok'][tr].all()
    assert (A['champion_id'][tr] > 0).mean() > 0.999 and ((A['spell_a'][tr] > 0) & (A['spell_b'][tr] > 0)).mean() > 0.999
    assert (A['weak_role_label'][~tr] == -1).all() and (A['team_reason'][~tr] == 'not_read').all()
    with np.load(K.FC / 'labels' / 'MAIN_TRAIN_features_pre_only.npz', allow_pickle=False) as z:
        X, fm, ok = z['X_input'][:20000], z['match'][:20000].astype(str), z['pre_ok'][:20000] == 1
    names = _schema_names()
    _, cc = K.participant_columns(names)
    pos = {m: i for i, m in enumerate(A['match'].astype(str).tolist())}
    ix = np.array([pos[m] for m in fm.tolist()])
    assert np.array_equal(X[ok][:, cc].astype(np.int64), A['champion_id'][ix][ok])


# ------------------------------------------------------------------------------------------ representation / io
def test_build_role_matrix_identity_assignment_and_differences():
    names = _schema_names()
    rng = np.random.default_rng(5)
    X = rng.normal(size=(6, len(names)))
    W = np.zeros((6, 10, 5))
    for j in range(10):
        W[:, j, j % 5] = 1
    U = rng.normal(size=(6, 2, len(K.UNCERTAINTY_FEATURES)))
    Z, zn = K.build_role_matrix(X, names, W, U)
    assert zn == K.global_names(names) + K.role_feature_names() and Z.shape == (6, len(zn))
    ix = {n: i for i, n in enumerate(zn)}
    for r_i, r in enumerate(K.ROLES):
        for f in ('totalGold_norm', 'alive'):
            assert np.allclose(Z[:, ix[f'role_blue_{r}_{f}']], X[:, names.index(f'participant_slot{r_i}_{f}')])
            assert np.allclose(Z[:, ix[f'role_red_{r}_{f}']], X[:, names.index(f'participant_slot{r_i + 5}_{f}')])
            assert np.allclose(Z[:, ix[f'role_diff_{r}_{f}']], Z[:, ix[f'role_blue_{r}_{f}']] - Z[:, ix[f'role_red_{r}_{f}']])
        assert np.allclose(Z[:, ix[f'role_diff_{r}_xp_norm_x_time_minutes']], Z[:, ix[f'role_diff_{r}_xp_norm']] * X[:, names.index('time_minutes')])
    assert np.allclose(Z[:, -14:], U.reshape(6, -1))
    perm = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]                                # swapping two blue players with their weights
    Z2, _ = K.build_role_matrix(X[:, :], names, W, U)
    assert np.array_equal(Z, Z2)
    Xp = X.copy()
    P, _ = K.participant_columns(names)
    Xp[:, P[0]], Xp[:, P[1]] = X[:, P[1]], X[:, P[0]]
    Zp, _ = K.build_role_matrix(Xp, names, W[:, perm, :], U)
    assert np.allclose(Zp, Z)


def test_save_load_identity_role_model_and_arm_bundle(tmp_path):
    import joblib
    names = _schema_names()
    rng = np.random.default_rng(9)
    n = 600
    X = rng.normal(size=(n, len(names)))
    X[:, names.index(C.P_PRE)] = rng.uniform(.05, .95, n)
    inputs = dict(X=X, names=names, champ=rng.integers(1, 40, size=(n, 10)), spa=np.full((n, 10), 4), spb=rng.choice([12, 14, 7], size=(n, 10)),
                  W=np.full((n, 10, 5), 0.2), U=rng.normal(size=(n, 2, 7)))
    y = rng.integers(0, 2, n)
    g = np.array([f'm{i // 3}' for i in range(n)])
    M = dict(train=np.arange(n) < 400, calibrate=(np.arange(n) >= 400) & (np.arange(n) < 500), select=np.arange(n) >= 500)
    for arm in ('participant', 'draft', 'role'):
        fz = K.ArmFeaturizer(arm, names).fit(inputs, M['train'])
        Mat = fz.transform(inputs, np.flatnonzero(M['train']))
        rb = K.RidgeBase().fit(Mat, y[M['train']], C.weights(g[M['train']]), fz.n_scaled)
        b = dict(candidate=f'{arm}_ridge_raw', calibration='raw', featurizer=fz, base_model=rb)
        p1 = K.predict_arm_bundle(b, inputs, np.flatnonzero(M['select']))
        joblib.dump(b, tmp_path / f'{arm}.joblib')
        p2 = K.predict_arm_bundle(joblib.load(tmp_path / f'{arm}.joblib'), inputs, np.flatnonzero(M['select']))
        assert np.array_equal(p1, p2)
        beta, b0 = rb.linear_parts()
        Ms = fz.transform(inputs, np.flatnonzero(M['select']))
        lin = (Ms @ beta if not hasattr(Ms, 'toarray') else Ms.toarray() @ beta) + b0
        assert np.allclose(lin, rb.decision(Ms), atol=1e-9)
    D = np.stack([inputs['champ'].ravel(), inputs['spa'].ravel(), inputs['spb'].ravel()], axis=1)
    rm = K.RoleModel().fit(D, np.tile(np.arange(5), n * 2), np.ones(n * 10), dict(model_id='fixture'))
    joblib.dump(rm, tmp_path / 'rm.joblib')
    assert np.array_equal(joblib.load(tmp_path / 'rm.joblib').predict_proba(D[:50]), rm.predict_proba(D[:50]))


def test_draft_slot_encoder_unknowns_are_zero_and_spells_unordered():
    champ = np.array([[1] * 10, [2] * 10])
    enc = K.DraftSlotEncoder().fit(champ, np.array([[4] * 10, [4] * 10]), np.array([[12] * 10, [14] * 10]))
    a = enc.transform(np.array([[1] * 10]), np.array([[4] * 10]), np.array([[12] * 10])).toarray()
    b = enc.transform(np.array([[1] * 10]), np.array([[12] * 10]), np.array([[4] * 10])).toarray()
    z = enc.transform(np.array([[99] * 10]), np.array([[0] * 10]), np.array([[77] * 10])).toarray()
    assert np.array_equal(a, b) and a.sum() == 30 and z.sum() == 0


# ------------------------------------------------------------------------------------------ frozen inputs / gates
def test_frozen_v_labels_and_prior_models_unchanged():
    lt = C.read_json(K.FC / 'labels' / 'labels_trainval_manifest.json')['summaries']
    for nm in ('MAIN_TRAIN', 'MAIN_VALIDATION'):
        assert C.sha256_file(K.FC / 'labels' / f'{nm}_labels.npz') == lt[nm]['labels_sha256']
        assert C.sha256_file(K.FC / 'labels' / f'{nm}_features_pre_only.npz') == lt[nm]['features_sha256']
    fz = C.read_json(K.FC / 'frozen_manifest.json')
    assert C.sha256_file(K.FC / fz['v_final_path']) == fz['v_final_sha256']
    for h, d in fz['q_bundle_sha256'].items():
        assert C.sha256_file(K.FC / 'models' / 'q' / h / f"{fz['q_selections'][h]['chosen']}.joblib") == d[fz['q_selections'][h]['chosen']]


@pytest.mark.skipif(not (COH / 'MAIN_TRAIN_cohort.npz').exists(), reason='cohort stage not run')
def test_smoke_loader_is_train_only_and_test_labels_are_gated():
    D = KD.load_trainval(smoke=True)
    assert D['parts'] == ['MAIN_TRAIN'] and np.all(np.char.startswith(D['sr_true'], 'fold'))
    assert set(np.unique(D['sr']).tolist()) <= {'fold0', 'fold1', 'fold2', 'Q_CAL', 'Q_SELECT'}
    assert not set(D['g'].tolist()) & set(np.load(COH / 'MAIN_VALIDATION_cohort.npz')['match'].astype(str).tolist())
    if not (K.OUT / 'frozen_manifest.json').exists():
        with pytest.raises(PermissionError):
            KD.load_eval_set('MAIN_TEST', 'contract test gate probe')
