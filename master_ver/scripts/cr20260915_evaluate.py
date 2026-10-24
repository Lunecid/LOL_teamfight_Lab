"""Stage T: frozen evaluation on TEST 15.16 and each external set (requires frozen_manifest.json; no refit / reselection).

A  identical T rows: pooled frozen q vs T specialist; identical N rows: pooled vs N specialist; inside N the pick and
   skirmish rows; h90 / h60 / h120; E baseline = pooled on all valid rows (must reproduce the full-run numbers); oracle-cohort
   routing (post-cutoff membership, NOT deployable) vs pooled on all known rows.
C  h90 per cohort: every arm candidate; predeclared paired comparisons; role-block ablations on T.
Role reliability: final role model vs external raw teamPosition (post-freeze diagnostic), proxy-mechanism audit.
Match bootstrap 1000 (seed 20260915), fixed models, equal match weights per cell. Frozen hashes verified before and after.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import sys

sys.dont_write_bytecode = True
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402
import cr20260915_data as KD  # noqa: E402
import cr20260915_role_models as KRM  # noqa: E402

import json  # noqa: E402

os.environ['LOL_OUTPUT_ROOT'] = str(K.OUT / 'runtime_repo')
os.environ['LOL_CFG_PRESET'] = 'v3.3'

import argparse  # noqa: E402
from collections import Counter  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

EVAL_SETS = [('MAIN_TEST', 'MAIN')] + [(f'EXT_{s}', s) for s in K.EXT]
FINE = {0: 'pick', 1: 'skirmish'}


def exact01(y, p):
    return dict(exact_0_or_1=int(np.sum((p == 0) | (p == 1))), exact_0_or_1_opposite_label=int(np.sum(((p == 1) & (y == 0)) | ((p == 0) & (y == 1)))),
                min=float(p.min()) if len(p) else None, max=float(p.max()) if len(p) else None)


def cell(y, p, g, bins=True):
    out = C.evaluate(y, p, g, bins=bins)
    out.update(exact01(y, p))
    return out


def verify(fz):
    res = {p: C.sha256_file(K.OUT / p) == sha for p, sha in fz['frozen_files_sha256'].items()}
    for h, sha in fz['pooled_reference']['chosen_bundle_sha256'].items():
        ch = fz['pooled_reference']['q_selections'][h]['chosen']
        res[f'PRIOR:models/q/{h}/{ch}.joblib'] = C.sha256_file(K.FC / 'models' / 'q' / h / f'{ch}.joblib') == sha
    return res


def role_outputs_for(set_key, name, final, final_sha):
    path = K.OUT / 'draft' / ('MAIN_draft.npz' if set_key == 'MAIN' else f'EXT_{set_key}_draft.npz')
    with np.load(path, allow_pickle=False) as z:
        A = {k: z[k] for k in ('match', 'champion_id', 'spell_a', 'spell_b') + (('split_role',) if set_key == 'MAIN' else ())}
    sel = (A['split_role'] == 'TEST') if set_key == 'MAIN' else np.ones(len(A['match']), dtype=bool)
    o = KRM.role_outputs(final, A['champion_id'][sel], A['spell_a'][sel], A['spell_b'][sel])
    arrays = dict(match=A['match'][sel], generator=np.full(int(sel.sum()), 'final'), role_model_sha256=np.asarray(final_sha), **o,
                  role=np.asarray('ROLE_POSTERIOR_FROM_DRAFT_ONLY (estimated meta role, not observed lane); computed after freeze'))
    sha = C.save_npz(K.OUT / 'role_outputs' / f'{name}_roles.npz', **arrays)
    draft = dict(match=A['match'][sel].astype(str), champ=A['champion_id'][sel], spa=A['spell_a'][sel], spb=A['spell_b'][sel])
    summ = dict(file=f'role_outputs/{name}_roles.npz', sha256=sha, matches=int(sel.sum()),
                mean_assignment_entropy=float(o['entropy'].mean()), mean_max_marginal=float(o['max_marginal'].mean()),
                champion_unseen_participant_rate=float(o['flag_champion_unseen'].mean()),
                spell_unseen_participant_rate=float(o['flag_spell_unseen'].mean()),
                teams_with_any_unseen_champion=float(np.mean(np.concatenate([o['flag_champion_unseen'][:, :5].any(1), o['flag_champion_unseen'][:, 5:].any(1)]))),
                normalization_max_abs_dev=float(max(np.max(np.abs(o['W'].sum(2) - 1)), np.max(np.abs(o['W'][:, :5].sum(1) - 1)), np.max(np.abs(o['W'][:, 5:].sum(1) - 1)))))
    return draft, o, summ


def build_inputs(F, draft, o):
    g = F['match'].astype(str)
    pos = {m: i for i, m in enumerate(draft['match'].tolist())}
    ix = np.asarray([pos[m] for m in g.tolist()])
    names = [str(x) for x in F['input_names']]
    inputs = dict(X=np.nan_to_num(F['X_input']), names=names, champ=draft['champ'][ix], spa=draft['spa'][ix], spb=draft['spb'][ix],
                  W=o['W'][ix], U=o['U'][ix])
    _, cc = K.participant_columns(names)
    ok = F['pre_ok'] == 1
    check = dict(pre_ok_rows=int(ok.sum()), champion_state_equals_draft=int(np.sum(np.all(F['X_input'][ok][:, cc].astype(np.int64) == inputs['champ'][ok], axis=1))))
    if check['pre_ok_rows'] != check['champion_state_equals_draft']:
        raise SystemExit(f'champion identity check failed {check}')
    return inputs, check


def arm_predictions(bundles, cands, inputs, rows, ident_rows=2000):
    """Predictions for arm/ablation candidates, base scores cached per (featurizer group, base); identity-checked."""
    out, ident = {}, {}
    groups = {}
    for c in cands:
        b = bundles[c]
        key = (b['arm'], tuple(b['drop_groups']), b['base'])
        groups.setdefault(key, []).append(c)
    for key, cs in groups.items():
        b0 = bundles[cs[0]]
        M = b0['featurizer'].transform(inputs, rows)
        raw = b0['base_model'].raw(M)
        for c in cs:
            b = bundles[c]
            out[c] = K.apply_calibration(b['calibration'], b.get('calibrator'), raw)
            sub = rows[:ident_rows]
            d = float(np.max(np.abs(K.predict_arm_bundle(b, inputs, sub) - out[c][:len(sub)]))) if len(sub) else 0.0
            ident[c] = dict(max_abs_diff_own_bundle_on_first_rows=d, within_1e_12=d <= 1e-12,
                            note='separately loaded bundle, sub-batch (dense BLAS batch composition may change the last bits)')
    return out, ident


def raw_role_worker(task):
    sys.path.insert(0, str(K.REPO))
    from core.roles import get_role_slots_from_detail
    res = []
    for mid, path, sha in task:
        rec = dict(match=mid, sha_ok=0, error='')
        try:
            b = Path(path).read_bytes()
            rec['sha_ok'] = int(hashlib.sha256(b).hexdigest() == sha)
            d = json.loads(b)
            parts = d['info']['participants']
            tm = {int(p['participantId']): int(p['teamId']) for p in parts}
            order = sorted(tm, key=lambda pid: (tm[pid], pid))
            byp = {int(p['participantId']): p for p in parts}
            rec['team_position'] = [K.ROLE_INDEX.get(str(byp[p].get('teamPosition') or ''), -1) for p in order]
            rec['individual_position_raw'] = [str(byp[p].get('individualPosition') or '') for p in order]
            rec['champ'] = [int(byp[p].get('championId') or 0) for p in order]
            rec['spells'] = [sorted([int(byp[p].get('summoner1Id') or 0), int(byp[p].get('summoner2Id') or 0)]) for p in order]
            rs = get_role_slots_from_detail(d)
            seq = list(rs.items())
            rec['proxy_role'] = [rs[p] - (0 if tm[p] == 100 else 5) for p in order]
            det = []
            for tid in (100, 200):
                sl = [s for p, s in seq if tm.get(p) == tid]
                det.append(int(any(b2 <= a for a, b2 in zip(sl, sl[1:]))))
            rec['proxy_detectable_fill'] = det
        except Exception as exc:
            rec['error'] = f'{type(exc).__name__}:{exc}'[:200]
        res.append(rec)
    return res


def role_reliability(st, final, workers):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    out = {}
    for s in K.EXT:
        rows = {r['match_id']: r for r in csv.DictReader(open(C.PREFLIGHT / 'external' / f'external_{s}.csv', encoding='utf-8'))}
        with np.load(K.OUT / 'role_outputs' / f'EXT_{s}_roles.npz', allow_pickle=False) as z:
            rm = z['match'].astype(str)
            W, P, fl_unseen = z['W'], z['P'], z['flag_champion_unseen']
        with np.load(K.OUT / 'draft' / f'EXT_{s}_draft.npz', allow_pickle=False) as z:
            dch, dsa, dsb = z['champion_id'], z['spell_a'], z['spell_b']
        tasks = [(m, str(Path(rows[m]['raw_folder']) / 'detail' / f'{m}.json'), rows[m]['detail_sha256']) for m in rm.tolist()]
        chunks = [tasks[i:i + 500] for i in range(0, len(tasks), 500)]
        recs = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as ex:
            for part in ex.map(raw_role_worker, chunks):
                recs.extend(part)
        by = {r['match']: r for r in recs}
        n = len(rm)
        tp = np.full((n, 10), -1)
        proxy = np.full((n, 10), -1)
        det = np.zeros((n, 2), dtype=int)
        errors, sha_bad, draft_mismatch = 0, 0, 0
        ind_invalid_when_tp_empty = Counter()
        for i, m in enumerate(rm.tolist()):
            r = by[m]
            if r['error']:
                errors += 1
                continue
            sha_bad += int(not r['sha_ok'])
            tp[i] = r['team_position']
            proxy[i] = r['proxy_role']
            det[i] = r['proxy_detectable_fill']
            draft_mismatch += int(r['champ'] != dch[i].tolist() or r['spells'] != [sorted([a, b]) for a, b in zip(dsa[i].tolist(), dsb[i].tolist())])
            for j in range(10):
                if tp[i, j] < 0:
                    ind_invalid_when_tp_empty[r['individual_position_raw'][j]] += 1
        team_ok = np.stack([np.array([sorted(tp[i, :5].tolist()) == [0, 1, 2, 3, 4] for i in range(n)]),
                            np.array([sorted(tp[i, 5:].tolist()) == [0, 1, 2, 3, 4] for i in range(n)])], axis=1)
        valid = np.concatenate([np.repeat(team_ok[:, :1], 5, 1), np.repeat(team_ok[:, 1:], 5, 1)], axis=1)
        pred_w, pred_p = W.argmax(2), P.argmax(2)
        lab = np.clip(tp, 0, 4)
        pw_true = np.take_along_axis(W, lab[..., None], 2)[..., 0]
        conf = np.zeros((5, 5), dtype=int)
        np.add.at(conf, (tp[valid], pred_w[valid]), 1)
        seen = valid & ~fl_unseen.astype(bool)
        unseen = valid & fl_unseen.astype(bool)
        team_exact = np.concatenate([np.all(pred_w[:, :5] == tp[:, :5], 1)[team_ok[:, 0]], np.all(pred_w[:, 5:] == tp[:, 5:], 1)[team_ok[:, 1]]])
        proxy_eq = proxy == tp
        out[s] = dict(
            matches=n, raw_read_errors=errors, raw_sha256_mismatch=sha_bad, draft_vs_raw_champion_or_spell_mismatch_matches=draft_mismatch,
            teams=2 * n, teams_with_valid_unique_teamPosition=int(team_ok.sum()),
            participants_evaluated=int(valid.sum()),
            posterior_argmax_accuracy=float(np.mean(pred_w[valid] == tp[valid])),
            classifier_argmax_accuracy=float(np.mean(pred_p[valid] == tp[valid])),
            mean_posterior_prob_true_role=float(np.mean(pw_true[valid])),
            posterior_log_loss=float(-np.mean(np.log(np.clip(pw_true[valid], 1e-12, 1)))),
            team_exact_assignment_rate=float(np.mean(team_exact)) if len(team_exact) else None,
            per_role_recall={K.ROLES[r]: float(np.mean(pred_w[valid & (tp == r)] == r)) for r in range(5)},
            confusion_rows_teamPosition_cols_posterior_argmax=conf.tolist(),
            seen_champion=dict(participants=int(seen.sum()), accuracy=float(np.mean(pred_w[seen] == tp[seen])) if seen.any() else None),
            unseen_champion=dict(participants=int(unseen.sum()), accuracy=float(np.mean(pred_w[unseen] == tp[unseen])) if unseen.any() else None),
            empty_teamPosition_participants=int((tp < 0).sum()), individualPosition_when_teamPosition_empty=dict(ind_invalid_when_tp_empty),
            proxy_mechanism_audit=dict(
                note='core/roles.get_role_slots_from_detail applied to external raw detail vs teamPosition (estimates how the weak TRAIN proxy behaves; different patches)',
                valid_teams_proxy_equals_teamPosition_participant_rate=float(np.mean(proxy_eq[valid])),
                invalid_teams=int((~team_ok).sum()), invalid_teams_detectable_by_key_order=int(det[~team_ok].sum()),
                invalid_teams_undetectable=int((~team_ok).sum() - det[~team_ok].sum()),
                detectable_flag_on_valid_teams=int(det[team_ok].sum())))
        st.update('running', f'role_reliability_{s}', summary={k: out[s][k] for k in ('participants_evaluated', 'posterior_argmax_accuracy', 'team_exact_assignment_rate')},
                  next_step='next set')
    return out


def main():
    warnings.filterwarnings('ignore', message='X does not have valid feature names')
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--skip-roles-reliability', action='store_true')
    args = ap.parse_args()
    st = K.Status('evaluate')
    edir = K.OUT / 'eval'
    try:
        fz_path = K.OUT / 'frozen_manifest.json'
        if not fz_path.exists():
            raise SystemExit('evaluation requires frozen_manifest.json')
        fz = C.read_json(fz_path)
        before = verify(fz)
        if not all(before.values()):
            raise SystemExit(f'frozen hash mismatch before evaluation: {[k for k, v in before.items() if not v][:5]}')
        edir.mkdir(parents=True, exist_ok=True)
        C.write_json(edir / 'hashes_before.json', before)
        prior_q = C.read_json(K.FC / 'eval' / 'results_q.json')
        pooled = {}
        for h in K.HS:
            ch = fz['pooled_reference']['q_selections'][f'h{h}']['chosen']
            pooled[h] = (ch, joblib.load(K.FC / 'models' / 'q' / f'h{h}' / f'{ch}.joblib'))
        spec = {coh: {h: {c: joblib.load(K.OUT / 'models' / 'q_specialist' / coh / f'h{h}' / f'{c}.joblib') for c in C.Q_CANDIDATES} for h in K.HS}
                for coh in K.COHORTS}
        have_roles = fz.get('role_models') is not None
        have_arms = bool(fz.get('arms'))
        final = None
        if have_roles:
            rman = C.read_json(K.OUT / 'role_models_manifest.json')
            final_sha = rman['models']['final']['sha256']
            final = joblib.load(K.OUT / rman['models']['final']['path'])
        arms = {coh: {c: joblib.load(K.OUT / 'models' / 'arms' / coh / 'h90' / f'{c}.joblib') for arm in K.ARMS for c in K.arm_candidates(arm)}
                for coh in fz.get('arms', {})}
        abl = {}
        if fz.get('ablations'):
            for g_, v in fz['ablations']['blocks'].items():
                for cal in K.CALS:
                    c = f'drop_{g_}_ridge_{cal}'
                    abl[c] = joblib.load(K.OUT / 'models' / 'ablations' / 'T' / 'h90' / f'{c}.joblib')
        RA, RC, role_summ, checks = {}, {}, {}, {}
        for name, set_key in EVAL_SETS:
            st.update('running', f'load_{name}', next_step='role outputs')
            F, Lb, Co = KD.load_eval_set(name, purpose=f'frozen evaluation {name} (A specialists, C arms)')
            inputs = None
            if have_roles:
                draft, o, role_summ[name] = role_outputs_for(set_key, name, final, final_sha)
                inputs, checks[name] = build_inputs(F, draft, o)
            X = np.nan_to_num(F['X_input'])
            g_all = F['match'].astype(str)
            RA[name] = {}
            for h in K.HS:
                v = Lb[f'valid_h{h}'] == 1
                y, g, Xv = Lb[f'Y_h{h}'][v].astype(int), g_all[v], X[v]
                pch, pb = pooled[h]
                p_pool = C.predict_q_bundle(pb, Xv)
                cellE = cell(y, p_pool, g, bins=False)
                prior_m = prior_q['results'][name][f'h{h}']['metrics'][pch]
                cellE['reproduces_full_run_metrics'] = all(cellE[k] == prior_m[k] for k in ('auc', 'brier', 'logloss'))
                cohort_v, fine_v = Co['cohort'][v], Co['fine'][v]
                routed = np.full(len(y), np.nan)
                res_h = dict(rows=int(v.sum()), matches=int(len(np.unique(g))), pooled_chosen=pch, E_pooled_all_valid_rows=cellE,
                             unknown_scale_rows=int((cohort_v < 0).sum()), cohorts={})
                for coh in K.COHORTS:
                    m = cohort_v == K.COHORT_CODE[coh]
                    sch = fz['A_specialists'][f'{coh}_h{h}']['chosen']
                    Ps = {c: C.predict_q_bundle(spec[coh][h][c], Xv[m]) for c in C.Q_CANDIDATES}
                    routed[m] = Ps[sch]
                    ym, gm = y[m], g[m]
                    cc = dict(rows=int(m.sum()), matches=int(len(np.unique(gm))), specialist_chosen=sch,
                              positive_rate_match_weighted=float(np.average(ym, weights=C.weights(gm))) if m.any() else None,
                              pooled=cell(ym, p_pool[m], gm), specialist=cell(ym, Ps[sch], gm),
                              specialist_all_candidates={c: cell(ym, Ps[c], gm, bins=False) for c in C.Q_CANDIDATES},
                              pooled_all_candidates_note='pooled reference = frozen chosen full-run q only')
                    if len(np.unique(gm)) >= 20:
                        cc['bootstrap'] = K.paired_bootstrap(ym, {'specialist': Ps[sch], 'pooled': p_pool[m]}, gm, [('specialist', 'pooled')])
                    if coh == 'N':
                        cc['inside_N'] = {}
                        for code, lab in FINE.items():
                            mm = fine_v[m] == code
                            sub = dict(rows=int(mm.sum()), matches=int(len(np.unique(gm[mm]))),
                                       pooled=cell(ym[mm], p_pool[m][mm], gm[mm], bins=False), N_specialist=cell(ym[mm], Ps[sch][mm], gm[mm], bins=False))
                            if sub['matches'] >= 20:
                                sub['bootstrap'] = K.paired_bootstrap(ym[mm], {'N_specialist': Ps[sch][mm], 'pooled': p_pool[m][mm]}, gm[mm], [('N_specialist', 'pooled')])
                            cc['inside_N'][lab] = sub
                    res_h['cohorts'][coh] = cc
                    C.save_npz(edir / 'predictions' / f'A_{name}_h{h}_{coh}.npz', match=gm, s_ms=Lb['s'][v][m], y=ym, fine=fine_v[m], pooled=p_pool[m],
                               **{f'spec_{c}': Ps[c] for c in C.Q_CANDIDATES}, specialist_chosen=np.asarray(sch))
                    st.log(f'A {name} h{h} {coh}: rows={cc["rows"]} pooled brier={cc["pooled"]["brier"]:.6f} spec({sch}) brier={cc["specialist"]["brier"]:.6f}')
                known = cohort_v >= 0
                orc = dict(label='ORACLE-COHORT ROUTING: membership uses post-cutoff participation; NOT a deployable live model',
                           rows=int(known.sum()), routed=cell(y[known], routed[known], g[known], bins=False), pooled=cell(y[known], p_pool[known], g[known], bins=False))
                if len(np.unique(g[known])) >= 20:
                    orc['bootstrap'] = K.paired_bootstrap(y[known], {'oracle_routed': routed[known], 'pooled': p_pool[known]}, g[known], [('oracle_routed', 'pooled')])
                res_h['oracle_routing_NOT_DEPLOYABLE'] = orc
                RA[name][f'h{h}'] = res_h
                # ------------------------------------------------ C (h90)
                if h == K.PRIMARY_H and have_arms and inputs is not None:
                    RC[name] = {}
                    vi = np.flatnonzero(v)
                    for coh, sel in fz['arms'].items():
                        m = cohort_v == K.COHORT_CODE[coh]
                        rows = vi[m]
                        ym, gm = y[m], g[m]
                        cands = [c for arm in K.ARMS for c in K.arm_candidates(arm)]
                        Pa, ident = arm_predictions(arms[coh], cands, inputs, rows)
                        sch = fz['A_specialists'][f'{coh}_h{h}']['chosen']
                        Pa['A_specialist'] = C.predict_q_bundle(spec[coh][h][sch], Xv[m])
                        Pa['pooled'] = p_pool[m]
                        aw = sel['arm_winners']
                        overall = sel['overall']
                        Pa['overall'] = Pa['A_specialist'] if overall.startswith('A_specialist:') else Pa[overall]
                        named = dict(role_winner=Pa[aw['role']], participant_winner=Pa[aw['participant']], draft_winner=Pa[aw['draft']],
                                     overall_winner=Pa['overall'], A_specialist=Pa['A_specialist'], pooled=Pa['pooled'],
                                     participant_lgbm_raw=Pa['participant_lgbm_raw'], participant_ridge_raw=Pa['participant_ridge_raw'])
                        pairs = [('role_winner', 'participant_winner'), ('draft_winner', 'participant_winner'), ('role_winner', 'draft_winner'),
                                 ('overall_winner', 'A_specialist'), ('A_specialist', 'pooled')]
                        ccell = dict(rows=int(m.sum()), matches=int(len(np.unique(gm))), arm_winners=aw, overall=overall, A_specialist=sch,
                                     metrics={c: cell(ym, Pa[c], gm, bins=(c in aw.values() or c in ('A_specialist', 'pooled'))) for c in cands + ['A_specialist', 'pooled']},
                                     reload_identity_first_rows=ident)
                        if coh == 'T' and abl:
                            ab = fz['ablations']
                            Pab, ident_ab = arm_predictions(abl, list(abl), inputs, rows)
                            ccell['ablation_metrics'] = {c: cell(ym, Pab[c], gm, bins=False) for c in Pab}
                            ccell['ablation_reload_identity_first_rows'] = ident_ab
                            named['full_role_ridge_preselected'] = Pa[ab['preselected']]
                            for g_, bv in ab['blocks'].items():
                                named[f'drop_{g_}'] = Pab[bv['primary']]
                                pairs.append((f'drop_{g_}', 'full_role_ridge_preselected'))
                        if ccell['matches'] >= 20:
                            ccell['bootstrap'] = K.paired_bootstrap(ym, named, gm, pairs)
                        RC[name][coh] = ccell
                        C.save_npz(edir / 'predictions' / f'C_{name}_h90_{coh}.npz', match=gm, s_ms=Lb['s'][v][m], y=ym,
                                   **{k: val for k, val in Pa.items()}, **({f'abl_{k}': val for k, val in Pab.items()} if coh == 'T' and abl else {}))
                        st.log(f'C {name} {coh}: ' + ', '.join(f'{k}={ccell["metrics"][c]["brier"]:.6f}' for k, c in
                                                               [('role', aw['role']), ('participant', aw['participant']), ('draft', aw['draft']), ('A', 'A_specialist'), ('pooled', 'pooled')]))
            F = Lb = Co = X = inputs = None
            C.write_json(edir / 'results_A.json', dict(role=K.ROLE_TAG, frozen_manifest_sha256=C.sha256_file(fz_path), results=RA))
            if RC:
                C.write_json(edir / 'results_C.json', dict(role=K.ROLE_TAG, frozen_manifest_sha256=C.sha256_file(fz_path), results=RC,
                                                           role_outputs=role_summ, input_checks=checks))
        rel = None
        if have_roles and not args.skip_roles_reliability:
            st.update('running', 'role_reliability', next_step='external raw teamPosition')
            rel = role_reliability(st, final, args.workers)
            C.write_json(edir / 'role_reliability_external_raw.json', dict(
                role=K.ROLE_TAG, statement='post-freeze diagnostic on independent external raw teamPosition annotations (16.13-16.15, KR/NA1); '
                                           'teamPosition is Riot position assignment, not observed spatial lane', results=rel))
        after = verify(fz)
        C.write_json(edir / 'hashes_after.json', after)
        ok = all(after.values())
        st.update('complete' if ok else 'failed', 'evaluate', hashes_unchanged=ok, next_step='SHAP')
        return 0 if ok else 1
    except SystemExit as exc:
        st.update('failed', 'evaluate', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'evaluate', error=repr(exc), next_step='inspect, fix, rerun (read-only on frozen models)')
        return 3


if __name__ == '__main__':
    sys.exit(main())
