"""Stage B1: auxiliary meta-role classifier (draft-only inference) and team role posteriors for TRAIN / VALIDATION.

Classifier (fixed engineering choice, no tuning): DraftEncoder (champion OneHotEncoder(handle_unknown='ignore'),
unordered spell multi-hot) -> multinomial LogisticRegression(C=1, lbfgs, max_iter=2000); inputs are ONLY champion ID
and the two selected summoner spells of a participant. Targets: weak role annotations (cached role_slots) of eligible
15.14 TRAIN teams; one row per participant, equal total weight per match. A final model is fit on all eligible TRAIN
teams and five OOF models on the existing TRAIN folds (fold = sha256('full-v-oof-20260915:'+match)[:8] mod 5): both
teams and every engagement row of a held-out match are absent from the generating model and its encoder vocabulary.
If lbfgs does not converge at max_iter the cap alone is raised and logged.

Team posterior: 120 one-to-one assignments weighted by prod_i p_i(role), logsumexp-normalized, marginalized to
W[i, role]; entropy, per-player max marginal and unknown flags are kept. This standard one-of-each-role composition is
a modeling assumption. TRAIN match outputs come from the OOF model of the match's fold; VALIDATION from the final
model. TEST / external outputs are produced only after the freeze (evaluation stage) with the final model.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fc20260915_common as C  # noqa: E402
import cr20260915_common as K  # noqa: E402

import argparse  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402

MAX_ITER = 2000
RAISED_MAX_ITER = 20000


def participant_rows(A, match_mask):
    """Eligible supervision rows (weak label >= 0) of the selected matches -> (draft (n,3), y, match ids)."""
    idx = np.flatnonzero(match_mask)
    lab = A['weak_role_label'][idx]
    ii, jj = np.nonzero(lab >= 0)
    D = np.stack([A['champion_id'][idx][ii, jj], A['spell_a'][idx][ii, jj], A['spell_b'][idx][ii, jj]], axis=1)
    return D, lab[ii, jj].astype(np.int64), A['match'][idx][ii]


def fit_role_model(D, y, g, record, st=None):
    w = C.weights(g)
    t0 = time.time()
    m = K.RoleModel(C_value=1.0, max_iter=MAX_ITER).fit(D, y, w, record)
    if not m.fit_record['converged']:
        if st:
            st.log(f'role model {record.get("model_id")} did not converge at max_iter={MAX_ITER}; raising cap only to {RAISED_MAX_ITER}')
        first = dict(m.fit_record)
        m = K.RoleModel(C_value=1.0, max_iter=RAISED_MAX_ITER).fit(D, y, w, dict(record, first_attempt=first))
    m.fit_record['fit_seconds'] = round(time.time() - t0, 1)
    return m


def role_outputs(model, champ, spa, spb):
    """Draft arrays (n_matches, 10) in StateV2 slot order -> P (n,10,5), W (n,10,5), U (n,2,7), flags."""
    n = len(champ)
    D = np.stack([champ.reshape(-1), spa.reshape(-1), spb.reshape(-1)], axis=1)
    P = model.predict_proba(D).reshape(n, 10, 5)
    fl = model.encoder.flags(D)
    W = np.empty_like(P)
    H = np.empty((n, 2))
    for t, sl in enumerate((slice(0, 5), slice(5, 10))):
        Wt, Ht, _ = K.team_role_posterior(P[:, sl, :])
        W[:, sl, :] = Wt
        H[:, t] = Ht
    mx = W.max(axis=2)
    U = np.zeros((n, 2, len(K.UNCERTAINTY_FEATURES)))
    for t, sl in enumerate((slice(0, 5), slice(5, 10))):
        U[:, t, 0] = H[:, t]
        U[:, t, 1] = mx[:, sl].mean(axis=1)
        U[:, t, 2] = mx[:, sl].min(axis=1)
        for k, key in enumerate(('champion_missing', 'champion_unseen', 'spell_missing', 'spell_unseen')):
            U[:, t, 3 + k] = fl[key].reshape(n, 10)[:, sl].sum(axis=1)
    return dict(P=P, W=W, U=U, entropy=H, max_marginal=mx, **{f'flag_{k}': v.reshape(n, 10) for k, v in fl.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='TRAIN-only: pseudo folds, outputs to smoke_train_only/')
    args = ap.parse_args()
    base = K.OUT / 'smoke_train_only' if args.smoke else K.OUT
    st = K.Status('role_models' + ('_smoke' if args.smoke else ''))
    try:
        if (K.OUT / 'frozen_manifest.json').exists():
            raise SystemExit('frozen manifest exists')
        prov = K.OUT / 'role_supervision_provenance.json'
        if not prov.exists():
            raise SystemExit('role supervision provenance must be written before fitting')
        if not args.smoke:
            ct = K.OUT / 'contract_tests' / 'result.json'
            if not ct.exists() or not C.read_json(ct).get('passed'):
                raise SystemExit('contract tests must pass before full fits')
        dman = C.read_json(K.OUT / 'draft' / 'draft_manifest.json')
        if C.sha256_file(K.OUT / 'draft' / 'MAIN_draft.npz') != dman['main']['sha256']:
            raise SystemExit('draft file changed')
        with np.load(K.OUT / 'draft' / 'MAIN_draft.npz', allow_pickle=False) as z:
            A = {k: z[k] for k in z.files}
        A['match'] = A['match'].astype(str)
        tr = A['split_role'] == 'TRAIN'
        if args.smoke:
            import cr20260915_data as KD
            tr &= KD.smoke_keep(A['match'])
        fold = np.full(len(A['match']), -1)
        fold[tr] = [C.train_fold(m) for m in A['match'][tr].tolist()]
        if args.smoke:
            fold[tr] = [C.h8('cr-smoke-role-fold:', m) % 5 for m in A['match'][tr].tolist()]
        out_models = base / 'role_models'
        out_models.mkdir(parents=True, exist_ok=True)
        manifest = dict(role=K.ROLE_TAG, provenance_sha256=C.sha256_file(prov), draft_sha256=dman['main']['sha256'],
                        classifier='DraftEncoder -> LogisticRegression(C=1, solver=lbfgs, multinomial), no tuning', roles=list(K.ROLES),
                        fold_rule=("sha256('full-v-oof-20260915:'+match)[:8] mod 5 (existing TRAIN folds)" if not args.smoke
                                   else 'SMOKE pseudo folds'), models={})
        D, y, g = participant_rows(A, tr)
        st.update('running', 'fit_final', rows=int(len(y)), matches=int(len(np.unique(g))), next_step='OOF models')
        fit_ids = sorted(set(g.tolist()))
        final = fit_role_model(D, y, g, dict(model_id='final', fit_split='15.14 TRAIN all folds', fit_match_ids_sha256=C.sha256_json(fit_ids),
                                             fit_matches=len(fit_ids), held_out_fold=None), st)
        joblib.dump(final, out_models / 'role_final.joblib')
        manifest['models']['final'] = dict(path=str((out_models / 'role_final.joblib').relative_to(K.OUT)),
                                           sha256=C.sha256_file(out_models / 'role_final.joblib'), fit_record=final.fit_record)
        models = {'final': final}
        for k in range(5):
            mm = tr & (fold != k)
            Dk, yk, gk = participant_rows(A, mm)
            ids = sorted(set(gk.tolist()))
            st.update('running', f'fit_oof_fold{k}', rows=int(len(yk)), matches=len(ids), next_step='next fold')
            mk = fit_role_model(Dk, yk, gk, dict(model_id=f'oof_fold{k}', fit_split=f'15.14 TRAIN folds != {k}', held_out_fold=k,
                                                 fit_match_ids_sha256=C.sha256_json(ids), fit_matches=len(ids)), st)
            held = set(A['match'][tr & (fold == k)].tolist())
            if held & set(ids):
                raise SystemExit(f'fold {k}: held-out matches inside fit set')
            mk.fit_record['held_out_matches_in_fit_set'] = 0
            p = out_models / f'role_oof_fold{k}.joblib'
            joblib.dump(mk, p)
            manifest['models'][f'oof_fold{k}'] = dict(path=str(p.relative_to(K.OUT)), sha256=C.sha256_file(p), fit_record=mk.fit_record)
            models[f'oof_fold{k}'] = mk
        # role outputs: TRAIN by OOF fold model, VALIDATION by final (smoke: TRAIN only)
        out_roles = base / 'role_outputs'
        out_roles.mkdir(parents=True, exist_ok=True)
        groups = [('MAIN_TRAIN', tr)] + ([] if args.smoke else [('MAIN_VALIDATION', A['split_role'] == 'VALIDATION')])
        diag = {}
        for name, sel in groups:
            idx = np.flatnonzero(sel)
            R = {k: None for k in ('P', 'W', 'U', 'entropy', 'max_marginal')}
            gen = np.empty(len(idx), dtype='U12')
            parts = []
            if name == 'MAIN_TRAIN':
                for k in range(5):
                    ii = idx[fold[idx] == k]
                    parts.append((ii, f'oof_fold{k}'))
            else:
                parts.append((idx, 'final'))
            store = {}
            for ii, mid in parts:
                o = role_outputs(models[mid], A['champion_id'][ii], A['spell_a'][ii], A['spell_b'][ii])
                pos = np.searchsorted(idx, ii)
                for key, val in o.items():
                    if key not in store:
                        store[key] = np.zeros((len(idx),) + val.shape[1:], dtype=val.dtype)
                    store[key][pos] = val
                gen[pos] = mid
            for key in R:
                R[key] = store[key]
            W = store['W']
            checks = dict(row_sums_max_abs_dev=float(np.max(np.abs(W.sum(axis=2) - 1))),
                          col_sums_max_abs_dev=float(max(np.max(np.abs(W[:, :5].sum(axis=1) - 1)), np.max(np.abs(W[:, 5:].sum(axis=1) - 1)))))
            if checks['row_sums_max_abs_dev'] > 1e-9 or checks['col_sums_max_abs_dev'] > 1e-9:
                raise SystemExit(f'{name}: role marginals not normalized {checks}')
            arrays = dict(match=A['match'][idx], sub_role=A['sub_role'][idx], generator=gen, **store,
                          role=np.asarray('ROLE_POSTERIOR_FROM_DRAFT_ONLY (estimated meta role, not observed lane)'))
            sha = C.save_npz(out_roles / f'{name}_roles.npz', **arrays)
            d = dict(file=f'role_outputs/{name}_roles.npz', sha256=sha, matches=int(len(idx)), normalization=checks,
                     generators=dict(zip(*np.unique(gen, return_counts=True))),
                     mean_assignment_entropy=float(store['entropy'].mean()), mean_max_marginal=float(store['max_marginal'].mean()),
                     champion_unseen_participant_rate=float(store['flag_champion_unseen'].mean()),
                     champion_missing_participant_rate=float(store['flag_champion_missing'].mean()))
            if name == 'MAIN_TRAIN':
                lab = A['weak_role_label'][idx]
                ok = lab >= 0
                pred = W.argmax(axis=2)
                Pm = store['P']
                d['OOF_agreement_with_WEAK_proxy_NOT_validated_accuracy'] = dict(
                    participants=int(ok.sum()), posterior_argmax_agreement=float(np.mean(pred[ok] == lab[ok])),
                    classifier_argmax_agreement=float(np.mean(Pm.argmax(axis=2)[ok] == lab[ok])),
                    posterior_mean_prob_of_proxy_role=float(np.mean(np.take_along_axis(W, np.clip(lab, 0, 4)[..., None], axis=2)[..., 0][ok])),
                    per_role_posterior_agreement={K.ROLES[r]: float(np.mean(pred[ok & (lab == r)] == r)) for r in range(5)},
                    teams_exact_assignment=float(np.mean(np.all((pred[:, :5] == lab[:, :5]), axis=1)[np.all(ok[:, :5], axis=1)])))
                # self-match exclusion on actual outputs
                viol = 0
                for k in range(5):
                    fit_ids_k = set(A['match'][tr & (fold != k)].tolist())
                    viol += sum(1 for m in A['match'][idx][gen == f'oof_fold{k}'].tolist() if m in fit_ids_k)
                d['oof_self_match_exclusion_violations'] = int(viol)
                if viol:
                    raise SystemExit('role OOF self-match exclusion violated')
            diag[name] = d
            st.update('running', f'outputs_{name}', summary={k: v for k, v in d.items() if k not in ('generators',)}, next_step='next')
        manifest['outputs'] = diag
        manifest['written_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        C.write_json(base / 'role_models_manifest.json', manifest)
        st.update('complete', 'role_models', converged={k: v['fit_record']['converged'] for k, v in manifest['models'].items()},
                  next_step='fit arms')
        return 0
    except SystemExit as exc:
        st.update('failed', 'role_models', error=str(exc), next_step='inspect')
        raise
    except Exception as exc:
        st.log(traceback.format_exc())
        st.update('failed', 'role_models', error=repr(exc), next_step='inspect, fix, rerun')
        return 3


if __name__ == '__main__':
    sys.exit(main())
