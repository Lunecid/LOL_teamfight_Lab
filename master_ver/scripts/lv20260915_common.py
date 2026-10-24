"""Label validity diagnostics 2026-09-15: shared helpers (importing has no side effects).

Design: docs/CLAUDE_LABEL_VALIDITY_FULL_20260915.md (Codex). Implementation/execution: Claude Opus 5.
Everything written by this study lands in outputs/label_validity_full_20260915. The frozen parents
outputs/full_corpus_training_20260915 and outputs/cohort_role_training_20260915 are READ ONLY.

The parent readers in fc20260915_data are pure except load_outcomes(), which appends to the PARENT access
log. ParentReadOnly below reuses the pure readers (chunk paths still resolve to the parent extraction) and
re-implements outcome/label loading with the parent's role checks plus this study's freeze gate; access
records are written only to this study's root.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

import fc20260915_common as C
import fc20260915_data as D

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'label_validity_full_20260915'
FC = C.OUT
CR = ROOT / 'outputs' / 'cohort_role_training_20260915'
AUDIT = ROOT / 'outputs' / 'tog_readiness_audit_20260915'
SPEC = ROOT / 'docs' / 'CLAUDE_LABEL_VALIDITY_FULL_20260915.md'
AUDIT_DOC = ROOT / 'docs' / 'TOG_END_TO_END_READINESS_AUDIT_20260915.md'
PYTHON = 'C:/Users/todtj/anaconda3/python.exe'

VERSION = 'label_validity_full_20260915'
ROLE_TAG = 'EXPLORATORY_LABEL_VALIDITY_DIAGNOSTICS_AFTER_PRIOR_TEST_EXPOSURE_NOT_GROUND_TRUTH'
HS = C.HORIZONS_S
PRIMARY_H = C.PRIMARY_H
EXT = ('KR_16.13', 'KR_16.14_pilot', 'KR_16.15', 'NA1_16.13')
LABEL_SETS = ('MAIN_TRAIN', 'MAIN_VALIDATION', 'MAIN_TEST') + tuple(f'EXT_{s}' for s in EXT)
SEALED_LABEL_SETS = ('MAIN_TEST',) + tuple(f'EXT_{s}' for s in EXT)
BOOT_REPS = 1000
BOOT_SEED = 20260915
PACKET_SEED = 'seed20260915'

# ------------------------------------------------------------------ the two fixed comparators (no search)
MODELS = ('B_reg', 'B_econ')
ALL_V = ('A',) + MODELS
ECON_SLOT_SUFFIXES = ('totalGold_norm', 'curGold_norm', 'level_norm', 'xp_norm', 'laneCS_norm', 'jgCS_norm')
MODEL_SPECS = {
    'B_reg': dict(C=0.1, feature_rule='expanded (identical to primary A: every StateV2 column except snapshot_age_s)',
                  purpose='regularization sensitivity of the primary family; not a new architecture'),
    'B_econ': dict(C=0.01, feature_rule='time_minutes, time_minutes_sq; participant_slot*_{' + ','.join(ECON_SLOT_SUFFIXES)
                   + '}; participant_slot*_champion_id (categorical one-hot)',
                   purpose='deliberately narrower snapshot economic/resource valuation comparator; objective gains can '
                           'still enter later observed resources, so it is NOT a no-objective causal ablation'),
}
FORBIDDEN_ECON_TOKENS = ('kills', 'deaths', 'death', 'alive', 'hp_pct', 'mp_pct', 'baron', 'elder', 'dragon', 'soul',
                         'herald', 'horde', 'atakhan', 'tower', 'inhibitor', 'plates', 'age', 'unknown_objective',
                         'snapshot', '_x_time', '_ever', 'acquired')
ALT_MAX_ITER_LADDER = (1500, 5000, 20000)


def econ_features(state_names):
    """B_econ input columns by exact suffix rule, in StateV2 order."""
    out = []
    for n in state_names:
        if n in ('time_minutes', 'time_minutes_sq'):
            out.append(n)
        elif n.startswith('participant_slot') and (any(n.endswith('_' + s) for s in ECON_SLOT_SUFFIXES)
                                                   or n.endswith('_champion_id')):
            out.append(n)
    expected = {'time_minutes', 'time_minutes_sq'} | {f'participant_slot{i}_{s}' for i in range(10)
                                                      for s in ECON_SLOT_SUFFIXES + ('champion_id',)}
    if set(out) != expected or len(out) != 72:
        raise ValueError('B_econ feature rule does not produce the 72 declared columns')
    for n in out:
        core = n.split('_', 2)[-1] if n.startswith('participant_slot') else n
        if n != 'time_minutes_sq' and any(t in core for t in FORBIDDEN_ECON_TOKENS):
            raise ValueError(f'forbidden channel in B_econ: {n}')
    return out


def expanded_features(state_names):
    return [n for n in state_names if n != 'snapshot_age_s']


def model_features(model, state_names):
    if model == 'B_reg':
        return expanded_features(state_names)
    if model == 'B_econ':
        return econ_features(state_names)
    raise ValueError(model)


class AltWinProbV(C.FullCorpusWinProbV2):
    """Comparator V adapter with the primary adapter interface; distinct model_version per comparator."""

    def __init__(self, alt_model, state_names, feature_names, base, calibrator, fit_record):
        super().__init__(state_names, feature_names, base, calibrator, fit_record)
        if alt_model not in MODELS:
            raise ValueError(alt_model)
        self.alt_model = alt_model
        self.model_version = f'lv20260915_{alt_model}_wp_v2_participant_order'


def load_alt_adapter(path, expected_sha256, alt_model):
    import joblib
    path = Path(path)
    if C.sha256_file(path) != expected_sha256:
        raise ValueError(f'adapter hash mismatch for {path}')
    m = joblib.load(path)
    if not isinstance(m, AltWinProbV) or m.alt_model != alt_model:
        raise TypeError(f'{path} is not a {alt_model} adapter')
    if m.model_version != f'lv20260915_{alt_model}_wp_v2_participant_order' or m.state_version != C.STATE_VERSION:
        raise ValueError('adapter version mismatch')
    return m


# ------------------------------------------------------------------ status / logging (this root only)
class Status(C.Status):
    def __init__(self, group, out=OUT):
        super().__init__(out, group)

    def update(self, state, stage, processed=None, total=None, error=None, next_step=None, **extra):
        rec = dict(group=self.group, state=state, stage=stage, pid=os.getpid(), updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                   elapsed_s=round(time.time() - self.t0, 1), processed=processed, total=total, error=error,
                   next_step=next_step, **extra)
        if not self.history or self.history[-1]['stage'] != stage or self.history[-1]['state'] != state:
            self.history.append(dict(stage=stage, state=state, at=rec['updated']))
        rec['history'] = self.history[-80:]
        C.write_json(self.out / 'status' / f'{self.group}.json', rec)
        aggregate_status(self.out)
        self.log(f'STATUS {state} {stage} processed={processed} total={total} error={error} next={next_step} '
                 + json.dumps(C.clean(extra), ensure_ascii=False)[:500])


def aggregate_status(out=OUT):
    out = Path(out)
    groups = {}
    for p in sorted((out / 'status').glob('*.json')):
        try:
            groups[p.stem] = json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:  # concurrent writer
            groups[p.stem] = {'read_error': repr(exc)}
    latest = max(groups.items(), key=lambda kv: kv[1].get('updated', ''), default=(None, {}))
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
               running_groups=[g for g, r in groups.items() if r.get('state') == 'running'],
               failed_groups=[g for g, r in groups.items() if r.get('state') == 'failed'],
               latest_group=latest[0],
               groups={g: {k: r.get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s', 'error',
                                                  'next_step', 'updated')} for g, r in groups.items()},
               failed_run_history='logs/*.log and commands.txt keep every attempt, including failures')
    try:
        C.write_json(out / 'status.json', agg)
    except PermissionError:
        pass


def log_command(argv=None, note=''):
    import sys
    OUT.mkdir(parents=True, exist_ok=True)
    line = ' '.join([PYTHON] + list(argv if argv is not None else sys.argv))
    with open(OUT / 'commands.txt', 'a', encoding='utf-8') as f:
        f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} pid={os.getpid()} {line} {note}'.rstrip() + '\n')


def frozen_manifest_path(base=OUT):
    return Path(base) / 'frozen_manifest.json'


def _append_jsonl(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(C.clean(rec), ensure_ascii=False) + '\n')


# ------------------------------------------------------------------ read-only parent access
class ParentReadOnly:
    """Pure readers of the parent extraction + gated, locally logged outcome and label access.

    Parent role check preserved: outcomes of any non-MAIN set, of TEST, or of unspecified roles require the
    parent's frozen_manifest.json. This study adds: TEST/external/Q_CAL/Q_SELECT outcomes and TEST/external
    generated labels require THIS study's frozen_manifest.json (alternative families frozen first).
    """
    PRE_FREEZE_OUTCOME_ROLES = {f'fold{k}' for k in range(C.N_FOLDS)} | {'V_CAL', 'V_SELECT'}

    def __init__(self, base=OUT):
        self.L = D.Layout(False)          # parent layout; used only to resolve read paths
        if self.L.base != FC:
            raise RuntimeError('parent layout does not resolve to the frozen full-corpus root')
        self.base = Path(base)

    # pure
    def manifest(self, set_id):
        return self.L.manifest(set_id)

    def chunk_paths(self, set_id):
        return D.chunk_paths(self.L, set_id)

    def load_v_rows(self, set_id, sub_roles, bucket_only=True, with_X=True):
        return D.load_v_rows(self.L, set_id, sub_roles, bucket_only=bucket_only, with_X=with_X)

    def load_engagements(self, set_id, sub_roles, states=True, counts=False):
        return D.load_engagements(self.L, set_id, sub_roles, states=states, counts=counts)

    # gated
    def _gate(self, kind, set_id, sub_roles, purpose):
        parent_sealed = set_id != 'MAIN' or sub_roles is None or any(r == 'TEST' for r in sub_roles)
        if parent_sealed and not self.L.frozen_manifest.exists():
            raise PermissionError('parent role check: sealed outcomes before parent freeze')
        ours_needed = sub_roles is None or set_id != 'MAIN' or not set(sub_roles) <= self.PRE_FREEZE_OUTCOME_ROLES
        if ours_needed and not frozen_manifest_path(self.base).exists():
            raise PermissionError(f'{kind} of {set_id}/{sub_roles} sealed until this study writes frozen_manifest.json')
        return dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), kind=kind, set=set_id,
                    sub_roles=sorted(sub_roles) if sub_roles else None, purpose=purpose,
                    this_study_frozen_manifest_exists=frozen_manifest_path(self.base).exists())

    def load_outcomes(self, set_id, sub_roles, purpose):
        rec = self._gate('outcome_W', set_id, sub_roles, purpose)
        res = {}
        for sp, op, cm in self.chunk_paths(set_id):
            with np.load(sp, allow_pickle=False) as z:
                mm = z['m_match']
                sr = z['m_sub_role']
            wanted = set(mm[np.isin(sr, list(sub_roles))].tolist()) if sub_roles is not None else set(mm.tolist())
            if not wanted:
                continue
            with np.load(op, allow_pickle=False) as o:
                if str(o['plan_sha256']) != cm['plan_sha256']:
                    raise ValueError('outcome chunk plan mismatch')
                for m, w, t, r in zip(o['o_match'].tolist(), o['o_winner_blue'].tolist(), o['o_terminal_ms'].tolist(),
                                      o['o_reason'].tolist()):
                    if m in wanted:
                        res[m] = (int(w), int(t), r)
        rec['rows'] = len(res)
        _append_jsonl(self.base / 'outcome_access_log.jsonl', rec)
        return res

    def label_file(self, name, purpose, keys=None):
        """Primary generated label arrays (dict). TEST/external require this study's freeze."""
        if name not in LABEL_SETS:
            raise ValueError(name)
        if name in SEALED_LABEL_SETS and not frozen_manifest_path(self.base).exists():
            raise PermissionError(f'generated labels of {name} sealed until this study freezes')
        with np.load(FC / 'labels' / f'{name}_labels.npz', allow_pickle=False) as z:
            out = {k: z[k] for k in (keys or z.files)}
        _append_jsonl(self.base / 'label_access_log.jsonl',
                      dict(time=time.strftime('%Y-%m-%d %H:%M:%S'), pid=os.getpid(), kind='primary_generated_labels',
                           set=name, purpose=purpose, keys=sorted(keys) if keys else 'all',
                           this_study_frozen_manifest_exists=frozen_manifest_path(self.base).exists()))
        return out


def cohort_file(name):
    with np.load(CR / 'cohorts' / f'{name}_cohort.npz', allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ small numeric helpers
def match_weights(g):
    return C.weights(g) if len(g) else np.zeros(0)


def wmean(x, w):
    return float(np.average(x, weights=w)) if len(x) and np.sum(w) > 0 else None


def sign3(x):
    """-1 / 0 / +1 with exact zero kept as its own category."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0, 1, np.where(x < 0, -1, 0)).astype(np.int8)


def parent_file_inventory(roots):
    """(size, mtime_ns) of every file below the given roots (detects any write without hashing 16 GB)."""
    inv = {}
    for r in roots:
        r = Path(r)
        for dirpath, _dirs, files in os.walk(r):
            for fn in files:
                p = Path(dirpath) / fn
                try:
                    st = p.stat()
                except OSError as exc:
                    inv[str(p.relative_to(ROOT))] = dict(error=repr(exc))
                    continue
                inv[str(p.relative_to(ROOT))] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
    return inv


def read_parent_hash_targets():
    """Files this study reads from the frozen parents (sha256 before and after)."""
    files = []
    for p in sorted((FC / 'extract').rglob('*')):
        if p.is_file():
            files.append(p)
    for sub in ('labels', 'models/v', 'predictions', 'eval/predictions', 'selection'):
        files += sorted(q for q in (FC / sub).rglob('*') if q.is_file())
    for fn in ('protocol.json', 'frozen_manifest.json', 'selection_v.json', 'v_models_manifest.json', 'outcome_access_log.jsonl',
               'source_hashes.json', 'q_pre_only_schema.json', 'validation.json', 'REPORT.md', 'eval/results_v.json',
               'eval/results_q.json'):
        files.append(FC / fn)
    files += sorted(q for q in (CR / 'cohorts').rglob('*') if q.is_file())
    files += [CR / 'eval' / 'predictions' / f'A_MAIN_TEST_h90_{c}.npz' for c in ('T', 'N')]
    for fn in ('protocol.json', 'frozen_manifest.json', 'label_access_log.jsonl', 'validation.json', 'REPORT.md', 'eval/results_A.json'):
        files.append(CR / fn)
    files += [AUDIT / 'calculations.json']
    return [f for f in files if f.exists()]


def source_files():
    wt = C.WT
    repo = C.REPO
    return [SPEC, AUDIT_DOC, ROOT / 'scripts' / 'fc20260915_common.py', ROOT / 'scripts' / 'fc20260915_data.py',
            ROOT / 'scripts' / 'fc20260915_fit_v.py', ROOT / 'scripts' / 'fc20260915_labels.py',
            ROOT / 'scripts' / 'fc20260915_extract.py', ROOT / 'scripts' / 'fc20260915_evaluate.py',
            ROOT / 'scripts' / 'engagement_labels_v3_rules.py', ROOT / 'scripts' / 'cr20260915_common.py',
            ROOT / 'scripts' / 'cr20260915_data.py', ROOT / 'scripts' / 'audit_tog_readiness_20260915.py',
            wt / 'gameplay' / 'state_value.py', wt / 'gameplay' / 'state_value_v2.py',
            wt / 'train' / 'state_value_experiment.py', wt / 'train' / 'temporal_winprob.py',
            repo / 'gameplay' / 'pipeline_cache.py', repo / 'data' / 'cache_io.py', repo / 'gameplay' / 'pipeline.py']
