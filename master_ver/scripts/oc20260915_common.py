"""Objective-channel ablation 2026-09-15: shared helpers (importing has no side effects).

Design: docs/CLAUDE_OBJECTIVE_CHANNEL_ABLATION_20260915.md (Codex, frozen). Implementation/execution: Claude Opus 5.
Everything written lands in outputs/objective_channel_ablation_20260915. Parents (full corpus, cohort/role, label
validity) are READ ONLY. Reuses the proven read-only wrapper (lv20260915_common.ParentReadOnly with base = this root,
so access logs are written here only) and the pure analysis functions of lv20260915_analysis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import fc20260915_common as C
import lv20260915_common as LV

ROOT = C.ROOT
OUT = ROOT / 'outputs' / 'objective_channel_ablation_20260915'
FC = C.OUT
CR = LV.CR
LVO = LV.OUT
AUDIT = LV.AUDIT
SPEC = ROOT / 'docs' / 'CLAUDE_OBJECTIVE_CHANNEL_ABLATION_20260915.md'
FINDINGS_DOC = ROOT / 'docs' / 'LABEL_VALIDITY_FINDINGS_20260915.md'
AUDIT_DOC = LV.AUDIT_DOC
CAL_NOTE = ROOT / 'docs' / 'DELTA_LABEL_MONOTONE_CALIBRATION_NOTE_20260915.md'
PYTHON = LV.PYTHON

VERSION = 'objective_channel_ablation_20260915'
ROLE_TAG = 'EXPLORATORY_NAMED_OBJECTIVE_CHANNEL_ABLATION_AFTER_PRIOR_TEST_EXPOSURE_NOT_CAUSAL_NOT_GROUND_TRUTH'
MODEL = 'B_noobj'
MODEL_VERSION = 'oc20260915_B_noobj_wp_v2_participant_order'
HS = C.HORIZONS_S
LABEL_SETS = LV.LABEL_SETS
SEALED_LABEL_SETS = LV.SEALED_LABEL_SETS
EXT = LV.EXT
BOOT_REPS = 1000
BOOT_SEED = 20260915
OBJECTIVE_TOKENS = ('baron', 'elder', 'dragon', 'soul', 'herald', 'horde', 'atakhan')
EXPECTED = dict(primary=361, dropped=176, retained=185)
MAX_ITER_LADDER = (1500, 5000, 20000)
UNKNOWN_COL = 'unknown_objective_team_count'
UNKNOWN_SEMANTICS = ('StateV2 (gameplay/state_value.py): count of events with timestamp <= query whose team is not 100/200 — '
                     'ELITE_MONSTER_KILL with killerTeamId (or killer roster team) missing/other, DRAGON_SOUL_GIVEN with teamId not '
                     '100/200 (unassigned soul), BUILDING_KILL/TURRET_PLATE_DESTROYED with teamId not 100/200. It mixes neutral '
                     'objectives, souls and structures, so it cannot be split without changing another input; retained verbatim.')


def dropped_token(name):
    low = name.lower()
    hits = [t for t in OBJECTIVE_TOKENS if t in low]
    return hits


def feature_split(primary_features):
    """(retained, dropped) in primary order by the exact case-insensitive token rule."""
    dropped = [n for n in primary_features if dropped_token(n)]
    retained = [n for n in primary_features if not dropped_token(n)]
    return retained, dropped


class NoObjWinProbV(C.FullCorpusWinProbV2):
    """B_noobj adapter: parent adapter interface, distinct model_version."""

    def __init__(self, state_names, feature_names, base, calibrator, fit_record):
        super().__init__(state_names, feature_names, base, calibrator, fit_record)
        if any(dropped_token(n) for n in self.feature_names):
            raise ValueError('objective-token column in B_noobj features')
        self.model_version = MODEL_VERSION


def load_b_adapter(path, expected_sha256):
    import joblib
    path = Path(path)
    if C.sha256_file(path) != expected_sha256:
        raise ValueError(f'adapter hash mismatch for {path}')
    m = joblib.load(path)
    if not isinstance(m, NoObjWinProbV) or m.model_version != MODEL_VERSION or m.state_version != C.STATE_VERSION:
        raise TypeError(f'{path} is not a B_noobj adapter')
    return m


def parent_reader():
    return LV.ParentReadOnly(base=OUT)


def frozen_manifest_path():
    return OUT / 'frozen_manifest.json'


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
        except Exception as exc:
            groups[p.stem] = {'read_error': repr(exc)}
    agg = dict(version=VERSION, role=ROLE_TAG, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
               running_groups=[g for g, r in groups.items() if r.get('state') == 'running'],
               failed_groups=[g for g, r in groups.items() if r.get('state') == 'failed'],
               groups={g: {k: r.get(k) for k in ('state', 'stage', 'processed', 'total', 'pid', 'elapsed_s', 'error', 'next_step', 'updated')}
                       for g, r in groups.items()},
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


def parent_roots():
    return [FC, CR, LVO]


def read_parent_hash_targets():
    files = LV.read_parent_hash_targets()
    files += sorted(q for q in (LVO / 'observed').glob('*.npz'))
    for fn in ('protocol.json', 'frozen_manifest.json', 'feature_lists.json', 'validation.json', 'REPORT.md', 'DEFINITION_AND_EVIDENCE.md',
               'alt_labels/alt_labels_manifest.json', 'outcome_access_log.jsonl', 'label_access_log.jsonl'):
        files.append(LVO / fn)
    return [f for f in dict.fromkeys(files) if f.exists()]


def source_files():
    return LV.source_files() + [SPEC, FINDINGS_DOC, CAL_NOTE, ROOT / 'scripts' / 'lv20260915_common.py', ROOT / 'scripts' / 'lv20260915_analysis.py',
                                ROOT / 'scripts' / 'lv20260915_fit_v.py', ROOT / 'scripts' / 'lv20260915_labels.py',
                                ROOT / 'scripts' / 'lv20260915_analyze.py', LVO / 'DEFINITION_AND_EVIDENCE.md']
