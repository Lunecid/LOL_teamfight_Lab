"""Pin the v3 frozen inputs by SHA256, in the same layout as docs/CLAUDE_NEXT_STAGE_MANIFEST.json.

Promoting v3 to the frozen value model means the hash ledger has to move with it.  This lists
every artefact a downstream step may consume - models, protocol, sampling, scored engagements,
the three validation result files and the source that produced them - and records the
environment they were produced in, since the pickles do not load under other scikit-learn versions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FROZEN = [
    # value models and how they were made
    'outputs/temporal_winprob_v3_buckets/protocol.json',
    'outputs/temporal_winprob_v3_buckets/sampled_minutes.json',
    'outputs/temporal_winprob_v3_buckets/sampling_census.json',
    'outputs/temporal_winprob_v3_buckets/expanded_training_cv.json',
    'outputs/temporal_winprob_v3_buckets/selection.json',
    'outputs/temporal_winprob_v3_buckets/maymin_model.joblib',
    'outputs/temporal_winprob_v3_buckets/expanded_model.joblib',
    'outputs/temporal_winprob_v3_buckets/results.json',
    'outputs/temporal_winprob_v3_buckets/independent_time_curves.npz',
    # scored engagements, both partitions
    'outputs/temporal_winprob_v3_buckets/engagement_changes.npz',
    'outputs/temporal_winprob_v3_buckets/engagement_report.json',
    'outputs/temporal_winprob_v3_buckets/engagement_changes_predict_train.npz',
    'outputs/temporal_winprob_v3_buckets/engagement_report_predict_train.json',
    # validations re-run against v3
    'outputs/temporal_winprob_v3_buckets/validation_a/results.json',
    'outputs/temporal_winprob_v3_buckets/validation_b/results_b.json',
    'outputs/temporal_winprob_v3_buckets/validation_c/results_c.json',
    # the predictor trained on the contract
    'outputs/engagement_predictor_v3/results.json',
    'outputs/engagement_predictor_v3/engagement_predictor_deltaV_weighted.joblib',
    'outputs/engagement_predictor_v3/engagement_predictor_market_event.joblib',
    'outputs/engagement_predictor_v3/predictions.npz',
    'outputs/engagement_predictor_v3/magnitude_diagnostic.json',
    'outputs/engagement_predictor_v3/magnitude_headroom_decomposition.json',
    'outputs/engagement_predictor_v3/two_axis_summary.json',
    'outputs/engagement_predictor_v3/two_axis_test.csv',
    # upstream inputs that must not drift
    'outputs/state_value_main_50k/schema.json',
    'outputs/state_value_main_50k/manifest.json',
    'outputs/state_value_main_50k_eval/match_splits.json',
    # source
    'gameplay/state_value.py',
    'train/temporal_winprob.py',
    'train/state_value_experiment.py',
    'scripts/run_temporal_winprob_v3_buckets.py',
    'scripts/score_engagements_v3.py',
    'scripts/train_engagement_predictor_v3.py',
    'scripts/diagnose_deltaV_magnitude_predictability.py',
    'scripts/report_two_axis_output_v3.py',
    'scripts/validate_temporal_winprob_claude.py',
    'scripts/validate_boundary_sensitivity_claude.py',
    'scripts/validate_objective_passthrough_claude.py',
    'requirements.txt',
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=ROOT / 'docs/CLAUDE_V3_FROZEN_MANIFEST.json')
    ap.add_argument('--allow-missing', action='store_true')
    a = ap.parse_args()
    head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    files, missing = [], []
    for rel in FROZEN:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        files.append({'path': rel.replace('/', '\\'), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    if missing and not a.allow_missing:
        sys.exit('missing frozen inputs:\n  ' + '\n  '.join(missing))
    manifest = {
        'frozen_value_model': 'v3 (independent_wp_v3_bucket_sampling)',
        'supersedes': 'outputs/temporal_winprob_v2 (kept as the documented predecessor)',
        'base_commit': head,
        'workspace': str(ROOT),
        'label_contract': {
            'decided': '2026-09-10, revised the same day after the predictor run',
            'outcome_target': 'market_event (predicted from X at cutoff)',
            'value_axis': 'delta-V_end = V3(S_end) - V3(S_pre), reported alongside, not a target',
            'boundary': 'end = last cluster kill',
            'weights': 'match weights only (|delta-V| weighting withdrawn: 0.6344 -> 0.5885)',
            'evidence': 'docs/CLAUDE_V3_PREDICTOR_REPORT.md'},
        'environment': {'python': sys.version.split()[0],
                        'packages': {q: version(q) for q in ('numpy', 'scikit-learn', 'lightgbm', 'joblib')},
                        'note': 'the joblib pickles do not load under scikit-learn 1.8; use these versions'},
        'files': files,
        'missing': missing,
    }
    a.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{len(files)} files pinned -> {a.out}' + (f' ({len(missing)} missing)' if missing else ''))


if __name__ == '__main__':
    main()
