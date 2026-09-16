"""Compare existing predictions with a train-only constant; no fitting on test."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/position_ablation_20260914'

def main():
    source = OUT / 'predictions.csv'
    rows = pd.read_csv(source, dtype={'patch': str})
    train = rows[rows.patch == '15.14']
    test = rows[rows.patch == '15.16']
    assert len(train) == 9228 and len(test) == 7664
    assert not set(train['match']) & set(test['match'])
    train_w = 1 / train.groupby('match')['match'].transform('size')
    test_w = 1 / test.groupby('match')['match'].transform('size')
    prevalence = float(np.average(train.y, weights=train_w))
    result = dict(source=str(source.relative_to(ROOT)),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  train_patch='15.14', evaluation_patch='15.16',
                  train_rows=len(train), evaluation_rows=len(test),
                  weighting='Equal total weight per match within each split',
                  train_prevalence=prevalence, metrics={},
                  limitations='Exploratory reused patch; point estimates only, no CI for constant comparison.')
    for name, prediction in [('train_constant', np.full(len(test), prevalence)),
                             ('baseline', test.baseline), ('position', test.position)]:
        result['metrics'][name] = dict(
            auc=float(roc_auc_score(test.y, prediction, sample_weight=test_w)),
            brier=float(brier_score_loss(test.y, prediction, sample_weight=test_w)),
            logloss=float(log_loss(test.y, prediction, sample_weight=test_w)))
    (OUT / 'constant_baseline_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
