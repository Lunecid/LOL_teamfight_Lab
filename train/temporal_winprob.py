"""Literature-based snapshot win probability; no engagement-conditioned training."""
import hashlib

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def stable_int(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def random_minute_index(match_id, size):
    if size < 1:
        raise ValueError("no eligible nonterminal minutes")
    return int(np.random.default_rng(stable_int('minute:7:' + match_id)).integers(size))


def feature_matrix(states, names, family):
    states = np.asarray(states)
    if family == 'expanded':
        # Fixed minute queries fall just BEFORE jittered API frames, so age is
        # almost always ~60s in the source training grid. Scaling that narrow
        # nuisance feature extrapolates wildly at live 0-60s query ages.
        # Keep age as audit metadata, not a game-state valuation covariate.
        keep = [i for i,name in enumerate(names) if name != 'snapshot_age_s']
        return states[:,keep], [names[i] for i in keep]
    if family != 'maymin':
        raise ValueError(family)
    index = {name: i for i, name in enumerate(names)}
    columns = [states[:, index['time_minutes']]]
    labels = ['time_minutes']
    for kind, keys in [('kills', ['kills']),
                       ('towers', ['tower_' + t for t in ('OUTER_TURRET', 'INNER_TURRET', 'BASE_TURRET', 'NEXUS_TURRET', 'OTHER')]),
                       ('monsters', ['dragons', 'baron', 'elder', 'herald', 'horde', 'atakhan'])]:
        for team in ('red_', 'blue_'):
            columns.append(states[:, [index[team + key] for key in keys]].sum(axis=1))
            labels.append(team + kind)
    return np.column_stack(columns), labels


class SnapshotWinProbability:
    def __init__(self, family, state_names, base, calibration='raw', calibrator=None):
        self.family, self.state_names, self.base = family, state_names, base
        self.calibration, self.calibrator = calibration, calibrator

    def predict_proba(self, states):
        X, _ = feature_matrix(states, self.state_names, self.family)
        p = self.base.predict_proba(X)[:, 1]
        if self.calibration == 'sigmoid':
            logit = np.log(np.clip(p, 1e-10, 1 - 1e-10) / np.clip(1-p, 1e-10, 1))
            p = self.calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]
        elif self.calibration == 'isotonic':
            p = self.calibrator.predict(p)
        return np.column_stack([1-p, p])


def maymin_model():
    # Original model family, with standardized inputs for numerical stability.
    return make_pipeline(StandardScaler(), LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000))


def calibrated_variants(family, names, base, states, y):
    raw = SnapshotWinProbability(family, names, base)
    p = raw.predict_proba(states)[:, 1]
    logit = np.log(np.clip(p, 1e-10, 1-1e-10) / np.clip(1-p, 1e-10, 1))
    sigmoid = LogisticRegression(C=1e6, solver='lbfgs', max_iter=1000).fit(logit.reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds='clip').fit(p, y)
    return {'raw': raw, 'sigmoid': SnapshotWinProbability(family, names, base, 'sigmoid', sigmoid),
            'isotonic': SnapshotWinProbability(family, names, base, 'isotonic', isotonic)}


def observed_queries(cutoff, last_kill, terminal, last_snapshot):
    """No fabricated combat end: last recorded cluster kill is an explicit proxy."""
    pre = int(cutoff) - 1
    end = int(last_kill)
    if pre < 0 or end <= pre or end >= terminal or end > last_snapshot:
        raise ValueError('invalid_observed_engagement_boundary')
    return [pre, end, *[end + d if end + d < terminal and end + d <= last_snapshot else -1
                        for d in (30000, 60000)]]
