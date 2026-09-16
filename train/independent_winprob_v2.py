"""Versioned snapshot win-probability adapter for objective_history_v2_participant_order states.

The frozen v1 SnapshotWinProbability vectorises bare arrays in role-slot order. This adapter
accepts only StateV2 objects, or a matrix whose declared state version and exact column order
match, and rejects legacy slotN_ schemas before any vectorisation.
"""
from __future__ import annotations

import re

import numpy as np

from gameplay.state_value_v2 import STATE_VERSION, state_matrix
from train.temporal_winprob import feature_matrix

MODEL_VERSION = 'independent_wp_v2_participant_order'
FAMILIES = ('maymin', 'expanded')
CALIBRATIONS = ('raw', 'sigmoid', 'isotonic')
CHAMPION_COLUMNS = tuple(f'participant_slot{i}_champion_id' for i in range(10))
_LEGACY_SLOT = re.compile(r'^slot\d+_')


def check_v2_names(names):
    names = [str(n) for n in names]
    if len(names) != len(set(names)):
        raise ValueError('duplicate state feature names')
    if any(_LEGACY_SLOT.match(n) for n in names):
        raise ValueError('legacy role-slot feature names are not a V2 schema')
    if tuple(n for n in names if n.endswith('champion_id')) != CHAMPION_COLUMNS:
        raise ValueError('participant champion columns missing or out of order')
    return names


def categorical_columns(cols):
    """Columns that train.state_value_experiment.logistic() one-hot encodes."""
    return [k for k in cols if k.endswith('champion_id')]


def _logit(p):
    return np.log(np.clip(p, 1e-10, 1 - 1e-10) / np.clip(1 - p, 1e-10, 1))


class IndependentWinProbabilityV2:
    def __init__(self, family, state_names, base, calibration='raw', calibrator=None, *, fit_record=None):
        if family not in FAMILIES:
            raise ValueError(f'unknown family {family!r}')
        if calibration not in CALIBRATIONS:
            raise ValueError(f'unknown calibration {calibration!r}')
        if (calibration == 'raw') != (calibrator is None):
            raise ValueError('calibrator does not match calibration method')
        self.model_version = MODEL_VERSION
        self.state_version = STATE_VERSION
        self.family = family
        self.state_names = check_v2_names(state_names)
        self.feature_names = feature_matrix(np.zeros((1, len(self.state_names))), self.state_names, family)[1]
        self.base, self.calibration, self.calibrator = base, calibration, calibrator
        self.fit_record = dict(fit_record or {})

    @property
    def key(self):
        return f'{self.family}_{self.calibration}'

    def _from_matrix(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.state_names):
            raise ValueError('state matrix shape mismatch')
        if not np.isfinite(X).all():
            raise ValueError('non-finite state values')
        features, cols = feature_matrix(X, self.state_names, self.family)
        if cols != self.feature_names:
            raise ValueError('feature schema drift')
        p = self.base.predict_proba(features)[:, 1]
        if self.calibration == 'sigmoid':
            p = self.calibrator.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        elif self.calibration == 'isotonic':
            p = self.calibrator.predict(p)
        return np.asarray(p, dtype=float)

    def predict_states(self, states):
        return self._from_matrix(state_matrix(list(states), self.state_names, self.state_version))

    def predict_matrix(self, X, names, state_version):
        if state_version != self.state_version:
            raise ValueError('state version mismatch; regenerate states and refit model')
        if [str(n) for n in names] != self.state_names:
            raise ValueError('state feature schema mismatch')
        return self._from_matrix(X)

    def predict_proba(self, states):
        if isinstance(states, np.ndarray):
            raise TypeError('bare arrays are rejected; use predict_matrix with names and state version')
        p = self.predict_states(states)
        return np.column_stack([1 - p, p])


def load_adapter(path):
    import joblib
    model = joblib.load(path)
    if not isinstance(model, IndependentWinProbabilityV2):
        raise TypeError('not an IndependentWinProbabilityV2 adapter')
    if model.model_version != MODEL_VERSION or model.state_version != STATE_VERSION:
        raise ValueError('adapter version mismatch')
    return model