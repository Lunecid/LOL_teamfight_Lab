from __future__ import annotations

import random

import numpy as np

from core.config import F_GLOBAL, F_NODE, cfg
from gameplay.labels import compute_label


def _dummy_interp(cache, q_ms):
    return (
        np.zeros((10, F_NODE), dtype=np.float32),
        np.zeros((F_GLOBAL,), dtype=np.float32),
    )


def test_random_tie_policy_is_seeded_and_replay_stable(monkeypatch):
    monkeypatch.setattr(cfg, "LABEL_TYPE", "attention_value_win", raising=False)
    monkeypatch.setattr(cfg, "LABEL_TIE_STRATEGY", "random", raising=False)
    monkeypatch.setattr(cfg, "LABEL_TIE_SEED", 7, raising=False)

    cache = {
        "minute_ts": np.array([0, 60_000], dtype=np.int64),
        "events": [
            {"timestamp": 1_000, "type": "CHAMPION_KILL", "killerId": 1, "victimId": 6},
            {"timestamp": 2_000, "type": "CHAMPION_KILL", "killerId": 6, "victimId": 1},
        ],
    }
    tm = {pid: (100 if pid <= 5 else 200) for pid in range(1, 11)}

    random.seed(1)
    y1 = compute_label(cache, tm, 0, horizon_ms=30_000, interp_node_global=_dummy_interp)
    random.seed(999)
    y2 = compute_label(cache, tm, 0, horizon_ms=30_000, interp_node_global=_dummy_interp)

    assert y1 in (0, 1)
    assert y1 == y2
