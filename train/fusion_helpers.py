from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.common import sigmoid_np
from core.fight_types import FightRef, ref_key
from core.utils import metrics_from_probs

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore

    HAS_SK = True
except ImportError:
    LogisticRegression = None
    HAS_SK = False


def _sanitize_meta_X(X: np.ndarray, clip: float = 20.0) -> np.ndarray:
    if X is None:
        return X
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if clip is not None and clip > 0:
        X = np.clip(X, -clip, clip)
    return X


def _logit_from_prob(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _safe_tag(s: str) -> str:
    s = str(s)
    for ch in [" ", "/", "\\", ":", "|", ";", ",", "\t", "\n"]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "combo"


@dataclass
class StackingResult:
    ok: bool
    meta_method: str
    base_names: List[str]
    metrics: Dict[str, Any]
    out_dir: str
    pred_logit_map: Optional[Dict[str, float]] = None


def split_logit_map_by_refs(refs: List[FightRef], logit_map: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for r in refs:
        k = ref_key(r)
        if k in logit_map:
            out[k] = float(logit_map[k])
    return out


def _fit_logreg(X: np.ndarray, y: np.ndarray, seed: int) -> Any:
    if not HAS_SK or LogisticRegression is None:
        raise RuntimeError("scikit-learn not installed")
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        random_state=int(seed),
    )
    clf.fit(X, y)
    return clf


def _predict_proba(clf, X: np.ndarray) -> np.ndarray:
    X = _sanitize_meta_X(X)
    p = clf.predict_proba(X)[:, 1]
    return p


def _eval_single_base_auc(
    refs: List[FightRef],
    y_map: Dict[str, int],
    logit_map: Dict[str, float],
) -> Dict[str, Any]:
    ys: List[int] = []
    ps: List[float] = []

    for r in refs:
        k = ref_key(r)
        if k in y_map and k in logit_map:
            ys.append(int(y_map[k]))
            ps.append(float(logit_map[k]))

    if len(ys) < 10:
        return {"n": len(ys), "auc": float("nan")}

    y = np.asarray(ys, dtype=np.int64)
    z = np.asarray(ps, dtype=np.float64)
    p = sigmoid_np(z)

    met = metrics_from_probs(y, p, threshold=0.5)
    met["n"] = int(len(ys))
    return met
