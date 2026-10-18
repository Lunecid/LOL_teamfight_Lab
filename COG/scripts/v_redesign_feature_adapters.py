#!/usr/bin/env python3
"""Typed StateV2 feature adapters for V redesign (Expanded361 / Core267).

Implements CAT-1 contracts from docs/V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md:
  - numeric / champion categorical split
  - Core267 drops team_time_interaction columns
  - Logistic/RF: TRAIN one-hot (+ UNK)
  - LightGBM: native categorical
  - MLP: integer ids → embedding table (vocab from TRAIN)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def classify_name(name: str) -> str:
    if name.endswith("_champion_id") or name.endswith("champion_id"):
        return "champion"
    if name in ("time_minutes", "time_minutes_sq"):
        return "clock"
    if name == "unknown_objective_team_count":
        return "quality"
    if name == "snapshot_age_s":
        return "audit_excluded"
    if "x_time" in name:
        return "team_time_interaction"
    if name.startswith("participant_slot") and any(
        k in name
        for k in ("totalGold", "curGold", "level", "xp", "hp", "mp", "alive", "laneCS", "jgCS")
    ):
        return "player_snapshot"
    if name.startswith("participant_slot"):
        return "player_event"
    if name.startswith(("blue_", "red_")):
        return "team_event"
    return "other"


@dataclass
class FeatureSchema:
    raw_names: List[str]
    keep_names: List[str]
    keep_idx: np.ndarray
    numeric_names: List[str]
    numeric_idx_in_keep: np.ndarray
    champ_names: List[str]
    champ_idx_in_keep: np.ndarray
    core_numeric_names: List[str]
    core_numeric_idx_in_keep: np.ndarray
    groups: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "FeatureSchema":
        raw = list(names)
        keep_idx = np.asarray([i for i, n in enumerate(raw) if n != "snapshot_age_s"], dtype=np.int64)
        keep_names = [raw[i] for i in keep_idx]
        groups: Dict[str, int] = {}
        num_i, champ_i, core_i = [], [], []
        for j, n in enumerate(keep_names):
            g = classify_name(n)
            groups[g] = groups.get(g, 0) + 1
            if g == "champion":
                champ_i.append(j)
            else:
                num_i.append(j)
                if g != "team_time_interaction":
                    core_i.append(j)
        return cls(
            raw_names=raw,
            keep_names=keep_names,
            keep_idx=keep_idx,
            numeric_names=[keep_names[j] for j in num_i],
            numeric_idx_in_keep=np.asarray(num_i, dtype=np.int64),
            champ_names=[keep_names[j] for j in champ_i],
            champ_idx_in_keep=np.asarray(champ_i, dtype=np.int64),
            core_numeric_names=[keep_names[j] for j in core_i],
            core_numeric_idx_in_keep=np.asarray(core_i, dtype=np.int64),
            groups=groups,
        )

    def slice_raw(self, X_raw: np.ndarray) -> np.ndarray:
        return np.asarray(X_raw)[:, self.keep_idx]

    def numeric(self, X_keep: np.ndarray, profile: str) -> np.ndarray:
        idx = self.numeric_idx_in_keep if profile == "expanded" else self.core_numeric_idx_in_keep
        return np.asarray(X_keep)[:, idx].astype(np.float64, copy=False)

    def champions(self, X_keep: np.ndarray) -> np.ndarray:
        return np.asarray(X_keep)[:, self.champ_idx_in_keep].astype(np.int64)


@dataclass
class OneHotEncoder:
    """TRAIN-fit one-hot over champion slots; unknown → UNK column per slot."""

    slot_categories: List[List[int]]  # incl. reserved UNK sentinel handled as last
    unk_id: int = -1

    @classmethod
    def fit(cls, champs: np.ndarray) -> "OneHotEncoder":
        # champs: N x 10
        slots = []
        for s in range(champs.shape[1]):
            vals = sorted({int(v) for v in champs[:, s].tolist() if int(v) >= 0})
            slots.append(vals)  # UNK = anything not in list
        return cls(slot_categories=slots)

    @property
    def n_out(self) -> int:
        # per slot: |cats| + 1 UNK
        return sum(len(c) + 1 for c in self.slot_categories)

    def transform(self, champs: np.ndarray) -> np.ndarray:
        n = champs.shape[0]
        out = np.zeros((n, self.n_out), dtype=np.float64)
        col = 0
        for s, cats in enumerate(self.slot_categories):
            index = {c: i for i, c in enumerate(cats)}
            unk = len(cats)
            width = unk + 1
            for i in range(n):
                v = int(champs[i, s])
                j = index.get(v, unk)
                out[i, col + j] = 1.0
            col += width
        return out


@dataclass
class ChampVocab:
    """Shared champion id → embedding index (0=UNK)."""

    id_to_ix: Dict[int, int]
    n_vocab: int

    @classmethod
    def fit(cls, champs: np.ndarray) -> "ChampVocab":
        ids = sorted({int(v) for v in champs.reshape(-1).tolist() if int(v) >= 0})
        id_to_ix = {c: i + 1 for i, c in enumerate(ids)}  # 0 reserved UNK
        return cls(id_to_ix=id_to_ix, n_vocab=len(ids) + 1)

    def transform(self, champs: np.ndarray) -> np.ndarray:
        flat = champs.astype(np.int64).reshape(-1)
        out = np.zeros(flat.shape, dtype=np.int64)
        for i, v in enumerate(flat.tolist()):
            out[i] = self.id_to_ix.get(int(v), 0)
        return out.reshape(champs.shape)


def mean_one_match_weights(g: np.ndarray) -> np.ndarray:
    _, inv, c = np.unique(np.asarray(g).astype(str), return_inverse=True, return_counts=True)
    w = (1.0 / c[inv]).astype(np.float64)
    return w / w.mean()


def match_holdout_mask(g: np.ndarray, frac: float = 0.15, seed: int = 7) -> np.ndarray:
    """True = stop/holdout matches (match-disjoint)."""
    matches = np.unique(np.asarray(g).astype(str))
    rng = np.random.default_rng(seed)
    n_hold = max(1, int(round(len(matches) * frac)))
    hold = set(rng.choice(matches, size=n_hold, replace=False).tolist())
    return np.isin(np.asarray(g).astype(str), list(hold))


class ProfileBundle:
    """Fitted transforms for one profile (expanded|core)."""

    def __init__(self, schema: FeatureSchema, profile: str):
        assert profile in ("expanded", "core")
        self.schema = schema
        self.profile = profile
        self.ohe: Optional[OneHotEncoder] = None
        self.vocab: Optional[ChampVocab] = None
        self.num_mean: Optional[np.ndarray] = None
        self.num_std: Optional[np.ndarray] = None

    def fit(self, X_raw: np.ndarray) -> "ProfileBundle":
        Xk = self.schema.slice_raw(X_raw)
        num = self.schema.numeric(Xk, self.profile)
        ch = self.schema.champions(Xk)
        self.ohe = OneHotEncoder.fit(ch)
        self.vocab = ChampVocab.fit(ch)
        self.num_mean = num.mean(axis=0)
        self.num_std = num.std(axis=0)
        self.num_std = np.where(self.num_std < 1e-8, 1.0, self.num_std)
        return self

    def numeric_raw(self, X_raw: np.ndarray) -> np.ndarray:
        return self.schema.numeric(self.schema.slice_raw(X_raw), self.profile)

    def champions_raw(self, X_raw: np.ndarray) -> np.ndarray:
        return self.schema.champions(self.schema.slice_raw(X_raw))

    def matrix_onehot(self, X_raw: np.ndarray) -> np.ndarray:
        assert self.ohe is not None
        return np.hstack([self.numeric_raw(X_raw), self.ohe.transform(self.champions_raw(X_raw))])

    def matrix_lgbm(self, X_raw: np.ndarray) -> Tuple[np.ndarray, List[str], List[int]]:
        """Returns float matrix with champ cols as codes; champ column indices for categorical."""
        num = self.numeric_raw(X_raw)
        ch = self.champions_raw(X_raw).astype(np.float64)
        X = np.hstack([num, ch])
        names = (
            list(self.schema.numeric_names if self.profile == "expanded" else self.schema.core_numeric_names)
            + list(self.schema.champ_names)
        )
        cat_idx = list(range(num.shape[1], X.shape[1]))
        return X, names, cat_idx

    def standardize_numeric(self, num: np.ndarray) -> np.ndarray:
        assert self.num_mean is not None and self.num_std is not None
        return ((num - self.num_mean) / self.num_std).astype(np.float32)

    def embedding_ids(self, X_raw: np.ndarray) -> np.ndarray:
        assert self.vocab is not None
        return self.vocab.transform(self.champions_raw(X_raw))

    def dims(self) -> Dict[str, Any]:
        n_num = len(self.schema.numeric_names if self.profile == "expanded" else self.schema.core_numeric_names)
        return dict(
            profile=self.profile,
            n_numeric=n_num,
            n_champ=len(self.schema.champ_names),
            onehot_width=n_num + (self.ohe.n_out if self.ohe else 0),
            lgbm_width=n_num + len(self.schema.champ_names),
            emb_vocab=self.vocab.n_vocab if self.vocab else None,
        )
