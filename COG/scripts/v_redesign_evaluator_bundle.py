#!/usr/bin/env python3
"""Serialize / score a frozen V evaluator bundle (preproc + f + g).

Bundle contents (collaborator freeze-prep):
  - feature schema + profile
  - numeric mean/std, champion vocab / one-hot cats
  - model kind + weights/config
  - PosSlopeSigmoid calibration
  - fit scope metadata (fit85 vs full TRAIN)
  - input_impl / data version hashes
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from v_redesign_feature_adapters import (
    ChampVocab,
    FeatureSchema,
    OneHotEncoder,
    ProfileBundle,
)


class PosSlopeSigmoid:
    def __init__(self, coef: float = 1.0, intercept: float = 0.0, ok: bool = True):
        self.coef_, self.intercept_, self.ok = float(coef), float(intercept), bool(ok)

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.ok:
            return p
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        return 1.0 / (1.0 + np.exp(-(self.coef_ * logit + self.intercept_)))

    def to_dict(self) -> Dict[str, Any]:
        return dict(coef=self.coef_, intercept=self.intercept_, ok=self.ok)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PosSlopeSigmoid":
        return cls(d["coef"], d["intercept"], d.get("ok", True))


def schema_to_dict(schema: FeatureSchema) -> Dict[str, Any]:
    return dict(
        raw_names=list(schema.raw_names),
        keep_names=list(schema.keep_names),
        keep_idx=schema.keep_idx.tolist(),
        numeric_names=list(schema.numeric_names),
        numeric_idx_in_keep=schema.numeric_idx_in_keep.tolist(),
        champ_names=list(schema.champ_names),
        champ_idx_in_keep=schema.champ_idx_in_keep.tolist(),
        core_numeric_names=list(schema.core_numeric_names),
        core_numeric_idx_in_keep=schema.core_numeric_idx_in_keep.tolist(),
        groups=dict(schema.groups),
    )


def schema_from_dict(d: Dict[str, Any]) -> FeatureSchema:
    return FeatureSchema(
        raw_names=list(d["raw_names"]),
        keep_names=list(d["keep_names"]),
        keep_idx=np.asarray(d["keep_idx"], dtype=np.int64),
        numeric_names=list(d["numeric_names"]),
        numeric_idx_in_keep=np.asarray(d["numeric_idx_in_keep"], dtype=np.int64),
        champ_names=list(d["champ_names"]),
        champ_idx_in_keep=np.asarray(d["champ_idx_in_keep"], dtype=np.int64),
        core_numeric_names=list(d["core_numeric_names"]),
        core_numeric_idx_in_keep=np.asarray(d["core_numeric_idx_in_keep"], dtype=np.int64),
        groups=dict(d["groups"]),
    )


def bundle_to_dict(bun: ProfileBundle) -> Dict[str, Any]:
    assert bun.ohe is not None and bun.vocab is not None
    assert bun.num_mean is not None and bun.num_std is not None
    return dict(
        profile=bun.profile,
        schema=schema_to_dict(bun.schema),
        ohe_slot_categories=bun.ohe.slot_categories,
        vocab_id_to_ix={str(k): int(v) for k, v in bun.vocab.id_to_ix.items()},
        vocab_n=int(bun.vocab.n_vocab),
        num_mean=bun.num_mean.tolist(),
        num_std=bun.num_std.tolist(),
        dims=bun.dims(),
    )


def bundle_from_dict(d: Dict[str, Any]) -> ProfileBundle:
    schema = schema_from_dict(d["schema"])
    bun = ProfileBundle(schema, d["profile"])
    bun.ohe = OneHotEncoder(slot_categories=[list(map(int, s)) for s in d["ohe_slot_categories"]])
    bun.vocab = ChampVocab(
        id_to_ix={int(k): int(v) for k, v in d["vocab_id_to_ix"].items()},
        n_vocab=int(d["vocab_n"]),
    )
    bun.num_mean = np.asarray(d["num_mean"], dtype=np.float64)
    bun.num_std = np.asarray(d["num_std"], dtype=np.float64)
    return bun


def save_evaluator(path: Path, payload: Dict[str, Any]) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)


def load_evaluator(path: Path) -> Dict[str, Any]:
    import joblib

    return joblib.load(path)


def predict_raw_mlp(ev: Dict[str, Any], X_raw: np.ndarray) -> np.ndarray:
    """Raw f(x) for embedding MLP evaluator."""
    import torch
    import torch.nn as nn

    bun = bundle_from_dict(ev["preproc"])
    pack = ev["model"]
    num = bun.standardize_numeric(bun.numeric_raw(X_raw))
    ids = bun.embedding_ids(X_raw)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(pack["n_vocab"], pack["emb_dim"], padding_idx=0)
            d = pack["d_num"] + pack["n_slots"] * pack["emb_dim"]
            layers = []
            for h in pack["hidden"]:
                layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(pack["dropout"])]
                d = h
            layers += [nn.Linear(d, 1)]
            self.mlp = nn.Sequential(*layers)

        def forward(self, num_t, ids_t):
            e = self.emb(ids_t).reshape(ids_t.size(0), -1)
            return self.mlp(torch.cat([num_t, e], dim=-1)).squeeze(-1)

    net = Net()
    net.load_state_dict(pack["state_dict"])
    net.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(num), 65536):
            xb = torch.from_numpy(num[i : i + 65536].astype(np.float32))
            ib = torch.from_numpy(ids[i : i + 65536].astype(np.int64))
            outs.append(torch.sigmoid(net(xb, ib)).numpy())
    return np.concatenate(outs)


def predict_calibrated(ev: Dict[str, Any], X_raw: np.ndarray) -> np.ndarray:
    raw = predict_raw_mlp(ev, X_raw)
    g = PosSlopeSigmoid.from_dict(ev["calibration"])
    return g.transform(raw)


def build_mlp_evaluator_payload(
    *,
    name: str,
    bun: ProfileBundle,
    model_pack: Dict[str, Any],
    calibration: Dict[str, Any],
    fit_scope: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return dict(
        evaluator_id=name,
        kind="mlp_embedding",
        input_impl="corrected_v1_CAT1_PACK1_CAL1",
        fit_scope=fit_scope,  # e.g. train_fit85_match_holdout_seed7
        preproc=bundle_to_dict(bun),
        model=dict(model_pack),
        calibration=dict(calibration),
        evaluation_map="V = g(f(T(X)))",
        meta=meta or {},
    )
