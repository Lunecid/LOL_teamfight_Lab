#!/usr/bin/env python3
"""Synthetic PACK-1 / left-align checks (no game data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    # API contract
    x_right = torch.tensor([[[0.0], [0.0], [0.0], [10.0], [20.0]]])
    packed_right = pack_padded_sequence(
        x_right, lengths=torch.tensor([2]), batch_first=True, enforce_sorted=False
    )
    x_left = torch.tensor([[[10.0], [20.0], [0.0], [0.0], [0.0]]])
    packed_left = pack_padded_sequence(
        x_left, lengths=torch.tensor([2]), batch_first=True, enforce_sorted=False
    )

    # builder left-align
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "w3", REPO / "scripts" / "rr20260919_v_redesign_fit_wave3_tier23.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    match = np.array(["m", "m", "m"])
    tmin = np.array([1.0, 2.0, 3.0])
    X = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    seq_l, mask_l, _ = mod.build_history_stacks(match, tmin, X, K=5, align="left")
    seq_r, mask_r, _ = mod.build_history_stacks(match, tmin, X, K=5, align="right")
    # last row should have length 3
    assert mask_l[2].sum() == 3 and list(seq_l[2, :3, 0]) == [1.0, 2.0, 3.0]
    assert mask_r[2].sum() == 3 and list(seq_r[2, 2:, 0]) == [1.0, 2.0, 3.0]
    # pack left-aligned last row
    xb = torch.from_numpy(seq_l[2:3])
    mb = torch.from_numpy(mask_l[2:3])
    lengths = mb.sum(dim=1).long()
    packed = pack_padded_sequence(xb, lengths, batch_first=True, enforce_sorted=False)
    packed_vals = packed.data.view(-1).tolist()

    out = dict(
        packed_right_api=packed_right.data.view(-1).tolist(),
        packed_left_api=packed_left.data.view(-1).tolist(),
        builder_left_packed=packed_vals,
        pack1_pass=(
            packed_right.data.view(-1).tolist() == [0.0, 0.0]
            and packed_left.data.view(-1).tolist() == [10.0, 20.0]
            and packed_vals == [1.0, 2.0, 3.0]
        ),
        history_align_default="left",
    )
    path = REPO / "docs" / "V_MODEL_INPUT_DESIGN_20260919" / "synthetic_checks_local.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not out["pack1_pass"]:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
