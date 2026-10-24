"""Focused checks of the E2b/E2c rules (protocol cells, sealing columns, layout, gates); no data run, no label read."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.external_validation_20261009 import e2bc_common as B  # noqa: E402


def test_protocol_cells_match_the_protocol():
    from studies.external_validation_20261009.e2c_scores import cell_keys, protocol_cells

    cells = protocol_cells()
    assert cells["P1"] == ["final__module__LGBM_S1_A1_M1_E1_C1_R1_D1__lightgbm__selected~lightgbm"]
    assert cells["PT_flex"] == ["final__reference__PT_FLEX__PT_flex__selected~PT_flex"]
    assert len(cells["P2_quads_C1"]) == len(cells["P2_quads_C0"]) == 8 and len(cells["position_effect"]) == 4
    assert all("_C1_" in k for k in cells["P2_quads_C1"]) and all("_C0_" in k for k in cells["P2_quads_C0"])
    keys = cell_keys()
    assert len(keys) == len(set(keys)) == 23


def test_sealed_columns_cover_every_label_bearing_column():
    for name in ("direction_positive", "svi_delta", "exact_zero", "near_zero_bits", "p_post", "logit_post", "raw_post", "raw_delta",
                 "clip_saturation_tie_bits"):
        assert name in B.SEALED_LABEL_COLUMNS
    for public in ("p_pre", "logit_pre", "raw_pre", "valid", "invalid_bits", "query_pre"):
        assert public not in B.SEALED_LABEL_COLUMNS


def test_layout_and_datasets():
    assert B.labels_dir("kr_16_13") == B.C.DATA_ROOT / "labels" / "kr_16_13"
    assert B.sealed_labels_path("na1_16_13").name == "EXTERNAL_LABELS_SEALED_na1_16_13.npz"
    assert set(B.EXTERNAL_DATASETS) == {"parity_15_16", "kr_15_18_22", "kr_16_13", "na1_16_13"}


def test_cpu_gate_caps_workers():
    assert 1 <= B.cpu_workers_allowed(6) <= 6
    assert B.cpu_workers_allowed(1) == 1


def test_external_surface_overrides_only_the_role_decoding():
    cls = B.make_surface_class()
    from studies.label_production_20261005.queries import QuerySurface

    assert issubclass(cls, QuerySurface) and "_records_for_match" in cls.__dict__
    assert "iter_query_rows" not in cls.__dict__
