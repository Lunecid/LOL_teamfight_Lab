"""Focused checks of the E2a role coding, layout, parity hash and the static-lookup diagnostic hook (no data run)."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from studies.external_validation_20261009 import e2a_common as C  # noqa: E402


def test_external_codes_are_outside_the_frozen_tuples():
    from studies.v_preparation_20261002.run import PATCH_ROLES, ROLES

    assert C.EXTERNAL_ROLE not in ROLES and C.EXTERNAL_ROLE not in PATCH_ROLES
    assert C.EXTERNAL_ROLE_CODE == len(ROLES) and C.EXTERNAL_ROLE_CODE != ROLES.index("TEST")
    assert C.EXTERNAL_PATCH_ROLE_CODE == len(PATCH_ROLES)


def test_layout_and_parity_hash():
    assert C.stage_dir("parity_15_16", "bank") == C.DATA_ROOT / "parity_15_16" / "bank"
    assert C.stage_dir("kr_16_13", "history") == C.DATA_ROOT / "history" / "kr_16_13"
    expected = hashlib.sha256(b"parity_20261009:KR_7758739368").hexdigest()
    assert C.parity_hash("KR_7758739368") == expected
    assert C.canonical_key(" KR_1 ") == hashlib.sha256(b"KR_1").hexdigest()


def test_static_lookup_hook_delegates_for_frozen_patches():
    from studies.observable_state_20261002.categorical import static_lookup_counts
    from studies.external_validation_20261009.e2a_bank import external_static_lookup_counts

    values = np.zeros((10, 14), dtype=np.int64)
    values[:, 0] = 1
    bans = np.full((2, 5), -1, dtype=np.int64)
    assert external_static_lookup_counts(values, bans, "15.16") == static_lookup_counts(values, bans, "15.16")


def test_static_lookup_hook_later_patch_vocabulary():
    from studies.external_validation_20261009.e2a_bank import external_static_lookup_counts

    values = np.zeros((10, 14), dtype=np.int64)
    values[:, 0] = 1            # Annie: present in every Data Dragon championFull
    values[:, 1] = 4            # Flash: no offline summoner table after 15.16
    values[0, 0] = 999999       # not a champion
    bans = np.asarray([[1, -1, 0, 999999, 1], [1, 1, 1, 1, 1]], dtype=np.int64)
    out = external_static_lookup_counts(values, bans, "16.13")
    assert out["champion_id"] == {"positive_ID_lookup_missing": 1, "canonical_ID_known": 9}
    assert out["summoner_spell_1_id"] == {"recorded_ID_map_unavailable": 10}
    assert out["primary_rune_1"] == {"nonpositive_cache_ID_presence_unknown": 10}
    assert out["bans"] == {"canonical_ID_known": 7, "nonpositive_cache_ID_presence_unknown": 2, "positive_ID_lookup_missing": 1}
