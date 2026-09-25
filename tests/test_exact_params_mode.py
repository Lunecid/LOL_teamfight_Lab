"""ExactParams.from_cfg: the cfg's ENG_* switches must agree with the requested detector mode."""
from __future__ import annotations

import copy

import pytest

from core.config import cfg
from core.presets import apply_preset
from gameplay.exact_population import ExactParams

V4_FLAGS = {"ENG_ALIVE_SOURCE": "event", "ENG_PARTICIPATION": "kill_credit", "ENG_OVERLAP_RULE": "none",
            "ENG_MERGE_A6": True, "ENG_ISOLATION": True}


def _cfg(preset=None, **over):
    c = copy.copy(cfg)
    if preset:
        apply_preset(c, preset)
    for k, v in over.items():
        setattr(c, k, v)
    return c


def test_v4_exact_preset_passes_for_mode_v4():
    c = _cfg("v4-exact")
    for k, v in V4_FLAGS.items():
        assert getattr(c, k) == v, k
    p = ExactParams.from_cfg(c, "v4")
    assert p == ExactParams.from_cfg(c)                        # default mode is 'v4'
    assert (p.gap_ms, p.diameter, p.clean_max_age_ms) == (int(c.TF2_KILL_CLUSTER_GAP_MS),
                                                          float(c.CLUSTER_MAX_DIAMETER), 10000)
    assert p.merge is True


@pytest.mark.parametrize("key,bad", [("ENG_OVERLAP_RULE", "legacy_priority"), ("ENG_ISOLATION", False),
                                     ("ENG_ALIVE_SOURCE", "frame"), ("ENG_PARTICIPATION", "kill_plus_interactions"),
                                     ("ENG_MERGE_A6", False), ("CONTINUOUS_FIGHT_MERGE", False),
                                     ("ENG_ISOLATION", "yes"), ("ENG_MERGE_A6", None)])
def test_v4_rejects_each_inconsistent_flag(key, bad):
    c = _cfg("v4-exact", **{key: bad})
    with pytest.raises(ValueError, match=key):
        ExactParams.from_cfg(c, "v4")


def test_v4_rejects_v33_defaults_and_preset():
    with pytest.raises(ValueError, match="inconsistent with detector mode 'v4'"):
        ExactParams.from_cfg(_cfg(), "v4")
    with pytest.raises(ValueError):
        ExactParams.from_cfg(_cfg("v3.3"))
    # the global cfg object is never modified by the check
    assert getattr(cfg, "ENG_OVERLAP_RULE") == "legacy_priority"


def test_missing_flag_is_reported():
    class Bare:
        TF2_KILL_CLUSTER_GAP_MS = 14000
        CLUSTER_MAX_DIAMETER = 4300.0
    with pytest.raises(ValueError, match="<missing>"):
        ExactParams.from_cfg(Bare(), "v4")
    assert ExactParams.from_cfg(Bare(), None).gap_ms == 14000   # explicit opt-out


def test_string_flags_are_case_insensitive():
    c = _cfg("v4-exact", ENG_OVERLAP_RULE="None", ENG_ALIVE_SOURCE=" Event ")
    assert ExactParams.from_cfg(c, "v4").diameter == float(c.CLUSTER_MAX_DIAMETER)


def test_r2_repro_mode_flags():
    r2 = dict(ENG_ALIVE_SOURCE="event", ENG_PARTICIPATION="kill_credit", ENG_OVERLAP_RULE="legacy_priority",
              ENG_MERGE_A6=False, ENG_ISOLATION=False)
    c = _cfg("v3.3", **r2)
    p = ExactParams.from_cfg(c, "r2_repro")
    assert (p.gap_ms, p.diameter) == (13700, 4264.0)
    with pytest.raises(ValueError):
        ExactParams.from_cfg(_cfg("v4-exact"), "r2_repro")      # v4 flags are not the R2 prototype
    with pytest.raises(ValueError):
        ExactParams.from_cfg(c, "v4")


def test_unknown_mode_and_lock():
    c = _cfg("v4-exact")
    with pytest.raises(ValueError, match="unknown mode"):
        ExactParams.from_cfg(c, "legacy")
    c.ENG_BOUNDARIES_LOCKED = False
    with pytest.raises(ValueError, match="LOCKED"):
        ExactParams.from_cfg(c, "v4", require_locked=True)
    c.ENG_BOUNDARIES_LOCKED = True
    assert ExactParams.from_cfg(c, "v4", require_locked=True).gap_ms == int(c.TF2_KILL_CLUSTER_GAP_MS)
