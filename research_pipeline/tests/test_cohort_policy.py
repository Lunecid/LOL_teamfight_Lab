import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cohort_policy import bucket_for, load_policy, role_for, roles_for_macro, split_roles


def test_roles_are_fully_operator_configured(tmp_path, monkeypatch):
    config = {
        "salt": "unit-test-only",
        "region_prefix": "ZZ",
        "bucket_hex_chars": 8,
        "bucket_modulus": 10,
        "patch_rules": {
            "patch-alpha": [
                {"min_bucket": 0, "max_bucket": 4, "split_role": "TRAIN_ALPHA", "macro_role": "FIT"},
                {"min_bucket": 5, "max_bucket": 9, "split_role": "EVAL_ALPHA", "macro_role": "TEST"},
            ]
        },
    }
    config_path = tmp_path / "policy.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("LOL_COHORT_POLICY_PATH", str(config_path))
    policy = load_policy()
    value = bucket_for("12345", policy)
    assert 0 <= value < 10
    assert role_for("patch-alpha", value, policy) == (
        "TRAIN_ALPHA" if value <= 4 else "EVAL_ALPHA",
        "FIT" if value <= 4 else "TEST",
    )
    assert split_roles(policy) == {"TRAIN_ALPHA", "EVAL_ALPHA"}
    assert roles_for_macro(policy, {"FIT"}) == {"TRAIN_ALPHA"}
