import pytest

from acquisition.config import (
    CollectorConfig,
    parse_patch,
    patch_from_game_version,
    public_patch_label,
)


def test_api_and_public_patch_labels_are_explicit():
    assert parse_patch("16.13") == (16, 13)
    assert patch_from_game_version("16.13.123.456") == (16, 13)
    assert public_patch_label((16, 13)) == "26.13"
    assert public_patch_label((15, 24)) is None


def test_patch_parser_rejects_ambiguous_values():
    with pytest.raises(ValueError):
        parse_patch("26")


def test_cycle_work_budget_must_leave_room_for_hourly_refresh(tmp_path):
    with pytest.raises(ValueError):
        CollectorConfig(
            key_file=tmp_path / "key.env",
            output_root=tmp_path / "raw",
            state_dir=tmp_path / "state",
            cycle_interval_seconds=3600,
            cycle_work_budget_seconds=3600,
        )
