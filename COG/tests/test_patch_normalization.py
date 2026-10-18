from gameplay import fights


def test_normalize_patch_canonical_major_minor(monkeypatch):
    monkeypatch.setattr(fights.cfg, "PATCH_LEVEL", "major_minor", raising=False)
    assert fights.normalize_patch("15.20.719.545") == "15.20"


def test_normalize_patch_handles_noisy_prefix(monkeypatch):
    monkeypatch.setattr(fights.cfg, "PATCH_LEVEL", "major_minor", raising=False)
    assert fights.normalize_patch("Version 15.20.719.545") == "15.20"


def test_normalize_patch_full_level_passthrough(monkeypatch):
    monkeypatch.setattr(fights.cfg, "PATCH_LEVEL", "full", raising=False)
    assert fights.normalize_patch("15.20.719.545") == "15.20.719.545"
