"""A source refresh must not overwrite a derived module layout."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sync_overlay", ROOT / "tools/sync_from_sources.py")
SYNC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYNC)


class SyncOverlayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.hub = Path(self.temp.name)
        marker = self.hub / "master_ver/module-layout.json"
        marker.parent.mkdir()
        marker.write_text('{"schema":"module-layout-v1"}', encoding="utf-8")
        patch = mock.patch.object(SYNC, "HUB", self.hub)
        patch.start()
        self.addCleanup(patch.stop)

    def test_apply_refuses_before_reading_original_sources(self):
        for target in ("master_ver", "all"):
            with self.subTest(target=target), mock.patch("sys.argv", ["sync", target, "--apply"]), \
                 mock.patch.object(SYNC, "plan_master_ver") as plan, \
                 mock.patch.object(SYNC, "plan_research_pipeline") as other:
                with self.assertRaisesRegex(SystemExit, "derived module layout"):
                    SYNC.main()
                plan.assert_not_called()
                other.assert_not_called()

    def test_dry_run_and_other_target_are_allowed(self):
        for target, apply in (("master_ver", False), ("all", False), ("research_pipeline", True)):
            with self.subTest(target=target, apply=apply):
                SYNC.refuse_overlay_overwrite(target, apply)

    def test_unmodified_mirror_remains_refreshable(self):
        (self.hub / "master_ver/module-layout.json").unlink()
        SYNC.refuse_overlay_overwrite("master_ver", True)
        SYNC.refuse_overlay_overwrite("all", True)


if __name__ == "__main__":
    unittest.main()
