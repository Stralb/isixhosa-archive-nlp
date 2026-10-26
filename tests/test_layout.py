"""Check native research outputs and reconstruction of an empty frozen workspace."""
from __future__ import annotations
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StudyLayoutTests(unittest.TestCase):
    def test_native_artifacts_are_real_files(self):
        self.assertFalse((ROOT / "releases").exists())
        self.assertFalse((ROOT / "coursework").exists())
        self.assertFalse(list(ROOT.rglob("*.b64")))
        self.assertEqual(len(list((ROOT / "results").rglob("*.model"))), 60)
        self.assertEqual(len(list((ROOT / "results").rglob("*.vocab"))), 60)
        self.assertEqual(len(list((ROOT / "results" / "figures").glob("*.png"))), 4)
        for name in ("Phase3/outputs/heldout_scores.csv",
                     "Phase5/outputs/heldout_lm_scores.csv"):
            self.assertGreater((ROOT / "results" / name).stat().st_size, 10_000_000)

    def test_original_scripts_build_a_clean_workspace(self):
        spec = importlib.util.spec_from_file_location(
            "reproduce_all", ROOT / "scripts" / "reproduce_all.py"
        )
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            work = Path(td) / "fresh"
            module.prepare_workspace(work)
            for phase in range(1, 7):
                self.assertTrue((work / f"Phase{phase}" / "scripts").is_dir())
                self.assertTrue((work / f"Phase{phase}" / "outputs").is_dir())
                self.assertFalse(list((work / f"Phase{phase}" / "outputs").iterdir()))
            for phase, name in ((1, "FROZEN_PROTOCOL.md"),
                                (4, "PRECOMMIT.md"), (5, "PRECOMMIT.md")):
                original = (ROOT / "protocols" / f"Phase{phase}" / name).read_bytes()
                staged = (work / f"Phase{phase}" / name).read_bytes()
                self.assertEqual(hashlib.sha256(original).digest(),
                                 hashlib.sha256(staged).digest())
            self.assertTrue((work / "Phase2/scripts/run_phase2.py").is_file())
            self.assertTrue((work / "Phase6/scripts/audit_phase6.py").is_file())


if __name__ == "__main__":
    unittest.main()
