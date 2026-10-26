"""Basic checks of the published experiment and clean-run layout."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StudyTests(unittest.TestCase):
    def test_original_workspace(self):
        spec = importlib.util.spec_from_file_location("reproduce", ROOT / "src/reproduce.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            work = Path(folder) / "fresh"
            module.prepare_workspace(work)
            for phase in range(1, 7):
                self.assertTrue((work / f"Phase{phase}/scripts").is_dir())
                self.assertEqual(list((work / f"Phase{phase}/outputs").iterdir()), [])
            self.assertEqual(
                module.checksum(ROOT / "reports/protocol_phase1.md"),
                module.checksum(work / "Phase1/FROZEN_PROTOCOL.md"),
            )
            self.assertTrue((work / "Phase2/scripts/run_phase2.py").is_file())

    def test_no_transport_archives(self):
        self.assertFalse((ROOT / "releases").exists())
        self.assertFalse(list(ROOT.rglob("*.b64")))
        self.assertEqual(len(list((ROOT / "results/models").glob("*.model"))), 60)
        self.assertEqual(len(list((ROOT / "results/models").glob("*.vocab"))), 60)
        original_figures = {
            "PHASE6_ENGLISH_LM.png",
            "PHASE6_ENGLISH_SEGMENTATION.png",
            "PHASE6_WHOLE_POLICY_COUNTEREXAMPLE.png",
            "PHASE6_XH_TRADEOFF.png",
        }
        exploratory_figures = {
            "metric_coupling_1888.png",
            "length_quartiles_1888.png",
            "archivelens_claim_sensitivity.png",
        }
        actual_figures = {p.name for p in (ROOT / "results/figures").glob("*.png")}
        self.assertEqual(actual_figures, original_figures | exploratory_figures)


if __name__ == "__main__":
    unittest.main()
