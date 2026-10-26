"""Check that the recovered research source still matches the original archives.

This tests file provenance and Python syntax. It does not claim a fresh rerun
of the older experiments or the separate historical inference benchmarks.
"""
from pathlib import Path
import ast
import csv
import hashlib
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "recovery_manifest.csv"


class RecoveredResearchTests(unittest.TestCase):
    def test_original_files_and_source_syntax(self):
        with MANIFEST.open(encoding="utf-8", newline="") as stream:
            items = list(csv.DictReader(stream))
        self.assertEqual(len(items), 413, "The original research recovery is incomplete")
        python_count = 0
        for item in items:
            relative = Path(item["path"])
            self.assertFalse(relative.is_absolute())
            self.assertNotIn("..", relative.parts)
            self.assertIn(relative.parts[0], ("src", "results"))
            path = ROOT / relative
            self.assertTrue(path.is_file(), str(relative))
            content = path.read_bytes()
            self.assertEqual(int(item["size_bytes"]), len(content), str(relative))
            self.assertEqual(hashlib.sha256(content).hexdigest(), item["source_sha256"], str(relative))
            if relative.suffix == ".py":
                ast.parse(content.decode("utf-8"), filename=str(relative))
                python_count += 1
        self.assertEqual(python_count, 272)

    def test_separate_research_is_not_substituted_for_coursework(self):
        self.assertTrue((ROOT / "src" / "phase2" / "run_phase2.py").is_file())
        self.assertTrue((ROOT / "src" / "earlier" / "phase33" / "scripts" / "run_phase33.py").is_file())
        self.assertTrue((ROOT / "src" / "historical" / "phase04" / "01_recovery_value.py").is_file())
        self.assertTrue((ROOT / "reports" / "report.tex").is_file())


if __name__ == "__main__":
    unittest.main()
