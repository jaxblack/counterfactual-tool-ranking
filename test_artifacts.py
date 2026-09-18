from pathlib import Path
import unittest

from pypdf import PdfReader

from verify_artifacts import same_statistics, verify


class ArtifactTests(unittest.TestCase):
    def test_published_statistics_and_source_hashes(self):
        result = verify(Path(__file__).parent)
        self.assertEqual(result["runs"], 45)
        self.assertEqual(result["logged_decision_instances"], 387000)

    def test_roundoff_tolerance_does_not_hide_changes(self):
        self.assertTrue(same_statistics({"mean": 0.1 + 0.2}, {"mean": 0.3}))
        self.assertFalse(same_statistics({"mean": 0.3}, {"mean": 0.30001}))
        self.assertFalse(same_statistics({"mean": float("nan")}, {"mean": float("nan")}))
        self.assertFalse(same_statistics({"mean": 0.3}, {"other": 0.3}))

    def test_pdf_has_content_and_references(self):
        reader = PdfReader(Path(__file__).parent / "paper" / "paper.pdf")
        text = [page.extract_text() for page in reader.pages]
        self.assertGreater(len(text), 5)
        self.assertTrue(all(page.strip() for page in text))
        self.assertIn("References", "\n".join(text))
        self.assertIn("Conclusion", "\n".join(text))


if __name__ == "__main__":
    unittest.main()