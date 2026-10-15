"""Offline regression checks: python3 -m unittest discover -s research_updates/state_of_ai_2025_report."""

from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from urllib.parse import unquote, urlsplit

from build import normalize_image_paths


REPORT_DIR = Path(__file__).resolve().parent


class ImageSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == "img":
            self.sources.append(dict(attrs)["src"])


class ReportBuildTests(unittest.TestCase):
    def test_mixed_image_paths(self):
        source = '''<img src="../images/chart.png?size=2#figure">
<IMG alt="Chart" SRC = '../images/chart.png'>
<img src="images/already-correct.png">
<img src="https://example.com/images/chart.png">
<img src="//example.com/images/chart.png">
<img src="data:image/png;base64,abc">
<a href="../images/chart.png">Download</a>'''
        expected = source.replace('src="../images/', 'src="images/').replace(
            "SRC = '../images/", "SRC = 'images/"
        )
        self.assertEqual(normalize_image_paths(source), expected)
        self.assertEqual(normalize_image_paths(expected), expected)

    def test_rebuild_from_another_directory_preserves_report_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report"
            report.mkdir()
            shutil.copy2(REPORT_DIR / "build.py", report)
            shutil.copytree(REPORT_DIR / "sections", report / "sections")
            for _ in range(2):
                subprocess.run(
                    [sys.executable, str(report / "build.py")],
                    cwd=tmp, check=True, capture_output=True,
                )
                output = (report / "index.html").read_text(encoding="utf-8")
                self.assertEqual(output, (REPORT_DIR / "index.html").read_text(encoding="utf-8"))
            images = ImageSources()
            images.feed(output)
            self.assertTrue(images.sources)
            for source in images.sources:
                url = urlsplit(source)
                if not url.scheme and not url.netloc:
                    with self.subTest(source=source):
                        self.assertTrue((REPORT_DIR / unquote(url.path)).is_file())
            self.assertFalse((Path(tmp) / "index.html").exists())


if __name__ == "__main__":
    unittest.main()
