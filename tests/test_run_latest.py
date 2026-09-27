import tempfile
import unittest
from pathlib import Path

from run_latest import newest_video


class NewestVideoTests(unittest.TestCase):
    def test_selects_newest_supported_video_case_insensitively(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            old = folder / "old.MOV"
            new = folder / "new.webm"
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            old.touch()
            new.touch()
            old_mtime = old.stat().st_mtime_ns
            new_mtime = old_mtime + 1_000_000
            import os

            os.utime(old, ns=(old_mtime, old_mtime))
            os.utime(new, ns=(new_mtime, new_mtime))

            self.assertEqual(newest_video(folder), new)

    def test_ignores_non_video_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "README.txt").write_text("not a video")

            with self.assertRaises(FileNotFoundError):
                newest_video(folder)


if __name__ == "__main__":
    unittest.main()
