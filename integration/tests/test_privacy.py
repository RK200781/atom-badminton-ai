# -*- coding: utf-8 -*-
"""映像・画像が Git に入っていないことの確認(人が映るため)。

学習データ(integration/shots/shots_*.csv)は Git で共有するが、同じフォルダには録画も置かれる。
.gitignore の設定が崩れても気付けるよう、Git に登録されている・登録されそうなファイルを調べる。
git が無い環境やリポジトリの外では飛ばす。
"""

import shutil
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEDIA_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".m4v",
                    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".webp"}


def git(*args):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


@unittest.skipUnless(shutil.which("git") and (REPO / ".git").exists(), "git のリポジトリではない")
class NoMediaInGitTest(unittest.TestCase):
    def test_no_media_tracked_or_addable(self):
        tracked = git("ls-files").splitlines()
        # 除外されていない(git add すると入ってしまう)新しいファイル
        addable = git("ls-files", "--others", "--exclude-standard").splitlines()
        bad = [p for p in tracked + addable if Path(p).suffix.lower() in MEDIA_EXTENSIONS]
        self.assertEqual(bad, [], "映像・画像が Git に入っている、または入りそうです")

    def test_only_shot_csvs_in_shots_folder(self):
        files = git("ls-files", "--cached", "--others", "--exclude-standard", "integration/shots").splitlines()
        bad = [p for p in files if not (Path(p).name.startswith("shots_") and p.endswith(".csv"))]
        self.assertEqual(bad, [], "integration/shots/ には shots_*.csv 以外を Git に入れない")

    def test_recordings_are_ignored(self):
        for path in ("integration/shots/session_x.mp4", "integration/shots/session_x_frames.csv",
                     "integration/shots/anything.mov", "detection_test/recordings/rec.mp4",
                     "images/a.jpg", "some/where/clip.mkv"):
            result = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO)
            self.assertEqual(result.returncode, 0, f"{path} が除外されていません")
        result = subprocess.run(["git", "check-ignore", "-q", "--no-index", "integration/shots/shots_x.csv"], cwd=REPO)
        self.assertEqual(result.returncode, 1, "学習データ shots_*.csv は共有する(除外しない)")


if __name__ == "__main__":
    unittest.main()
