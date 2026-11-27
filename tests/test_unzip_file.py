# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from zipfile import ZipFile

from ultralytics.utils.downloads import unzip_file


def test_unzip_file_single_top_level_file(tmp_path: Path):
    """A zip with one top-level file extracts into a directory named after the archive."""
    archive = tmp_path / "bundle.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("note.txt", "hello")
    destination = tmp_path / "out"

    result = unzip_file(archive, destination, progress=False)

    assert result == destination / "bundle"
    assert result.is_dir()
    assert (result / "note.txt").read_text() == "hello"
