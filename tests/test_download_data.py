import io
import tarfile
from pathlib import Path

import pytest

from scripts.download_data import Archive, dataset_ready, extract_archive, safe_member_path


def add_file(archive: tarfile.TarFile, name: str, value: bytes = b"x") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    archive.addfile(info, io.BytesIO(value))


def test_safe_member_path_rejects_archive_traversal():
    with pytest.raises(RuntimeError, match="Небезопасный"):
        safe_member_path("../outside.txt")
    with pytest.raises(RuntimeError, match="Небезопасный"):
        safe_member_path("/absolute.txt")


def test_train_archive_is_extracted_to_stable_data_path(tmp_path):
    archive = tmp_path / "train.tar"
    with tarfile.open(archive, "w") as stream:
        add_file(stream, "train/af/masks/AF_1_MASK.tif")
        add_file(stream, "train/bs/masks/BS_1_MASK.tif")
    spec = Archive("/train.tar", "train.tar", "train", archive.stat().st_size, "unused")
    target = extract_archive(archive, tmp_path / "data", spec, False)
    assert target == tmp_path / "data" / "train"
    assert dataset_ready(target, "train")
    assert (target / ".source.json").is_file()


def test_test_dataset_requires_submission_template(tmp_path):
    root = tmp_path / "test"
    (root / "af").mkdir(parents=True)
    (root / "bs").mkdir()
    assert not dataset_ready(root, "test")
    (root / "sample_submission.csv").write_text("chip_id,class_id,rle\n", encoding="utf-8")
    assert dataset_ready(root, "test")
