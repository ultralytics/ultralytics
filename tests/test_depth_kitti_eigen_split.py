# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import cv2
import numpy as np

from ultralytics.utils import ROOT, TQDM, YAML

# PR #25650: 3 KITTI Eigen test drives were missing from the old VAL_DRIVES list and leaked 72 test frames
# into train; 3 other drives were wrongly included in val despite not being part of the canonical Eigen test
# scenes. These tests execute the split-building and per-frame filtering code embedded verbatim in
# depth-kitti.yaml (not a reimplementation) to guard against a regression of either kind.
DOWNLOAD_SCRIPT = YAML.load(ROOT / "cfg/datasets/depth-kitti.yaml")["download"]


def _eigen_test():
    """Exec the EIGEN_TEST/SKIP_DRIVES-building code sliced verbatim from the dataset YAML's download script."""
    src = DOWNLOAD_SCRIPT.split("# Download the improved")[0]
    ns = {}
    exec(src, ns)
    return ns["EIGEN_TEST"]


def test_depth_kitti_eigen_split_counts():
    """The split must stay at the canonical 28 drives / 652 frames documented in the YAML header."""
    eigen_test = _eigen_test()
    assert len(eigen_test) == 28
    assert sum(len(frames) for frames in eigen_test.values()) == 652


def test_depth_kitti_eigen_split_drives():
    """The 3 drives PR #25650 moved across the split must stay on the correct side."""
    eigen_test = _eigen_test()
    # previously missing from VAL_DRIVES -> their test frames leaked into train
    fixed_into_val = {"2011_09_30_drive_0018", "2011_09_30_drive_0027", "2011_10_03_drive_0027"}
    assert fixed_into_val <= eigen_test.keys()
    # previously in VAL_DRIVES despite not being part of the canonical Eigen test scenes
    fixed_into_train = {"2011_09_29_drive_0026", "2011_10_03_drive_0034", "2011_10_03_drive_0042"}
    assert fixed_into_train.isdisjoint(eigen_test.keys())


def test_depth_kitti_eigen_split_no_train_leak(tmp_path):
    """Runs the real conversion loop on a synthetic drive to prove non-listed frames never reach train."""
    eigen_test = _eigen_test()
    leaked_drive = "2011_09_30_drive_0018"  # one of the drives PR #25650 moved into val
    frames = eigen_test[leaked_drive]
    in_list_frame = min(frames)
    out_of_list_frame = in_list_frame - 1  # adjacent raw frame, not part of the Eigen test list

    drive_dir = tmp_path / "source" / "data_depth_annotated" / f"{leaked_drive}_sync"
    for cam in ("image_02", "image_03"):
        (drive_dir / "proj_depth" / "groundtruth" / cam).mkdir(parents=True)
        raw_data_dir = tmp_path / "source" / "raw" / leaked_drive[:10] / drive_dir.name / cam / "data"
        raw_data_dir.mkdir(parents=True)
        for frame in (in_list_frame, out_of_list_frame):
            name = f"{frame:010d}.png"
            cv2.imwrite(str(drive_dir / "proj_depth" / "groundtruth" / cam / name), np.zeros((4, 4), dtype=np.uint16))
            cv2.imwrite(str(raw_data_dir / name), np.zeros((4, 4, 3), dtype=np.uint8))

    conv_src = DOWNLOAD_SCRIPT[DOWNLOAD_SCRIPT.index("# Convert: GT PNGs") :]
    conv_src = conv_src.rsplit("shutil.rmtree", 1)[0]  # keep source/ around so the test can inspect it after
    exec(conv_src, {"dir": tmp_path, "gt_drives": [drive_dir], "EIGEN_TEST": eigen_test, "cv2": cv2, "np": np, "TQDM": TQDM})

    assert list((tmp_path / "depth" / "train").iterdir()) == []
    assert list((tmp_path / "images" / "train").iterdir()) == []
    val_depth = list((tmp_path / "depth" / "val").iterdir())
    assert [p.name for p in val_depth] == [f"{leaked_drive}_02_{in_list_frame:010d}.npy"]

    # the off-list frame was dropped, not converted: its raw source file is still untouched in place
    assert (tmp_path / "source" / "raw" / leaked_drive[:10] / drive_dir.name / "image_02" / "data" / f"{out_of_list_frame:010d}.png").exists()
