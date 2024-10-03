# Ultralytics YOLO 🚀, AGPL-3.0 license

import PIL
import pytest

from ultralytics import Explorer
from ultralytics.utils import ASSETS
from ultralytics.utils.torch_utils import TORCH_1_13


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_13, reason="Explorer requires torch>=1.13")
def test_similarity():
    """Test the correctness and response length of similarity calculations and SQL queries in the Explorer."""
    exp = Explorer(data="coco8.yaml")
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=1)
    assert len(similar) == 4
    similar = exp.get_similar(img=ASSETS / "bus.jpg")
    assert len(similar) == 4
    similar = exp.get_similar(idx=[1, 2], limit=2)
    assert len(similar) == 2
    sim_idx = exp.similarity_index()
    assert len(sim_idx) == 4
    sql = exp.sql_query("WHERE labels LIKE '%zebra%'")
    assert len(sql) == 1


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_13, reason="Explorer requires torch>=1.13")
def test_det():
    """Test detection functionalities and verify embedding table includes bounding boxes."""
    exp = Explorer(data="coco8.yaml", model="yolo11n.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["bboxes"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_13, reason="Explorer requires torch>=1.13")
def test_seg():
    """Test segmentation functionalities and ensure the embedding table includes segmentation masks."""
    exp = Explorer(data="coco8-seg.yaml", model="yolo11n-seg.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["masks"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)


@pytest.mark.slow
@pytest.mark.skipif(not TORCH_1_13, reason="Explorer requires torch>=1.13")
def test_pose():
    """Test pose estimation functionality and verify the embedding table includes keypoints."""
    exp = Explorer(data="coco8-pose.yaml", model="yolo11n-pose.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["keypoints"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)
