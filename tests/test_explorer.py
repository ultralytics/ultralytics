# Ultralytics YOLO 🚀, AGPL-3.0 license

import PIL
import pytest

from ultralytics import Explorer
from ultralytics.utils import ASSETS


@pytest.mark.slow
def test_similarity():
    """Test similarity calculations and SQL queries for correctness and response length."""
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
def test_det():
    """Test detection functionalities and ensure the embedding table has bounding boxes."""
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["bboxes"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)


@pytest.mark.slow
def test_seg():
    """Test segmentation functionalities and verify the embedding table includes masks."""
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["masks"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)


@pytest.mark.slow
def test_pose():
    """Test pose estimation functionalities and check the embedding table for keypoints."""
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["keypoints"]) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)
