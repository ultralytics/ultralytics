# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics import Explorer
from ultralytics.utils import ASSETS

import PIL


def test_similarity():
    """Test similarity calculations and SQL queries for correctness and response length."""
    exp = Explorer()
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=1)
    assert len(similar) == 25
    similar = exp.get_similar(img=ASSETS / 'zidane.jpg')
    assert len(similar) == 25
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) == 10
    sim_idx = exp.similarity_index()
    assert len(sim_idx) > 0
    sql = exp.sql_query("WHERE labels LIKE '%person%'")
    assert len(sql) > 0


def test_det():
    """Test detection functionalities and ensure the embedding table has bounding boxes."""
    exp = Explorer(data='coco8.yaml', model='yolov8n.pt')
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()['bboxes']) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    # This is a loose test, just checks errors not correctness
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)


def test_seg():
    """Test segmentation functionalities and verify the embedding table includes masks."""
    exp = Explorer(data='coco8-seg.yaml', model='yolov8n-seg.pt')
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()['masks']) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)


def test_pose():
    """Test pose estimation functionalities and check the embedding table for keypoints."""
    exp = Explorer(data='coco8-pose.yaml', model='yolov8n-pose.pt')
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()['keypoints']) > 0
    similar = exp.get_similar(idx=[1, 2], limit=10)
    assert len(similar) > 0
    similar = exp.plot_similar(idx=[1, 2], limit=10)
    assert isinstance(similar, PIL.Image.Image)
