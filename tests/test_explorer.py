from ultralytics import Explorer

def test_similarity():
    exp = Explorer()
    exp.create_embeddings_table()
    similar = exp.get_similar(idx=1)
    assert len(similar) == 25
    similar = exp.get_similar(img="https://ultralytics.com/images/zidane.jpg")
    assert len(similar) == 25
    similar = exp.get_similar(idx=[1,2], limit=10)
    assert len(similar) == 10

def test_det():
    exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["bboxes"]) > 0

def test_pose():
    exp = Explorer(data="coco8-pose.yaml", model="yolov8n-pose.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["keypoints"]) > 0

def test_seg():
    exp = Explorer(data="coco8-seg.yaml", model="yolov8n-seg.pt")
    exp.create_embeddings_table(force=True)
    assert len(exp.table.head()["masks"]) > 0
    # exp.plot_similar(idx=1)
