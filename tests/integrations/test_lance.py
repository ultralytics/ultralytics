from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.data.embedding import DatasetUtil

DEPS = ['lancedb']

check_requirements(DEPS)

# YOLODatabase tests

def test_embeddings_creation():
    ds = DatasetUtil("coco8.yaml")
    ds.build_embeddings()
    assert ds.table_name == "coco8.yaml", "the table name should be coco8.yaml"
    assert len(ds.table.to_arrow()) == 4, "the length of the embeddings table should be 8"

def test_sim_idx():
    ds = DatasetUtil("coco8.yaml")
    ds.build_embeddings()

    idx = ds.get_similarity_index(0, 1) # get all imgs
    assert len(idx) == 4, "the length of the similar index should be 8"


def test_copy_embeddings_from_table():
    project = "runs/test/temp/"
    ds = DatasetUtil("coco8.yaml", project=project)
    ds.build_embeddings()

    table = project + ds.table_name + ".lance"
    ds2 = DatasetUtil(table=table)
    assert ds2.table_name == "coco8.yaml", "the table name should be coco8.yaml"

def test_operations():
    ds = DatasetUtil('coco8.yaml')
    ds.build_embeddings('yolov8n.pt')

    #ds.plot_similar_imgs(4, 10)
    #ds.plot_similirity_index()
    sim = ds.get_similarity_index()
    paths, ids = ds.get_similar_imgs(3, 10)
    ds.remove_imgs(ids[0])
    ds.reset()
    ds.log_status()
    ds.remove_imgs([0, 1])
    ds.remove_imgs([0])

    ds.persist()
    assert len(ds.table.to_arrow()) == 1, "the length of the embeddings table should be 1"

