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



