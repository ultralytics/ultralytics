import pytest

from ultralytics import Explorer

try:
    import lancedb
    import sklearn
except ImportError:
    lancedb = False
    sklearn = False


@pytest.mark.skipif(not lancedb or not sklearn, reason='requires lancedb and sklearn')
class TestExplorer:

    def test_embeddings_creation(self):
        ds = Explorer('coco8.yaml')
        ds.build_embeddings()
        assert ds.table_name == 'coco8.yaml', 'the table name should be coco8.yaml'
        assert len(ds.table.to_arrow()) == 4, 'the length of the embeddings table should be 8'

    def test_sim_idx(self):
        ds = Explorer('coco8.yaml')
        ds.build_embeddings()

        idx = ds.get_similarity_index(0, 1)  # get all imgs
        assert len(idx) == 4, 'the length of the similar index should be 8'

    def test_copy_embeddings_from_table(self):
        project = 'runs/test/temp/'
        ds = Explorer('coco8.yaml', project=project)
        ds.build_embeddings()

        table = project + ds.table_name + '.lance'
        ds2 = Explorer(table=table)
        assert ds2.table_name == 'coco8.yaml', 'the table name should be coco8.yaml'

    def test_operations(self):
        ds = Explorer('coco8.yaml')
        ds.build_embeddings('yolov8n.pt')

        sim = ds.get_similarity_index()
        assert sim.shape[0] == 4, 'the length of the embeddings table should be 1'

        _, ids = ds.get_similar_imgs(3, 10)
        ds.remove_imgs(ids[0])
        ds.reset()
        ds.log_status()
        ds.remove_imgs([0, 1])
        ds.remove_imgs([0])
        assert len(ds.table.to_arrow()) == 1, 'the length of the embeddings table should be 1'
        ds.persist()
        assert len(ds.table.to_arrow()) == 1, 'the length of the embeddings table should be 1'
