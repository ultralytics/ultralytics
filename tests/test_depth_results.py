import numpy as np

from ultralytics.engine.results import Results, DepthMap


def test_results_depth_field():
    img = np.zeros((20, 24, 3), dtype=np.uint8)
    depth = np.random.rand(20, 24).astype(np.float32)
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"}, depth=depth)
    assert isinstance(r.depth, DepthMap)
    assert r.depth.data.shape == (20, 24)
    rc = r.cpu().numpy()                 # exercises BaseTensor _keys plumbing (.cpu()/.numpy())
    assert rc.depth is not None


def test_results_depth_none_is_none():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    r = Results(orig_img=img, path="x.jpg", names={}, depth=None)
    assert r.depth is None


def test_results_depth_only_summary_empty_and_len():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"}, depth=depth)
    assert r.summary() == []          # depth-only Results has no per-instance summary
    assert len(r) == 1                # __len__ returns the depth map count


def test_results_update_depth():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"})
    r.update(depth=np.ones((8, 8), dtype=np.float32))
    from ultralytics.engine.results import DepthMap
    assert isinstance(r.depth, DepthMap)
