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


def test_depth_predictor_postprocess_sets_depthmap():
    import torch
    from ultralytics.engine.results import DepthMap
    from ultralytics.models.yolo.depth.predict import DepthPredictor

    p = DepthPredictor.__new__(DepthPredictor)   # bypass __init__
    p.batch = None

    class _M:  # minimal stand-in for self.model
        names = {0: "depth"}

    p.model = _M()
    img = torch.zeros(1, 3, 32, 32)
    orig = np.zeros((40, 48, 3), dtype=np.uint8)
    preds = torch.rand(1, 1, 32, 32)
    res = p.postprocess(preds, img, [orig])
    assert isinstance(res[0].depth, DepthMap)
    assert res[0].depth.data.shape == (40, 48)   # resized to original image size
