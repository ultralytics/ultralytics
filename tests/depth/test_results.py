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
    assert rc.depth.data.shape == (20, 24)  # shape survives the .cpu().numpy() chain


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


def test_annotator_depth_map_runs():
    from ultralytics.utils.plotting import Annotator

    ann = Annotator(np.zeros((32, 32, 3), dtype=np.uint8))
    ann.depth_map(np.random.rand(32, 32).astype(np.float32))
    out = ann.result()
    assert out.shape == (32, 32, 3)


def test_results_plot_with_depth_runs():
    from ultralytics.engine.results import Results

    img = np.zeros((24, 24, 3), dtype=np.uint8)
    depth = np.random.rand(24, 24).astype(np.float32)
    r = Results(orig_img=img, path="x.jpg", names={0: "depth"}, depth=depth)
    out = r.plot()                      # must not raise; returns an annotated image (masks=True by default)
    assert out.shape[:2] == (24, 48)    # RGB + colorized depth placed side-by-side (width doubled)


def test_annotator_depth_map_all_zero():
    from ultralytics.utils.plotting import Annotator

    ann = Annotator(np.zeros((16, 16, 3), dtype=np.uint8))
    ann.depth_map(np.zeros((16, 16), dtype=np.float32))   # no valid pixels → must not divide-by-zero
    out = ann.result()
    assert out.shape == (16, 16, 3)
