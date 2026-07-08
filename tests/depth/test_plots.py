"""Unit tests for depth panel plotting (val_batch grids and calibrated variants)."""

import cv2
import torch

from ultralytics.models.yolo.depth.val import DepthValidator, plot_depth_panels


def test_plot_depth_panels_writes_grid(tmp_path):
    """Grid = one row per image, columns RGB | GT | one per preds entry, panel size = img size."""
    imgs = torch.rand(2, 3, 32, 32)
    gt = torch.rand(2, 32, 32) * 5 + 0.5
    preds = [torch.rand(2, 1, 32, 32) * 5 + 0.5, torch.rand(2, 32, 32) * 5 + 0.5]  # both shapes accepted
    fname = tmp_path / "panels.jpg"
    plot_depth_panels(imgs, gt, preds, fname)
    img = cv2.imread(str(fname))
    assert img is not None
    assert img.shape == (2 * 32, 4 * 32, 3)  # 2 rows, RGB|GT|pred|pred


def test_plot_depth_panels_titles_add_header_strip(tmp_path):
    """Passing titles prepends a 24 px labeled header strip."""
    imgs = torch.rand(1, 3, 32, 32)
    gt = torch.rand(1, 32, 32) * 5 + 0.5
    fname = tmp_path / "panels.jpg"
    plot_depth_panels(imgs, gt, [torch.rand(1, 1, 32, 32) * 5], fname, titles=["RGB", "GT", "pred"])
    img = cv2.imread(str(fname))
    assert img.shape == (24 + 32, 3 * 32, 3)


def test_plot_depth_panels_respects_max_images(tmp_path):
    """Rows are capped at max_images."""
    imgs = torch.rand(6, 3, 32, 32)
    gt = torch.rand(6, 32, 32) * 5 + 0.5
    fname = tmp_path / "panels.jpg"
    plot_depth_panels(imgs, gt, [torch.rand(6, 1, 32, 32) * 5], fname, max_images=4)
    img = cv2.imread(str(fname))
    assert img.shape == (4 * 32, 3 * 32, 3)


def test_plot_predictions_layout_unchanged(tmp_path):
    """The validator wrapper still writes 3-column val_batch{ni}.jpg with no header strip."""
    v = DepthValidator.__new__(DepthValidator)  # skip __init__ (needs full args); wrapper only uses save_dir
    v.save_dir = tmp_path
    batch = {"img": torch.rand(2, 3, 32, 32), "depth": torch.rand(2, 32, 32) * 5 + 0.5}
    preds = {"depth": torch.rand(2, 1, 32, 32) * 5 + 0.5}
    v.plot_predictions(batch, preds, ni=0)
    img = cv2.imread(str(tmp_path / "val_batch0.jpg"))
    assert img.shape == (2 * 32, 3 * 32, 3)


def test_calibrate_checkpoint_writes_calibrated_plots(tmp_path):
    """calibrate_checkpoint(plot_dir=...) writes 4-column val_batch{ni}_calibrated.jpg panels."""
    from ultralytics.models.yolo.depth.calibrate import calibrate_checkpoint
    from ultralytics.nn.tasks import DepthModel

    torch.manual_seed(0)
    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [
        {"img": (torch.rand(2, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.rand(2, 64, 64) * 5 + 0.5}
        for _ in range(4)
    ]
    path = tmp_path / "ckpt.pt"
    torch.save({"model": model}, path)
    calibrate_checkpoint(path, batches, device="cpu", plot_dir=tmp_path)
    for ni in range(3):  # max_batches=3 even though 4 batches are available
        img = cv2.imread(str(tmp_path / f"val_batch{ni}_calibrated.jpg"))
        assert img is not None, f"val_batch{ni}_calibrated.jpg missing"
        assert img.shape == (24 + 2 * 64, 4 * 64, 3)  # header strip + 2 rows, RGB|GT|raw|calibrated
    assert not (tmp_path / "val_batch3_calibrated.jpg").exists()


def test_depth_head_found_through_autobackend_wrapper():
    """_depth_head unwraps an AutoBackend-like wrapper (wrapper.model = DepthModel)."""
    from types import SimpleNamespace

    from ultralytics.models.yolo.depth.calibrate import _depth_head
    from ultralytics.nn.tasks import DepthModel

    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    assert _depth_head(SimpleNamespace(model=model)) is model.model[-1]


def test_init_metrics_captures_baked_calibration():
    """init_metrics records the head's baked (cal_a, cal_b) for the calibrated val plot."""
    from types import SimpleNamespace

    from ultralytics.nn.tasks import DepthModel

    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    model.model[-1].cal_b.fill_(0.5)
    v = DepthValidator.__new__(DepthValidator)
    v.init_metrics(SimpleNamespace(model=model))  # AutoBackend-like wrapper, as in standalone val
    assert v._cal_ab == (1.0, 0.5)


def test_standalone_val_writes_baked_calibration_plot(tmp_path):
    """Standalone val (training=False) also writes the 4-column baked-calibration comparison."""
    v = DepthValidator.__new__(DepthValidator)
    v.save_dir = tmp_path
    v.training = False
    v._cal_ab = (1.0, 0.5)
    batch = {"img": torch.rand(2, 3, 32, 32), "depth": torch.rand(2, 32, 32) * 5 + 0.5}
    preds = {"depth": torch.rand(2, 1, 32, 32) * 5 + 0.5}
    v.plot_predictions(batch, preds, ni=0)
    assert cv2.imread(str(tmp_path / "val_batch0.jpg")).shape == (2 * 32, 3 * 32, 3)
    img = cv2.imread(str(tmp_path / "val_batch0_calibrated.jpg"))
    assert img is not None
    assert img.shape == (24 + 2 * 32, 4 * 32, 3)  # header strip + RGB|GT|raw|calibrated


def test_training_val_skips_baked_calibration_plot(tmp_path):
    """During training-epoch validation the comparison is not written (final_eval owns it)."""
    v = DepthValidator.__new__(DepthValidator)
    v.save_dir = tmp_path
    v.training = True
    v._cal_ab = (1.0, 0.5)
    batch = {"img": torch.rand(1, 3, 32, 32), "depth": torch.rand(1, 32, 32) * 5 + 0.5}
    v.plot_predictions(batch, {"depth": torch.rand(1, 1, 32, 32) * 5 + 0.5}, ni=0)
    assert (tmp_path / "val_batch0.jpg").exists()
    assert not (tmp_path / "val_batch0_calibrated.jpg").exists()


class _StatefulLoader:
    """Mimics InfiniteDataLoader: one persistent iterator, so partial iteration leaves it mid-epoch."""

    def __init__(self, batches):
        self.batches = batches
        self.order = []  # batch indices in the order they were yielded
        self._pos = 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for _ in range(len(self.batches)):
            i = self._pos % len(self.batches)
            self._pos += 1
            self.order.append(i)
            yield self.batches[i]

    def reset(self):
        self._pos = 0


def _stateful_loader_and_model(n_batches: int = 5):
    torch.manual_seed(0)
    from ultralytics.nn.tasks import DepthModel

    model = DepthModel("yolo26n-depth.yaml", verbose=False)
    batches = [
        {"img": (torch.rand(1, 3, 64, 64) * 255).to(torch.uint8), "depth": torch.rand(1, 64, 64) * 5 + 0.5}
        for _ in range(n_batches)
    ]
    return _StatefulLoader(batches), model


def test_calibrated_plots_start_at_first_batch(tmp_path):
    """The plot pass rewinds a stateful loader so panels show batches 0..2, matching val_batch{ni}.jpg.

    The trainer's InfiniteDataLoader keeps one persistent iterator across for-loops; the calibration
    fit breaks mid-epoch, so without a rewind the plots would show arbitrary mid-dataset batches.
    """
    from ultralytics.models.yolo.depth.calibrate import _plot_calibrated_batches

    loader, model = _stateful_loader_and_model()
    for _ in zip(range(2), loader):  # leave the persistent iterator mid-epoch, like the fit pass does
        pass
    _plot_calibrated_batches(model, loader, "cpu", 1.0, 0.0, "identity", tmp_path)
    assert loader.order[-3:] == [0, 1, 2]


def test_calibration_fit_starts_at_first_batch():
    """The fit pass rewinds a stateful loader so "first max_images" means the leading images."""
    from ultralytics.models.yolo.depth.calibrate import _collect_logpairs

    loader, model = _stateful_loader_and_model()
    for _ in zip(range(2), loader):  # leave the persistent iterator mid-epoch
        pass
    _collect_logpairs(model, loader, "cpu", max_images=3)
    assert loader.order[-3:] == [0, 1, 2]


def test_final_eval_plots_only_representative_checkpoint(tmp_path, monkeypatch):
    """final_eval passes plot_dir for best.pt only; last.pt is calibrated without plotting."""
    from types import SimpleNamespace

    import ultralytics.models.yolo.depth.calibrate as calibrate
    from ultralytics.models import yolo
    from ultralytics.models.yolo.depth.train import DepthTrainer

    calls = []
    monkeypatch.setattr(yolo.detect.DetectionTrainer, "final_eval", lambda self: None)  # skip real eval
    monkeypatch.setattr(
        calibrate,
        "calibrate_checkpoint",
        lambda ckpt, dl, dev, dist_power=0.0, plot_dir=None: calls.append((ckpt.name, plot_dir)),
    )
    t = DepthTrainer.__new__(DepthTrainer)  # skip __init__ (needs data/model); final_eval uses only these attrs
    t.args = SimpleNamespace(auto_calibrate=True, plots=True, cal_dist_pw=0.0)
    t.best, t.last = tmp_path / "best.pt", tmp_path / "last.pt"
    t.best.touch()
    t.last.touch()
    t.save_dir, t.test_loader, t.device = tmp_path, [], "cpu"
    t.final_eval()
    assert calls == [("best.pt", tmp_path), ("last.pt", None)]

    calls.clear()
    t.args.plots = False  # plots disabled -> calibrate both, plot neither
    t.final_eval()
    assert calls == [("best.pt", None), ("last.pt", None)]

    calls.clear()
    t.args.plots = True
    t.best.unlink()  # best never saved -> last.pt becomes the representative checkpoint
    t.final_eval()
    assert calls == [("last.pt", tmp_path)]
