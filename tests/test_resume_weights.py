import torch
from types import SimpleNamespace

from ultralytics.engine import trainer as trainer_mod
from ultralytics.engine.trainer import BaseTrainer


def test_setup_model_resume_keeps_resumed_weights(monkeypatch):
    """Regression test for https://github.com/ultralytics/ultralytics/issues/25812.

    When training is resumed, ``BaseTrainer.setup_model`` must keep the weights loaded
    from the resume checkpoint (last.pt) and must NOT overwrite them with the
    initial pretrained weights, even when ``args.pretrained`` is a (string) model
    path. Otherwise the resumed run silently restarts from the initial pretrained
    weights while only the optimizer/epoch state is restored.
    """
    # Two distinct fake weight objects so we can tell which one is actually used.
    class FakeWeights:
        yaml = {}

    resumed_weights = FakeWeights()
    pretrained_weights = FakeWeights()

    def fake_load_checkpoint(path):
        if str(path).endswith("last.pt"):
            return resumed_weights, {"epoch": 0}
        return pretrained_weights, None

    # Patch the name bound in trainer.py's namespace (it does `from ultralytics.nn.tasks
    # import load_checkpoint`, so patching the tasks module alone would not affect it).
    monkeypatch.setattr(trainer_mod, "load_checkpoint", fake_load_checkpoint)

    captured = {}

    def fake_get_model(cfg=None, weights=None, verbose=False):
        captured["weights"] = weights
        return torch.nn.Linear(2, 2)

    ns = SimpleNamespace()
    ns.model = "last.pt"
    ns.resume = True
    ns.args = SimpleNamespace(model="last.pt", resume="last.pt", pretrained="yolov8n.pt")
    ns.get_model = fake_get_model

    ckpt = BaseTrainer.setup_model(ns)

    assert captured["weights"] is resumed_weights, (
        "Resumed checkpoint weights must be used; initial pretrained weights must not override them."
    )
    assert ckpt == {"epoch": 0}
