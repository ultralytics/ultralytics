# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models import yolo
from ultralytics.nn.tasks import WorldModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.data import build_yolomultimodal_dataset
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.checks import check_requirements
import itertools

try:
    import clip
except ImportError:
    check_requirements("git+https://github.com/openai/CLIP.git")
    import clip


def on_pretrain_routine_end(trainer):
    """Callback."""
    if RANK in (-1, 0):
        # NOTE: for evaluation
        names = list(trainer.test_loader.dataset.data["names"].values())
        trainer.model.set_classes(names)
    device = next(trainer.model.parameters()).device
    text_model, _ = clip.load("ViT-B/32", device=device)
    for p in text_model.parameters():
        p.requires_grad_(False)
    trainer.text_model = text_model


class WorldTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a world model.

    Example:
        ```python
        from ultralytics.models.yolo.segment import WorldModel

        args = dict(model='yolov8s-world.pt', data='coco8.yaml', epochs=3)
        trainer = WorldTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationModel initialized with specified config and weights."""
        # NOTE: following the official config, nc hard-coded to 80 for now.
        model = WorldModel(cfg["yaml_file"], ch=3, nc=80, verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        self.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolomultimodal_dataset(
            self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs
        )

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        # NOTE: add text features
        texts = list(itertools.chain(*batch["texts"]))
        text_token = clip.tokenize(texts).to(batch["img"].device)
        txt_feats = self.text_model.encode_text(text_token).to(dtype=batch["img"].dtype)  # torch.float32
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        batch["txt_feats"] = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
        return batch
