# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.tasks import TextClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, TQDM

# OpenAI CLIP 80-template prompt ensemble for ImageNet (Radford et al., 2021)
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]


class TextClassificationTrainer(ClassificationTrainer):
    """Trainer for text-aligned classification pre-training with MobileCLIP2 (https://arxiv.org/abs/2508.20691).

    Extend ClassificationTrainer with three loss modes for text-supervised training: 'contrastive' (CE + CLIP-style
    cosine similarity), 'text_similarity' (CE + KL from text embedding structure), and 'clip_distill' (CE + KL from
    pre-computed MobileCLIP2 image embeddings following dataset reinforcement https://arxiv.org/abs/2407.10886).

    Attributes:
        text_embeddings (torch.Tensor): Pre-computed (nc, embed_dim) text embeddings.
        text_similarity (torch.Tensor): Pre-computed (nc, nc) text similarity matrix.
        teacher_img_embeds (torch.Tensor): Pre-computed (N, embed_dim) MobileCLIP2 image embeddings.
        loss_mode (str): Active loss mode.
        teacher_variant (str): MobileCLIP2 variant for teacher ('s4' or 'l14').
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize TextClassificationTrainer.

        Args:
            cfg (dict[str, Any], optional): Default configuration dictionary.
            overrides (dict[str, Any], optional): Parameter overrides. Supports 'loss_mode' ('contrastive',
                'text_similarity', 'clip_distill'), 'teacher_variant' ('s4', 'l14', or 's4+l14' for multi-teacher), and
                'teacher_temps' (list of per-teacher temperatures for multi-teacher clip_distill).
            _callbacks (dict, optional): Callback functions.
        """
        if overrides is None:
            overrides = {}
        self.loss_mode = overrides.pop("loss_mode", "contrastive")
        self.teacher_variant = overrides.pop("teacher_variant", "s4")
        self.use_clip_classifier = overrides.pop("use_clip_classifier", False)
        self.prompt_ensemble = overrides.pop("prompt_ensemble", False)
        self.teacher_temps = overrides.pop("teacher_temps", None)
        self.teachers = self.teacher_variant.split("+")
        self.teacher_dims = []
        self.text_embeddings_per_teacher = []
        self.text_embeddings = None
        self.text_similarity = None
        self.teacher_img_embeds = None
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        """Return TextClassificationModel configured for text-aligned training.

        Args:
            cfg (Any, optional): Model configuration.
            weights (Any, optional): Pre-trained model weights.
            verbose (bool, optional): Whether to display model information.

        Returns:
            (TextClassificationModel): Model with projection head for text alignment.
        """
        model = TextClassificationModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            loss_mode=self.loss_mode,
            use_clip_classifier=self.use_clip_classifier,
        )
        if weights:
            model.load(weights)
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout
        for p in model.parameters():
            p.requires_grad = True
        return model

    def build_dataset(self, img_path: str, mode: str = "train", batch=None):
        """Build dataset and set up text embeddings for training mode.

        Args:
            img_path (str): Path to dataset images.
            mode (str, optional): Dataset mode ('train', 'val', or 'test').
            batch (Any, optional): Batch information (unused).

        Returns:
            (ClassificationDataset): Dataset for the specified mode.
        """
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train" and self.text_embeddings is None:
            self._setup_text_embeddings(Path(img_path).parent, dataset)
        return dataset

    def _setup_text_embeddings(self, cache_dir, dataset):
        """Pre-compute and cache text embeddings for all class names using MobileCLIP2.

        For multi-teacher (teacher_variant="s4+l14"), generates per-teacher text embeddings and stores them
        separately. Student uses first teacher's embeddings for CE/classification.

        Args:
            cache_dir (Path): Directory to cache text embeddings.
            dataset (ClassificationDataset): Training dataset (passed for teacher pre-compute).
        """
        from ultralytics.nn.text_model import build_text_model, encode_text

        names = list(self.data["names"].values())
        suffix = "_ensemble80" if self.prompt_ensemble else ""

        for teacher in self.teachers:
            variant = teacher.lower().replace("-", "")
            cache_path = cache_dir / f"text_embeddings_mobileclip2_{variant}{suffix}.pt"
            embeds = None

            if cache_path.exists():
                cached = torch.load(cache_path, map_location=self.device)
                if cached.get("names") == names:
                    embeds = cached["embeds"].to(self.device)
                    LOGGER.info(f"Loaded cached text embeddings from {cache_path}")

            if embeds is None:
                text_model = build_text_model(f"mobileclip2:{teacher}", device=self.device)
                if self.prompt_ensemble:
                    LOGGER.info(f"Generating 80-template ensemble text embeddings for {len(names)} classes ({teacher})")
                    class_embeds = []
                    for name in TQDM(names, desc=f"Encoding text ensemble ({teacher})"):
                        texts = [t.format(name) for t in IMAGENET_TEMPLATES]
                        class_embeds.append(encode_text(text_model, texts).mean(dim=0))
                    embeds = torch.stack(class_embeds)
                    embeds /= embeds.norm(p=2, dim=-1, keepdim=True)
                else:
                    LOGGER.info(f"Generating text embeddings for {len(names)} classes ({teacher})")
                    texts = [f"a photo of a {name}" for name in names]
                    embeds = encode_text(text_model, texts)
                torch.save({"names": names, "embeds": embeds.cpu()}, cache_path)
                del text_model

            self.text_embeddings_per_teacher.append(embeds)
            self.teacher_dims.append(embeds.shape[-1])

        self.text_embeddings = self.text_embeddings_per_teacher[0]
        self.text_similarity = self.text_embeddings @ self.text_embeddings.T
        self.model.text_similarity = self.text_similarity.to(self.device)
        self.model._text_embeddings = self.text_embeddings
        self.model.teacher_dims = self.teacher_dims if len(self.teacher_dims) > 1 else None
        self.model.teacher_temps = self.teacher_temps

        if self.loss_mode == "clip_distill":
            self._load_teacher_embeddings(dataset)

    def _load_teacher_embeddings(self, dataset):
        """Load or generate pre-computed MobileCLIP2 image embeddings for all training images.

        For multi-teacher, loads per-teacher caches and concatenates along feature dim.

        Args:
            dataset (ClassificationDataset): Training dataset for pre-computing embeddings.
        """
        per_teacher = []
        for teacher in self.teachers:
            variant = teacher.lower().replace("-", "")
            cache_path = Path(self.args.data) / f"teacher_img_embeds_mobileclip2_{variant}.pt"
            if not cache_path.exists():
                if RANK in {-1, 0}:
                    self._precompute_teacher_embeddings(cache_path, dataset, teacher)
                if RANK >= 0:
                    torch.distributed.barrier()
            embeds = torch.load(cache_path, map_location="cpu")
            LOGGER.info(f"Loaded teacher image embeddings: {embeds.shape} from {cache_path}")
            per_teacher.append(embeds)
        self.teacher_img_embeds = torch.cat(per_teacher, dim=-1) if len(per_teacher) > 1 else per_teacher[0]

    def _precompute_teacher_embeddings(self, cache_path, dataset, teacher_name):
        """Run MobileCLIP2 image encoder on all training images and save embeddings to disk.

        Image preprocessing uses CLIP-standard ImageNet normalization (not Ultralytics identity normalization).

        Args:
            cache_path (Path): Path to save the embeddings tensor.
            dataset (ClassificationDataset): Training dataset to read images from.
            teacher_name (str): Teacher variant name (e.g., 's4', 'l14').
        """
        from PIL import Image

        from ultralytics.nn.image_model import build_image_model

        LOGGER.info(f"Pre-computing MobileCLIP2-{teacher_name} image embeddings (one-time, ~30 min for ImageNet)...")
        teacher = build_image_model(f"mobileclip2:{teacher_name}", device=self.device)
        n = len(dataset)
        # Detect embed dim from a probe forward pass
        imgsz = teacher.image_preprocess.transforms[0].size  # get resize target from preprocessing
        probe = teacher.encode_image(torch.randn(1, 3, imgsz, imgsz).to(self.device))
        embed_dim = probe.shape[-1]
        embeds = torch.zeros(n, embed_dim)
        batch_size = 64
        for i in TQDM(range(0, n, batch_size), desc="Teacher embeddings"):
            end = min(i + batch_size, n)
            batch_pil = [Image.open(dataset.samples[j][0]).convert("RGB") for j in range(i, end)]
            batch_tensors = torch.stack([teacher.image_preprocess(img) for img in batch_pil])
            batch_embeds = teacher.encode_image(batch_tensors.to(self.device))
            embeds[i:end] = batch_embeds.cpu()
        torch.save(embeds, cache_path)
        LOGGER.info(
            f"Saved teacher embeddings ({embeds.shape}, {cache_path.stat().st_size / 1e9:.2f}GB) to {cache_path}"
        )
        del teacher
        torch.cuda.empty_cache()

    def preprocess_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Attach text embeddings and optional teacher embeddings to batch.

        For multi-teacher clip_distill, also attaches concatenated per-teacher text embeddings.

        Args:
            batch (dict[str, torch.Tensor]): Batch with 'img', 'cls', and 'idx' keys.

        Returns:
            (dict[str, torch.Tensor]): Batch with added 'txt_feats' and optionally 'teacher_img_embeds'.
        """
        batch = super().preprocess_batch(batch)
        batch["txt_feats"] = self.text_embeddings.to(device=batch["img"].device, dtype=batch["img"].dtype)
        if self.teacher_img_embeds is not None and "idx" in batch:
            batch["teacher_img_embeds"] = self.teacher_img_embeds[batch["idx"]].to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            if len(self.teachers) > 1:
                batch["txt_feats_teachers"] = torch.cat(
                    [
                        e.to(device=batch["img"].device, dtype=batch["img"].dtype)
                        for e in self.text_embeddings_per_teacher
                    ],
                    dim=-1,
                )
        return batch
