# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Open-vocabulary SKU recognition: detect products, embed each crop, assign a SKU by gallery retrieval.

Pipeline: a single-class product detector finds package instances, each detection crop is embedded by a YOLO
ReID model, and the crop is labeled by a nearest-neighbor vote against a folder-per-SKU gallery of reference
images. Adding a new SKU means dropping a folder of reference images into the gallery, with no detector or
embedding retraining. Retrieval is a pure in-memory NumPy dot product over L2-normalized embeddings, which is
fast and light for a few hundred SKUs (e.g. 400 SKUs x 20 references x 512-d float32 is ~16 MB). Pass ``--source``
for full shelf images (the detector finds each product) or ``--crops`` for pre-cropped products (the detector is
skipped and each crop is identified directly). Both accept a single image or a folder, expanded by Ultralytics.

Example:
    python sku_recognition.py --gallery gallery/ --source shelf.jpg    # detect, then identify each product
    python sku_recognition.py --gallery gallery/ --crops test_crops/   # skip the detector, identify crops directly
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.models.yolo.reid import retrieval
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors


def result_embedding(result) -> np.ndarray:
    """Read the (D,) float32 embedding from a ReID ``Results`` object."""
    import torch

    if result.embeddings is None:
        raise RuntimeError("reid model produced no embedding, is this a reid-task checkpoint?")
    data = result.embeddings.data
    data = data.detach().cpu().numpy() if isinstance(data, torch.Tensor) else np.asarray(data)
    return data.reshape(-1).astype(np.float32)


def embed_fn(reid_model, imgsz, device):
    """Return a callable that embeds image paths or BGR crops into an (N, D) float32 array.

    The callable batches one ``reid_model.predict`` call and reads the embedding from each ``Results.embeddings``. It
    accepts both file paths (gallery) and NumPy BGR crops (detections).
    """

    def _embed(images):
        images = [str(im) if isinstance(im, (str, Path)) else im for im in images]  # paths -> str, crops as-is
        results = reid_model.predict(images, imgsz=imgsz, task="reid", device=device, verbose=False)
        return np.stack([result_embedding(r) for r in results], axis=0)

    return _embed


class SKUGallery:
    """Folder-per-SKU reference gallery with in-memory nearest-neighbor SKU assignment.

    Each immediate subfolder of the gallery root is one SKU identity, holding a handful of reference images. Embeddings
    are extracted once (optionally cached), L2-normalized, and kept in RAM for cosine (dot-product)
    retrieval. Each query crop is labeled by a similarity-weighted vote over its top-k gallery neighbors.

    Attributes:
        labels (np.ndarray): (N,) SKU name per gallery embedding, taken from the parent folder name.
        embs (np.ndarray): (N, D) L2-normalized gallery embeddings.
        classes (list): Sorted unique SKU names present in the gallery.
    """

    def __init__(self, embed, gallery: str | Path, imgsz: int, model_id: str, cache: str | Path | None = None):
        """Scan the gallery, embed every reference image, and cache the L2-normalized embeddings in RAM.

        Args:
            embed (Callable): Function mapping a list of image paths to an (N, D) embedding array.
            gallery (str | Path): Gallery root; each immediate subfolder is one SKU identity.
            imgsz (int): Inference image size (part of the cache signature).
            model_id (str): Model identifier (part of the cache signature).
            cache (str | Path, optional): Path to a ``.pt`` embedding cache to reuse across runs.
        """
        paths, self.embs = retrieval.build_gallery(embed, gallery, cache, model_id, imgsz)
        self.labels = np.array([Path(p).parent.name for p in paths])
        self.classes = sorted(set(self.labels.tolist()))
        LOGGER.info(f"gallery: {len(self.labels)} reference images across {len(self.classes)} SKUs")

    def assign(self, query_embs: np.ndarray, topk: int, sim_thresh: float) -> list[tuple[str, float]]:
        """Assign a SKU name and confidence to each query embedding by a top-k similarity-weighted vote.

        Args:
            query_embs (np.ndarray): (Q, D) L2-normalized query embeddings.
            topk (int): Number of gallery neighbors to retrieve per query.
            sim_thresh (float): Minimum winning confidence, below which the query is labeled ``"unknown"``.

        Returns:
            (list[tuple[str, float]]): One ``(sku_name, confidence)`` per query, in query order.
        """
        idx, scores = retrieval.cosine_topk(query_embs, self.embs, topk)
        out = []
        for neigh_idx, neigh_scores in zip(idx, scores):
            totals, counts = defaultdict(float), defaultdict(int)
            for lb, sc in zip(self.labels[neigh_idx], neigh_scores):
                totals[lb] += float(sc)
                counts[lb] += 1
            best = max(totals, key=totals.get)
            conf = totals[best] / counts[best]  # mean similarity of the winning SKU's neighbors
            out.append((best if conf >= sim_thresh else "unknown", conf))
        return out


def recognize(args) -> None:
    """Identify products against the gallery, from full shelf images (--source) or pre-cropped images (--crops)."""
    reid = YOLO(args.reid, task="reid")
    embed = embed_fn(reid, args.imgsz, args.device)
    gallery = SKUGallery(embed, args.gallery, args.imgsz, model_id=args.reid, cache=args.cache)
    Path(args.out).mkdir(parents=True, exist_ok=True)  # results go here, never back into a --source/--crops folder
    if args.crops:
        identify_crops(args, reid, gallery)
    else:
        identify_shelves(args, embed, gallery)


def identify_shelves(args, embed, gallery) -> None:
    """Detect products in each shelf image, embed every crop, and save an annotated copy per image."""
    detector = YOLO(args.detector)
    for result in detector.predict(args.source, imgsz=args.det_imgsz, conf=args.conf, device=args.device, verbose=False):
        src = Path(result.path)
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        if not len(boxes):
            LOGGER.warning(f"{src.name}: detector found no products")
            continue
        image = result.orig_img  # BGR HWC
        crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in boxes]
        assignments = gallery.assign(retrieval.l2_normalize(embed(crops)), args.topk, args.sim_thresh)
        annotator = Annotator(image.copy())
        for (x1, y1, x2, y2), (sku, conf) in zip(boxes, assignments):
            color = colors(0 if sku == "unknown" else gallery.classes.index(sku) + 1, True)
            annotator.box_label((x1, y1, x2, y2), f"{sku} {conf:.2f}", color=color)
            LOGGER.info(f"{src.name} [{x1},{y1},{x2},{y2}] -> {sku} ({conf:.3f})")
        out_path = Path(args.out) / f"{src.stem}_sku.jpg"
        annotator.save(str(out_path))
        LOGGER.info(f"saved {len(boxes)} labeled detections to {out_path}")


def identify_crops(args, reid, gallery) -> None:
    """Embed each pre-cropped product image and assign it to a gallery SKU, skipping the detector."""
    results = reid.predict(args.crops, imgsz=args.imgsz, task="reid", device=args.device, verbose=False)
    embs = retrieval.l2_normalize(np.stack([result_embedding(r) for r in results]))
    for result, (sku, conf) in zip(results, gallery.assign(embs, args.topk, args.sim_thresh)):
        src = Path(result.path)
        image = result.orig_img  # BGR HWC
        color = colors(0 if sku == "unknown" else gallery.classes.index(sku) + 1, True)
        annotator = Annotator(image.copy())
        annotator.box_label((0, 0, image.shape[1] - 1, image.shape[0] - 1), f"{sku} {conf:.2f}", color=color)
        annotator.save(str(Path(args.out) / f"{src.stem}_sku.jpg"))
        LOGGER.info(f"{src.name} -> {sku} ({conf:.3f})")
    LOGGER.info(f"identified {len(results)} crops")


def parse_args():
    """Parse command-line arguments for the SKU recognition pipeline."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_mutually_exclusive_group(required=True)
    io.add_argument("--source", help="full shelf image or folder, runs the detector then identifies each crop")
    io.add_argument("--crops", help="pre-cropped product image or folder, skips the detector and identifies each directly")
    p.add_argument(
        "--detector",
        default="ul://fatih-enterprise/yolo26-sku-detection/yolo26l-sku-detector-sku-110k",
        help="SKU detector, a local .pt or a platform id/url (only used with --source, auto-downloads, needs ULTRALYTICS_API_KEY)",
    )
    p.add_argument(
        "--reid",
        default="ul://fatih-enterprise/yolo26-reid-sku-feature-extraction/yolo26l-reid-rp2k-pretrain",
        help="YOLO ReID model, a local .pt or a platform id/url (auto-downloads, needs ULTRALYTICS_API_KEY)",
    )
    p.add_argument("--gallery", required=True, help="gallery root, each subfolder is one SKU with reference images")
    p.add_argument("--imgsz", type=int, default=256, help="reid embedding image size")
    p.add_argument("--det-imgsz", type=int, default=640, help="detector image size")
    p.add_argument("--conf", type=float, default=0.25, help="detector confidence threshold")
    p.add_argument("--topk", type=int, default=5, help="gallery neighbors per crop for the vote")
    p.add_argument("--sim-thresh", type=float, default=0.5, help="min confidence to accept a SKU, else 'unknown'")
    p.add_argument("--cache", default=None, help="optional .pt gallery-embedding cache to reuse across runs")
    p.add_argument("--device", default=None, help="inference device, e.g. 0 or cpu")
    p.add_argument("--out", default="sku_out", help="output directory for annotated <name>_sku.jpg results")
    return p.parse_args()


if __name__ == "__main__":
    recognize(parse_args())
