# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP conflict on some systems

from pathlib import Path

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image

from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import select_device

check_requirements(["open-clip-torch", "faiss-cpu"])

import faiss
import open_clip as op


class VisualAISearch:
    """
    VisualAISearch uses OpenCLIP for embedding extraction and FAISS for fast similarity-based image retrieval.

    Attributes:
        data (str): Directory containing images.
        device (str): Computation device, e.g., 'cpu' or 'cuda'.
    """

    def __init__(self, data="images", device=None):
        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.model_name = "ViT-B-32-quickgelu"
        self.data_dir = Path(data)
        self.device = select_device(device)

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            LOGGER.warning(f"{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)

        self.clip_model, _, self.preprocess = op.create_model_and_transforms(self.model_name, pretrained="openai")
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = op.get_tokenizer(self.model_name)

        self.index = None
        self.image_paths = []

        self.load_or_build_index()

    def extract_image_feature(self, path):
        """Extract CLIP image embedding."""
        image = Image.open(path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device, memory_format=torch.channels_last)
        with torch.no_grad():
            features = self.clip_model.encode_image(tensor)
        return features.cpu().numpy()

    def extract_text_feature(self, text):
        """Extract CLIP text embedding."""
        tokens = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
        return features.cpu().numpy()

    def load_or_build_index(self):
        """Loads FAISS index or builds a new one from image features."""

        # Load existing FAISS index and image paths if available
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            LOGGER.info("Loading existing FAISS index...")
            self.index = faiss.read_index(self.faiss_index)
            self.image_paths = np.load(self.data_path_npy).tolist()
            return

        LOGGER.info("Building FAISS index from images...")

        # Filter image files
        image_files = [f for f in self.data_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        if not image_files:
            raise RuntimeError("No valid image files found.")

        vectors = []
        image_paths = []

        for file in image_files:
            try:
                vec = self.extract_image_feature(file)  # Could be batchified instead
                vectors.append(vec)
                image_paths.append(file.name)
            except (OSError, RuntimeError) as e:  # Narrow the exception scope
                LOGGER.warning(f"Skipping {file.name}: {e}")

        if not vectors:
            raise RuntimeError("No image embeddings could be generated.")

        # Stack and normalize vectors
        vectors = np.vstack(vectors).astype("float32", copy=False)
        faiss.normalize_L2(vectors)

        # Build FAISS index
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        # Save index and image paths
        faiss.write_index(self.index, self.faiss_index)
        np.save(self.data_path_npy, np.array(image_paths))

        self.image_paths = image_paths
        LOGGER.info(f"Indexed {len(image_paths)} images.")

    def search(self, query, k=30, similarity_thresh=0.1):
        """Returns top-k semantically similar images to the given query."""
        text_feat = self.extract_text_feature(query).astype("float32")
        faiss.normalize_L2(text_feat)

        D, I = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx])) for idx, i in enumerate(I[0]) if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.info("\n[INFO] Ranked Results:")
        for name, score in results:
            LOGGER.info(f"  - {name} | Similarity: {score:.4f}")

        return [r[0] for r in results]


class SearchApp:
    """
    Flask-based web interface for semantic image search.

    Args:
        image_dir (str): Path to images to index and search.
        device (str): Device to run inference on (e.g. 'cpu', 'cuda').
    """

    def __init__(self, image_dir="images", device=None):
        self.searcher = VisualAISearch(data=image_dir, device=device)
        self.app = Flask(__name__, template_folder="templates", static_folder="images")
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self):
        results = []
        if request.method == "POST":
            query = request.form.get("query", "").strip()
            results = self.searcher.search(query)
        return render_template("index.html", results=results)

    def run(self, debug=False):
        """Runs the Flask web app."""
        self.app.run(debug=debug)
