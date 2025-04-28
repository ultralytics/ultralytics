# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os

from ultralytics import ASSETS
from ultralytics.utils import LOGGER

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP conflict on some systems
import torch

from ultralytics.utils.checks import check_requirements
check_requirements(["open-clip-torch", "faiss-cpu"])

import faiss
import op as op

import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, render_template, request


class VisualAISearch:
    """
    A semantic image search system using OpenAI CLIP for feature
    extraction and FAISS for approximate nearest neighbor search.
    """

    def __init__(self, data='images', device=None):
        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.model_name = 'ViT-B-32-quickgelu'
        self.data_dir = Path(data)
        if not os.path.exists(self.data_dir):
            LOGGER.warn(f"{self.data_dir} don't exist, downloading `coco128` images from {ASSETS}")

        self.device = select_device(device)

        # Load CLIP model and preprocessor
        self.clip_model, _, self.preprocess = op.create_model_and_transforms(
            self.model_name, pretrained='openai')
        self.clip_model = self.clip_model.to(self.device).eval()
        self.tokenizer = op.get_tokenizer(self.model_name)

        # FAISS setup
        self.index = None
        self.image_paths = []

        self._load_or_build_index()

    def _extract_image_feature(self, path):
        """Extract CLIP embedding for a single image."""
        image = Image.open(path).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_image(tensor).cpu().numpy()

    def _extract_text_feature(self, text):
        """Convert search text to a CLIP text embedding."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_text(tokens).cpu().numpy()

    def _load_or_build_index(self):
        """Load an existing FAISS index or create one from the image directory."""
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            print("[INFO] Loading FAISS index...")
            self.index = faiss.read_index(self.faiss_index)
            self.image_paths = np.load(self.data_path_npy)
            return

        print("[INFO] Building FAISS index...")
        vectors = []
        for file in self.image_dir.iterdir():
            if file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            try:
                vec = self._extract_image_feature(file)
                vectors.append(vec)
                self.image_paths.append(file.name)
            except Exception as e:
                print(f"[WARN] Skipped {file.name}: {e}")

        if not vectors:
            raise RuntimeError("No image embeddings were generated.")

        vectors = np.vstack(vectors).astype("float32")
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        faiss.write_index(self.index, self.faiss_index)
        np.save(self.data_path_npy, np.array(self.image_paths))
        print(f"[INFO] Indexed {len(self.image_paths)} images.")

    def search(self, query, k=30, similarity_thresh=0.1):
        """
        Perform CLIP + FAISS image search.

        Args:
            query (str): User search prompt
            k (int): Number of top results
            similarity_thresh (float): Minimum similarity score to include

        Returns:
            List of image filenames sorted by semantic relevance
        """
        text_feat = self._extract_text_feature(query).astype("float32")
        faiss.normalize_L2(text_feat)

        D, I = self.index.search(text_feat, k)
        results = []
        for idx, i in enumerate(I[0]):
            sim = float(D[0][idx])
            if sim >= similarity_thresh:
                results.append((self.image_paths[i], sim))

        results.sort(key=lambda x: x[1], reverse=True)

        print("\n[INFO] Ranked results:")
        for name, score in results:
            print(f"  - {name} | Similarity: {score:.4f}")

        return [r[0] for r in results]


class SearchApp:
    def __init__(self, image_dir, device=None):
        self.searcher = ClipFaissSearcher(data=image_dir, device=device)
        self.app = Flask(__name__, template_folder="templates", static_folder=image_dir)
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self):
        results = []
        if request.method == "POST":
            query = request.form.get("query", "").strip()
            results = self.searcher.search(query)
        return render_template("index.html", results=results)

    def run(self):
        self.app.run(debug=True)


if __name__ == "__main__":
    app.run(debug=True)
