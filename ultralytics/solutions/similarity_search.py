# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import logging
import os

from ultralytics.data.utils import IMG_FORMATS

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
        image = Image.open(path)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_image(tensor).cpu().numpy()

    def extract_text_feature(self, text):
        """Extract CLIP text embedding."""
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            return self.clip_model.encode_text(tokens).cpu().numpy()

    def load_or_build_index(self):
        """Loads FAISS index or builds a new one from image features."""
        # Check if the FAISS index and corresponding image paths already exist
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            LOGGER.info("Loading existing FAISS index...")
            self.index = faiss.read_index(self.faiss_index)  # Load the FAISS index from disk
            self.image_paths = np.load(self.data_path_npy)  # Load the saved image path list
            return  # Exit the function as the index is successfully loaded

        # If the index doesn't exist, start building it from scratch
        LOGGER.info("Building FAISS index from images...")
        vectors = []  # List to store feature vectors of images

        # Iterate over all image files in the data directory
        for file in self.data_dir.iterdir():
            # Skip files that are not valid image formats
            if file.suffix.lower() not in IMG_FORMATS:
                continue
            try:
                # Extract feature vector for the image and add to the list
                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)  # Store the corresponding image name
            except Exception as e:
                LOGGER.warning(f"Skipping {file.name}: {e}")

        # If no vectors were successfully created, raise an error
        if not vectors:
            raise RuntimeError("No image embeddings could be generated.")

        vectors = np.vstack(vectors).astype("float32")  # Stack all vectors into a NumPy array and convert to float32
        faiss.normalize_L2(vectors)  # Normalize vectors to unit length for cosine similarity

        self.index = faiss.IndexFlatIP(vectors.shape[1])  # Create a new FAISS index using inner product
        self.index.add(vectors)  # Add the normalized vectors to the FAISS index
        faiss.write_index(self.index, self.faiss_index)  # Save the newly built FAISS index to disk
        np.save(self.data_path_npy, np.array(self.image_paths))  # Save the list of image paths to disk

        LOGGER.info(f"Indexed {len(self.image_paths)} images.")

    def search(self, query, k=30, similarity_thresh=0.1):
        """Returns top-k semantically similar images to the given query."""
        text_feat = self.extract_text_feature(query).astype("float32")
        faiss.normalize_L2(text_feat)

        D, I = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx])) for idx, i in enumerate(I[0]) if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.info("\nRanked Results:")
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
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=image_dir,  # Absolute path to serve images
            static_url_path="/images",  # URL prefix for images
        )
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
