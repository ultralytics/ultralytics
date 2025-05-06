# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP conflict on some systems


class VisualAISearch(BaseSolution):
    """
    VisualAISearch leverages OpenCLIP to generate high-quality image and text embeddings, aligning them in a shared
    semantic space. It then uses FAISS to perform fast and scalable similarity-based retrieval, allowing users to search
    large collections of images using natural language queries with high accuracy and speed.

    Attributes:
        data (str): Directory containing images.
        device (str): Computation device, e.g., 'cpu' or 'cuda'.
    """

    def __init__(self, **kwargs):
        """Initializes the VisualAISearch class with the FAISS index file and CLIP model."""
        super().__init__(**kwargs)
        check_requirements(["git+https://github.com/ultralytics/CLIP.git", "faiss-cpu"])
        import clip
        import faiss

        self.faiss = faiss
        self.clip = clip

        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.model_name = "ViT-B/32"
        self.data_dir = Path(self.CFG["data"])
        self.device = select_device(self.CFG["device"])

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            self.LOGGER.warning(f"{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")

        self.model, self.preprocess = clip.load(self.model_name, device=self.device)

        self.index = None
        self.image_paths = []

        self.load_or_build_index()

    def extract_image_feature(self, path):
        """Extract CLIP image embedding."""
        image = Image.open(path)
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.model.encode_image(tensor).cpu().numpy()

    def extract_text_feature(self, text):
        """Extract CLIP text embedding."""
        tokens = self.clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            return self.model.encode_text(tokens).cpu().numpy()

    def load_or_build_index(self):
        """Loads FAISS index or builds a new one from image features."""
        # Check if the FAISS index and corresponding image paths already exist
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            self.LOGGER.info("Loading existing FAISS index...")
            self.index = self.faiss.read_index(self.faiss_index)  # Load the FAISS index from disk
            self.image_paths = np.load(self.data_path_npy)  # Load the saved image path list
            return  # Exit the function as the index is successfully loaded

        # If the index doesn't exist, start building it from scratch
        self.LOGGER.info("Building FAISS index from images...")
        vectors = []  # List to store feature vectors of images

        # Iterate over all image files in the data directory
        for file in self.data_dir.iterdir():
            # Skip files that are not valid image formats
            if file.suffix.lower().lstrip(".") not in IMG_FORMATS:
                continue
            try:
                # Extract feature vector for the image and add to the list
                vectors.append(self.extract_image_feature(file))
                self.image_paths.append(file.name)  # Store the corresponding image name
            except Exception as e:
                self.LOGGER.warning(f"Skipping {file.name}: {e}")

        # If no vectors were successfully created, raise an error
        if not vectors:
            raise RuntimeError("No image embeddings could be generated.")

        vectors = np.vstack(vectors).astype("float32")  # Stack all vectors into a NumPy array and convert to float32
        self.faiss.normalize_L2(vectors)  # Normalize vectors to unit length for cosine similarity

        self.index = self.faiss.IndexFlatIP(vectors.shape[1])  # Create a new FAISS index using inner product
        self.index.add(vectors)  # Add the normalized vectors to the FAISS index
        self.faiss.write_index(self.index, self.faiss_index)  # Save the newly built FAISS index to disk
        np.save(self.data_path_npy, np.array(self.image_paths))  # Save the list of image paths to disk

        self.LOGGER.info(f"Indexed {len(self.image_paths)} images.")

    def search(self, query, k=30, similarity_thresh=0.1):
        """Returns top-k semantically similar images to the given query."""
        text_feat = self.extract_text_feature(query).astype("float32")
        self.faiss.normalize_L2(text_feat)

        D, index = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx])) for idx, i in enumerate(index[0]) if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        self.LOGGER.info("\nRanked Results:")
        for name, score in results:
            self.LOGGER.info(f"  - {name} | Similarity: {score:.4f}")

        return [r[0] for r in results]

    def __call__(self, query):
        """Direct call for search function."""
        return self.search(query)


class SearchApp:
    """
    A Flask-based web interface powers the semantic image search experience, enabling users to input natural language
    queries and instantly view the most relevant images retrieved from the indexed database—all through a clean,
    responsive, and easily customizable frontend.

    Args:
        data (str): Path to images to index and search.
        device (str): Device to run inference on (e.g. 'cpu', 'cuda').
    """

    def __init__(self, data="images", device=None):
        """Initialization of the VisualAISearch class for performing semantic image search."""
        check_requirements("flask")
        from flask import Flask, render_template, request

        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=Path(data).resolve(),  # Absolute path to serve images
            static_url_path="/images",  # URL prefix for images
        )
        self.app.add_url_rule("/", view_func=self.index, methods=["GET", "POST"])

    def index(self):
        """Function to process the user query and display output."""
        results = []
        if self.request.method == "POST":
            query = self.request.form.get("query", "").strip()
            results = self.searcher(query)
        return self.render_template("similarity-search.html", results=results)

    def run(self, debug=False):
        """Runs the Flask web app."""
        self.app.run(debug=debug)
