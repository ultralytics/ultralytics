# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import TORCH_2_4, select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP conflict on some systems


class VisualAISearch:
    """
    A semantic image search system that leverages OpenCLIP for generating high-quality image and text embeddings and
    FAISS for fast similarity-based retrieval.

    This class aligns image and text embeddings in a shared semantic space, enabling users to search large collections
    of images using natural language queries with high accuracy and speed.

    Attributes:
        data (str): Directory containing images.
        device (str): Computation device, e.g., 'cpu' or 'cuda'.
        faiss_index (str): Path to the FAISS index file.
        data_path_npy (str): Path to the numpy file storing image paths.
        data_dir (Path): Path object for the data directory.
        model: Loaded CLIP model.
        index: FAISS index for similarity search.
        image_paths (list[str]): List of image file paths.

    Methods:
        extract_image_feature: Extract CLIP embedding from an image.
        extract_text_feature: Extract CLIP embedding from text.
        load_or_build_index: Load existing FAISS index or build new one.
        search: Perform semantic search for similar images.

    Examples:
        Initialize and search for images
        >>> searcher = VisualAISearch(data="path/to/images", device="cuda")
        >>> results = searcher.search("a cat sitting on a chair", k=10)
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the VisualAISearch class with FAISS index and CLIP model."""
        assert TORCH_2_4, f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})"
        from ultralytics.nn.text_model import build_text_model

        check_requirements("faiss-cpu")

        self.faiss = __import__("faiss")
        self.faiss_index = "faiss.index"
        self.data_path_npy = "paths.npy"
        self.data_dir = Path(kwargs.get("data", "images"))
        self.device = select_device(kwargs.get("device", "cpu"))

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            LOGGER.warning(f"{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")

        self.model = build_text_model("clip:ViT-B/32", device=self.device)

        self.index = None
        self.image_paths = []

        self.load_or_build_index()

    def extract_image_feature(self, path: Path) -> np.ndarray:
        """Extract CLIP image embedding from the given image path."""
        return self.model.encode_image(Image.open(path)).cpu().numpy()

    def extract_text_feature(self, text: str) -> np.ndarray:
        """Extract CLIP text embedding from the given text query."""
        return self.model.encode_text(self.model.tokenize([text])).cpu().numpy()

    def load_or_build_index(self) -> None:
        """
        Load existing FAISS index or build a new one from image features.

        Checks if FAISS index and image paths exist on disk. If found, loads them directly. Otherwise, builds a new
        index by extracting features from all images in the data directory, normalizes the features, and saves both the
        index and image paths for future use.
        """
        # Check if the FAISS index and corresponding image paths already exist
        if Path(self.faiss_index).exists() and Path(self.data_path_npy).exists():
            LOGGER.info("Loading existing FAISS index...")
            self.index = self.faiss.read_index(self.faiss_index)  # Load the FAISS index from disk
            self.image_paths = np.load(self.data_path_npy)  # Load the saved image path list
            return  # Exit the function as the index is successfully loaded

        # If the index doesn't exist, start building it from scratch
        LOGGER.info("Building FAISS index from images...")
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
                LOGGER.warning(f"Skipping {file.name}: {e}")

        # If no vectors were successfully created, raise an error
        if not vectors:
            raise RuntimeError("No image embeddings could be generated.")

        vectors = np.vstack(vectors).astype("float32")  # Stack all vectors into a NumPy array and convert to float32
        self.faiss.normalize_L2(vectors)  # Normalize vectors to unit length for cosine similarity

        self.index = self.faiss.IndexFlatIP(vectors.shape[1])  # Create a new FAISS index using inner product
        self.index.add(vectors)  # Add the normalized vectors to the FAISS index
        self.faiss.write_index(self.index, self.faiss_index)  # Save the newly built FAISS index to disk
        np.save(self.data_path_npy, np.array(self.image_paths))  # Save the list of image paths to disk

        LOGGER.info(f"Indexed {len(self.image_paths)} images.")

    def search(self, query: str, k: int = 30, similarity_thresh: float = 0.1) -> list[str]:
        """
        Return top-k semantically similar images to the given query.

        Args:
            query (str): Natural language text query to search for.
            k (int, optional): Maximum number of results to return.
            similarity_thresh (float, optional): Minimum similarity threshold for filtering results.

        Returns:
            (list[str]): List of image filenames ranked by similarity score.

        Examples:
            Search for images matching a query
            >>> searcher = VisualAISearch(data="images")
            >>> results = searcher.search("red car", k=5, similarity_thresh=0.2)
        """
        text_feat = self.extract_text_feature(query).astype("float32")
        self.faiss.normalize_L2(text_feat)

        D, index = self.index.search(text_feat, k)
        results = [
            (self.image_paths[i], float(D[0][idx])) for idx, i in enumerate(index[0]) if D[0][idx] >= similarity_thresh
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.info("\nRanked Results:")
        for name, score in results:
            LOGGER.info(f"  - {name} | Similarity: {score:.4f}")

        return [r[0] for r in results]

    def __call__(self, query: str) -> list[str]:
        """Direct call interface for the search function."""
        return self.search(query)


class SearchApp:
    """
    A Flask-based web interface for semantic image search with natural language queries.

    This class provides a clean, responsive frontend that enables users to input natural language queries and
    instantly view the most relevant images retrieved from the indexed database.

    Attributes:
        render_template: Flask template rendering function.
        request: Flask request object.
        searcher (VisualAISearch): Instance of the VisualAISearch class.
        app (Flask): Flask application instance.

    Methods:
        index: Process user queries and display search results.
        run: Start the Flask web application.

    Examples:
        Start a search application
        >>> app = SearchApp(data="path/to/images", device="cuda")
        >>> app.run(debug=True)
    """

    def __init__(self, data: str = "images", device: str = None) -> None:
        """
        Initialize the SearchApp with VisualAISearch backend.

        Args:
            data (str, optional): Path to directory containing images to index and search.
            device (str, optional): Device to run inference on (e.g. 'cpu', 'cuda').
        """
        check_requirements("flask>=3.0.1")
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

    def index(self) -> str:
        """Process user query and display search results in the web interface."""
        results = []
        if self.request.method == "POST":
            query = self.request.form.get("query", "").strip()
            results = self.searcher(query)
        return self.render_template("similarity-search.html", results=results)

    def run(self, debug: bool = False) -> None:
        """Start the Flask web application server."""
        self.app.run(debug=debug)
