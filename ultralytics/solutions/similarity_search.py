# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import LOGGER, TORCH_VERSION
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.torch_utils import TORCH_2_4, select_device

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Avoid OpenMP conflict on some systems


class VisualAISearch:
    """A semantic image search system using CLIP embeddings and cosine similarity for image retrieval.

    This class leverages OpenAI's CLIP for generating image and text embeddings and NumPy cosine similarity for fast
    similarity-based retrieval. It aligns image and text embeddings in a shared semantic space, enabling users to search
    large collections of images using natural language queries with high accuracy and speed.

    Attributes:
        device (torch.device): Computation device selected from the `device` argument.
        data_dir (Path): Path object for the image directory.
        model: Loaded CLIP model.
        index (np.ndarray): L2-normalized image embeddings used for cosine similarity search.
        image_paths (list[str]): Image file names, aligned with the rows of `index`.

    Methods:
        extract_image_feature: Extract CLIP embedding from an image.
        extract_text_feature: Extract CLIP embedding from text.
        load_or_build_index: Build the embedding index from the images in the data directory.
        search: Perform semantic search for similar images.

    Examples:
        Initialize and search for images
        >>> searcher = VisualAISearch(data="path/to/images", device="cuda")
        >>> results = searcher.search("a cat sitting on a chair", k=10)
    """

    def __init__(self, data: str = "images", device: str | None = "cpu") -> None:
        """Initialize the VisualAISearch class with the embedding index and CLIP model.

        Args:
            data (str): Image directory to index and search, downloaded from Ultralytics assets if missing.
            device (str | None): Device used for CLIP inference (e.g. 'cpu', 'cuda', '0'), or None to auto-select.

        Raises:
            AssertionError: If the installed torch version is older than 2.4.
        """
        assert TORCH_2_4, f"VisualAISearch requires torch>=2.4 (found torch=={TORCH_VERSION})"
        from ultralytics.nn.text_model import build_text_model

        self.data_dir = Path(data)
        self.device = select_device(device)

        if not self.data_dir.exists():
            from ultralytics.utils import ASSETS_URL

            LOGGER.warning(f"{self.data_dir} not found. Downloading images.zip from {ASSETS_URL}/images.zip")
            from ultralytics.utils.downloads import safe_download

            safe_download(url=f"{ASSETS_URL}/images.zip", unzip=True, retry=3)
            self.data_dir = Path("images")

        self.model = build_text_model("clip:ViT-B/32", device=self.device)
        self.load_or_build_index()

    def extract_image_feature(self, path: Path) -> np.ndarray:
        """Extract CLIP image embedding with shape (1, D) from the given image path."""
        return self.model.encode_image(Image.open(path)).detach().cpu().numpy()

    def extract_text_feature(self, text: str) -> np.ndarray:
        """Extract CLIP text embedding with shape (1, D) from the given text query."""
        return self.model.encode_text(self.model.tokenize([text])).detach().cpu().numpy()

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        """L2-normalize each row of `x` so inner products equal cosine similarity.

        Args:
            x (np.ndarray): Feature array of shape (N, D).

        Returns:
            (np.ndarray): Row-wise L2-normalized array with the same shape as the input.
        """
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def load_or_build_index(self) -> None:
        """Build the image embedding index from the image directory.

        Extracts CLIP features from all images in the data directory and L2-normalizes them.

        Raises:
            RuntimeError: If no image embeddings could be generated from the data directory.
        """
        LOGGER.info("Building embeddings from images...")
        vectors = []  # List to store feature vectors of images
        self.image_paths = []

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
        self.index = self._normalize(vectors)  # L2-normalize so inner product equals cosine similarity

        LOGGER.info(f"Indexed {len(self.image_paths)} images.")

    def search(self, query: str, k: int = 30, similarity_thresh: float = 0.1) -> list[str]:
        """Return top-k semantically similar images to the given query.

        Args:
            query (str): Natural language text query to search for.
            k (int, optional): Maximum number of results to return.
            similarity_thresh (float, optional): Minimum similarity threshold for filtering results.

        Returns:
            (list[str]): Image filenames with similarity >= `similarity_thresh`, ranked by descending similarity.

        Examples:
            Search for images matching a query
            >>> searcher = VisualAISearch(data="images")
            >>> results = searcher.search("red car", k=5, similarity_thresh=0.2)
        """
        text_feat = self._normalize(self.extract_text_feature(query).astype("float32"))
        scores = self.index @ text_feat[0]  # cosine similarity (embeddings are L2-normalized)
        top_k = np.argsort(scores)[::-1][: max(k, 0)]
        results = [(self.image_paths[i], float(scores[i])) for i in top_k if scores[i] >= similarity_thresh]
        results.sort(key=lambda x: x[1], reverse=True)

        LOGGER.info("\nRanked Results:")
        for name, score in results:
            LOGGER.info(f"  - {name} | Similarity: {score:.4f}")

        return [r[0] for r in results]

    def __call__(self, query: str) -> list[str]:
        """Search for images matching `query` using the default `search` arguments."""
        return self.search(query)


class SearchApp:
    """A Flask-based web interface for semantic image search with natural language queries.

    This class provides a clean, responsive frontend that enables users to input natural language queries and instantly
    view the most relevant images retrieved from the indexed database.

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

    def __init__(self, data: str = "images", device: str | None = None) -> None:
        """Initialize the SearchApp with VisualAISearch backend.

        Args:
            data (str): Path to directory containing images to index and search.
            device (str | None): Device used for CLIP inference (e.g. 'cpu', 'cuda', '0'), or None to auto-select.
        """
        check_requirements("flask>=3.0.1")
        from flask import Flask, render_template, request

        self.render_template = render_template
        self.request = request
        self.searcher = VisualAISearch(data=data, device=device)
        self.app = Flask(
            __name__,
            template_folder="templates",
            static_folder=self.searcher.data_dir.resolve(),  # Absolute path to serve images
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
