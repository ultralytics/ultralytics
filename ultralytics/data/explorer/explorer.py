# Ultralytics YOLO 🚀, AGPL-3.0 license

from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import LOGGER, USER_CONFIG_DIR, IterableSimpleNamespace, checks

from .utils import get_sim_index_schema, get_table_schema, plot_query_result, prompt_sql_query, sanitize_batch


class ExplorerDataset(YOLODataset):
    """Extends YOLODataset for advanced data exploration and manipulation in model training workflows."""

    def __init__(self, *args, data: dict = None, **kwargs) -> None:
        """Initializes the ExplorerDataset with the provided data arguments, extending the YOLODataset class."""
        super().__init__(*args, data=data, **kwargs)

    def load_image(self, i: int) -> Union[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]], Tuple[None, None, None]]:
        """Loads 1 image from dataset index 'i' without any resize ops."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def build_transforms(self, hyp: IterableSimpleNamespace = None):
        """Creates transforms for dataset images without resizing."""
        return Format(
            bbox_format="xyxy",
            normalize=False,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio,
            mask_overlap=hyp.overlap_mask,
        )


class Explorer:
    """Utility class for image embedding, table creation, and similarity querying using LanceDB and YOLO models."""

    def __init__(
        self,
        data: Union[str, Path] = "coco128.yaml",
        model: str = "yolov8n.pt",
        uri: str = USER_CONFIG_DIR / "explorer",
    ) -> None:
        """Initializes the Explorer class with dataset path, model, and URI for database connection."""
        # Note duckdb==0.10.0 bug https://github.com/ultralytics/ultralytics/pull/8181
        checks.check_requirements(["lancedb>=0.4.3", "duckdb<=0.9.2"])
        import lancedb

        self.connection = lancedb.connect(uri)
        self.table_name = f"{Path(data).name.lower()}_{model.lower()}"
        self.sim_idx_base_name = (
            f"{self.table_name}_sim_idx".lower()
        )  # Use this name and append thres and top_k to reuse the table
        self.model = YOLO(model)
        self.data = data  # None
        self.choice_set = None

        self.table = None
        self.progress = 0

    def create_embeddings_table(self, force: bool = False, split: str = "train") -> None:
        """
        Create LanceDB table containing the embeddings of the images in the dataset. The table will be reused if it
        already exists. Pass force=True to overwrite the existing table.

        Args:
            force (bool): Whether to overwrite the existing table or not. Defaults to False.
            split (str): Split of the dataset to use. Defaults to 'train'.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            ```
        """
        if self.table is not None and not force:
            LOGGER.info("Table already exists. Reusing it. Pass force=True to overwrite it.")
            return
        if self.table_name in self.connection.table_names() and not force:
            LOGGER.info(f"Table {self.table_name} already exists. Reusing it. Pass force=True to overwrite it.")
            self.table = self.connection.open_table(self.table_name)
            self.progress = 1
            return
        if self.data is None:
            raise ValueError("Data must be provided to create embeddings table")

        data_info = check_det_dataset(self.data)
        if split not in data_info:
            raise ValueError(
                f"Split {split} is not found in the dataset. Available keys in the dataset are {list(data_info.keys())}"
            )

        choice_set = data_info[split]
        choice_set = choice_set if isinstance(choice_set, list) else [choice_set]
        self.choice_set = choice_set
        dataset = ExplorerDataset(img_path=choice_set, data=data_info, augment=False, cache=False, task=self.model.task)

        # Create the table schema
        batch = dataset[0]
        vector_size = self.model.embed(batch["im_file"], verbose=False)[0].shape[0]
        table = self.connection.create_table(self.table_name, schema=get_table_schema(vector_size), mode="overwrite")
        table.add(
            self._yield_batches(
                dataset,
                data_info,
                self.model,
                exclude_keys=["img", "ratio_pad", "resized_shape", "ori_shape", "batch_idx"],
            )
        )

        self.table = table

    def _yield_batches(self, dataset: ExplorerDataset, data_info: dict, model: YOLO, exclude_keys: List[str]):
        """Generates batches of data for embedding, excluding specified keys."""
        for i in tqdm(range(len(dataset))):
            self.progress = float(i + 1) / len(dataset)
            batch = dataset[i]
            for k in exclude_keys:
                batch.pop(k, None)
            batch = sanitize_batch(batch, data_info)
            batch["vector"] = model.embed(batch["im_file"], verbose=False)[0].detach().tolist()
            yield [batch]

    def query(
        self, imgs: Union[str, np.ndarray, List[str], List[np.ndarray]] = None, limit: int = 25
    ) -> Any:  # pyarrow.Table
        """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            imgs (str or list): Path to the image or a list of paths to the images.
            limit (int): Number of results to return.

        Returns:
            (pyarrow.Table): An arrow table containing the results. Supports converting to:
                - pandas dataframe: `result.to_pandas()`
                - dict of lists: `result.to_pydict()`

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.query(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
        if self.table is None:
            raise ValueError("Table is not created. Please create the table first.")
        if isinstance(imgs, str):
            imgs = [imgs]
        assert isinstance(imgs, list), f"img must be a string or a list of strings. Got {type(imgs)}"
        embeds = self.model.embed(imgs)
        # Get avg if multiple images are passed (len > 1)
        embeds = torch.mean(torch.stack(embeds), 0).cpu().numpy() if len(embeds) > 1 else embeds[0].cpu().numpy()
        return self.table.search(embeds).limit(limit).to_arrow()

    def sql_query(
        self, query: str, return_type: str = "pandas"
    ) -> Union[Any, None]:  # pandas.DataFrame or pyarrow.Table
        """
        Run a SQL-Like query on the table. Utilizes LanceDB predicate pushdown.

        Args:
            query (str): SQL query to run.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            (pyarrow.Table): An arrow table containing the results.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = "SELECT * FROM 'table' WHERE labels LIKE '%person%'"
            result = exp.sql_query(query)
            ```
        """
        assert return_type in {
            "pandas",
            "arrow",
        }, f"Return type should be either `pandas` or `arrow`, but got {return_type}"
        import duckdb

        if self.table is None:
            raise ValueError("Table is not created. Please create the table first.")

        # Note: using filter pushdown would be a better long term solution. Temporarily using duckdb for this.
        table = self.table.to_arrow()  # noqa NOTE: Don't comment this. This line is used by DuckDB
        if not query.startswith("SELECT") and not query.startswith("WHERE"):
            raise ValueError(
                f"Query must start with SELECT or WHERE. You can either pass the entire query or just the WHERE "
                f"clause. found {query}"
            )
        if query.startswith("WHERE"):
            query = f"SELECT * FROM 'table' {query}"
        LOGGER.info(f"Running query: {query}")

        rs = duckdb.sql(query)
        if return_type == "arrow":
            return rs.arrow()
        elif return_type == "pandas":
            return rs.df()

    def plot_sql_query(self, query: str, labels: bool = True) -> Image.Image:
        """
        Plot the results of a SQL-Like query on the table.
        Args:
            query (str): SQL query to run.
            labels (bool): Whether to plot the labels or not.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = "SELECT * FROM 'table' WHERE labels LIKE '%person%'"
            result = exp.plot_sql_query(query)
            ```
        """
        result = self.sql_query(query, return_type="arrow")
        if len(result) == 0:
            LOGGER.info("No results found.")
            return None
        img = plot_query_result(result, plot_labels=labels)
        return Image.fromarray(img)

    def get_similar(
        self,
        img: Union[str, np.ndarray, List[str], List[np.ndarray]] = None,
        idx: Union[int, List[int]] = None,
        limit: int = 25,
        return_type: str = "pandas",
    ) -> Any:  # pandas.DataFrame or pyarrow.Table
        """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            limit (int): Number of results to return. Defaults to 25.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            (pandas.DataFrame): A dataframe containing the results.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.get_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
        assert return_type in {"pandas", "arrow"}, f"Return type should be `pandas` or `arrow`, but got {return_type}"
        img = self._check_imgs_or_idxs(img, idx)
        similar = self.query(img, limit=limit)

        if return_type == "arrow":
            return similar
        elif return_type == "pandas":
            return similar.to_pandas()

    def plot_similar(
        self,
        img: Union[str, np.ndarray, List[str], List[np.ndarray]] = None,
        idx: Union[int, List[int]] = None,
        limit: int = 25,
        labels: bool = True,
    ) -> Image.Image:
        """
        Plot the similar images. Accepts images or indexes.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            labels (bool): Whether to plot the labels or not.
            limit (int): Number of results to return. Defaults to 25.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.plot_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
        similar = self.get_similar(img, idx, limit, return_type="arrow")
        if len(similar) == 0:
            LOGGER.info("No results found.")
            return None
        img = plot_query_result(similar, plot_labels=labels)
        return Image.fromarray(img)

    def similarity_index(self, max_dist: float = 0.2, top_k: float = None, force: bool = False) -> Any:  # pd.DataFrame
        """
        Calculate the similarity index of all the images in the table. Here, the index will contain the data points that
        are max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of the closest data points to consider when counting. Used to apply limit.
                           vector search. Defaults: None.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            (pandas.DataFrame): A dataframe containing the similarity index. Each row corresponds to an image,
                and columns include indices of similar images and their respective distances.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            sim_idx = exp.similarity_index()
            ```
        """
        if self.table is None:
            raise ValueError("Table is not created. Please create the table first.")
        sim_idx_table_name = f"{self.sim_idx_base_name}_thres_{max_dist}_top_{top_k}".lower()
        if sim_idx_table_name in self.connection.table_names() and not force:
            LOGGER.info("Similarity matrix already exists. Reusing it. Pass force=True to overwrite it.")
            return self.connection.open_table(sim_idx_table_name).to_pandas()

        if top_k and not (1.0 >= top_k >= 0.0):
            raise ValueError(f"top_k must be between 0.0 and 1.0. Got {top_k}")
        if max_dist < 0.0:
            raise ValueError(f"max_dist must be greater than 0. Got {max_dist}")

        top_k = int(top_k * len(self.table)) if top_k else len(self.table)
        top_k = max(top_k, 1)
        features = self.table.to_lance().to_table(columns=["vector", "im_file"]).to_pydict()
        im_files = features["im_file"]
        embeddings = features["vector"]

        sim_table = self.connection.create_table(sim_idx_table_name, schema=get_sim_index_schema(), mode="overwrite")

        def _yield_sim_idx():
            """Generates a dataframe with similarity indices and distances for images."""
            for i in tqdm(range(len(embeddings))):
                sim_idx = self.table.search(embeddings[i]).limit(top_k).to_pandas().query(f"_distance <= {max_dist}")
                yield [
                    {
                        "idx": i,
                        "im_file": im_files[i],
                        "count": len(sim_idx),
                        "sim_im_files": sim_idx["im_file"].tolist(),
                    }
                ]

        sim_table.add(_yield_sim_idx())
        self.sim_index = sim_table
        return sim_table.to_pandas()

    def plot_similarity_index(self, max_dist: float = 0.2, top_k: float = None, force: bool = False) -> Image:
        """
        Plot the similarity index of all the images in the table. Here, the index will contain the data points that are
        max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of closest data points to consider when counting. Used to apply limit when
                running vector search. Defaults to 0.01.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()

            similarity_idx_plot = exp.plot_similarity_index()
            similarity_idx_plot.show() # view image preview
            similarity_idx_plot.save('path/to/save/similarity_index_plot.png') # save contents to file
            ```
        """
        sim_idx = self.similarity_index(max_dist=max_dist, top_k=top_k, force=force)
        sim_count = sim_idx["count"].tolist()
        sim_count = np.array(sim_count)

        indices = np.arange(len(sim_count))

        # Create the bar plot
        plt.bar(indices, sim_count)

        # Customize the plot (optional)
        plt.xlabel("data idx")
        plt.ylabel("Count")
        plt.title("Similarity Count")
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Use Pillow to open the image from the buffer
        return Image.fromarray(np.array(Image.open(buffer)))

    def _check_imgs_or_idxs(
        self, img: Union[str, np.ndarray, List[str], List[np.ndarray], None], idx: Union[None, int, List[int]]
    ) -> List[np.ndarray]:
        """Determines whether to fetch images or indexes based on provided arguments and returns image paths."""
        if img is None and idx is None:
            raise ValueError("Either img or idx must be provided.")
        if img is not None and idx is not None:
            raise ValueError("Only one of img or idx must be provided.")
        if idx is not None:
            idx = idx if isinstance(idx, list) else [idx]
            img = self.table.to_lance().take(idx, columns=["im_file"]).to_pydict()["im_file"]

        return img if isinstance(img, list) else [img]

    def ask_ai(self, query):
        """
        Ask AI a question.

        Args:
            query (str): Question to ask.

        Returns:
            (pandas.DataFrame): A dataframe containing filtered results to the SQL query.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            answer = exp.ask_ai('Show images with 1 person and 2 dogs')
            ```
        """
        result = prompt_sql_query(query)
        try:
            return self.sql_query(result)
        except Exception as e:
            LOGGER.error("AI generated query is not valid. Please try again with a different prompt")
            LOGGER.error(e)
            return None

    def visualize(self, result):
        """
        Visualize the results of a query. TODO.

        Args:
            result (pyarrow.Table): Table containing the results of a query.
        """
        pass

    def generate_report(self, result):
        """
        Generate a report of the dataset.

        TODO
        """
        pass
