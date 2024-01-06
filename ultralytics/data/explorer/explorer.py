from io import BytesIO
from pathlib import Path
from typing import List

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
from ultralytics.utils import LOGGER, checks

from .utils import get_sim_index_schema, get_table_schema, plot_similar_images, sanitize_batch


class ExplorerDataset(YOLODataset):

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, **kwargs)

    # NOTE: Load the image directly without any resize operations.
    def load_image(self, i):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def build_transforms(self, hyp=None):
        transforms = Format(
            bbox_format='xyxy',
            normalize=False,
            return_mask=self.use_segments,
            return_keypoint=self.use_keypoints,
            batch_idx=True,
            mask_ratio=hyp.mask_ratio,
            mask_overlap=hyp.overlap_mask,
        )
        return transforms


class Explorer:

    def __init__(self, data='coco128.yaml', model='yolov8n.pt', uri='~/ultralytics/explorer') -> None:
        checks.check_requirements(['lancedb', 'duckdb'])
        import lancedb

        self.connection = lancedb.connect(uri)
        self.table_name = Path(data).name.lower() + '_' + model.lower()
        self.sim_idx_base_name = f'{self.table_name}_sim_idx'.lower(
        )  # Use this name and append thres and top_k to reuse the table
        self.model = YOLO(model)
        self.data = data  # None
        self.choice_set = None

        self.table = None
        self.progress = 0

    def create_embeddings_table(self, force=False, split='train'):
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
            LOGGER.info('Table already exists. Reusing it. Pass force=True to overwrite it.')
            return
        if self.table_name in self.connection.table_names() and not force:
            LOGGER.info(f'Table {self.table_name} already exists. Reusing it. Pass force=True to overwrite it.')
            self.table = self.connection.open_table(self.table_name)
            self.progress = 1
            return
        if self.data is None:
            raise ValueError('Data must be provided to create embeddings table')

        data_info = check_det_dataset(self.data)
        if split not in data_info:
            raise ValueError(
                f'Split {split} is not found in the dataset. Available keys in the dataset are {list(data_info.keys())}'
            )

        choice_set = data_info[split]
        choice_set = choice_set if isinstance(choice_set, list) else [choice_set]
        self.choice_set = choice_set
        dataset = ExplorerDataset(img_path=choice_set, data=data_info, augment=False, cache=False, task=self.model.task)

        # Create the table schema
        batch = dataset[0]
        vector_size = self.model.embed(batch['im_file'], verbose=False)[0].shape[0]
        Schema = get_table_schema(vector_size)
        table = self.connection.create_table(self.table_name, schema=Schema, mode='overwrite')
        table.add(
            self._yield_batches(dataset,
                                data_info,
                                self.model,
                                exclude_keys=['img', 'ratio_pad', 'resized_shape', 'ori_shape', 'batch_idx']))

        self.table = table

    def _yield_batches(self, dataset, data_info, model, exclude_keys: List):
        # Implement Batching
        for i in tqdm(range(len(dataset))):
            self.progress = float(i + 1) / len(dataset)
            batch = dataset[i]
            for k in exclude_keys:
                batch.pop(k, None)
            batch = sanitize_batch(batch, data_info)
            batch['vector'] = model.embed(batch['im_file'], verbose=False)[0].detach().tolist()
            yield [batch]

    def query(self, imgs=None, limit=25):
        """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            imgs (str or list): Path to the image or a list of paths to the images.
            limit (int): Number of results to return.

        Returns:
            An arrow table containing the results. Supports converting to:
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
            raise ValueError('Table is not created. Please create the table first.')
        if isinstance(imgs, str):
            imgs = [imgs]
        elif isinstance(imgs, list):
            pass
        else:
            raise ValueError(f'img must be a string or a list of strings. Got {type(imgs)}')
        embeds = self.model.embed(imgs)
        # Get avg if multiple images are passed (len > 1)
        embeds = torch.mean(torch.stack(embeds), 0).cpu().numpy() if len(embeds) > 1 else embeds[0].cpu().numpy()
        query = self.table.search(embeds).limit(limit).to_arrow()
        return query

    def sql_query(self, query, return_type='pandas'):
        """
        Run a SQL-Like query on the table. Utilizes LanceDB predicate pushdown.

        Args:
            query (str): SQL query to run.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            An arrow table containing the results.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = 'SELECT * FROM table WHERE labels LIKE "%person%"'
            result = exp.sql_query(query)
            ```
        """
        import duckdb

        if self.table is None:
            raise ValueError('Table is not created. Please create the table first.')

        # Note: using filter pushdown would be a better long term solution. Temporarily using duckdb for this.
        table = self.table.to_arrow()  # noqa
        if not query.startswith('SELECT') and not query.startswith('WHERE'):
            raise ValueError(
                'Query must start with SELECT or WHERE. You can either pass the entire query or just the WHERE clause.')
        if query.startswith('WHERE'):
            query = f"SELECT * FROM 'table' {query}"
        LOGGER.info(f'Running query: {query}')

        rs = duckdb.sql(query)
        if return_type == 'pandas':
            return rs.df()
        elif return_type == 'arrow':
            return rs.arrow()

    def plot_sql_query(self, query, labels=True):
        """
        Plot the results of a SQL-Like query on the table.
        Args:
            query (str): SQL query to run.
            labels (bool): Whether to plot the labels or not.

        Returns:
            PIL Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            query = 'SELECT * FROM table WHERE labels LIKE "%person%"'
            result = exp.plot_sql_query(query)
            ```
        """
        result = self.sql_query(query, return_type='arrow')
        img = plot_similar_images(result, plot_labels=labels)
        img = Image.fromarray(img)
        return img

    def get_similar(self, img=None, idx=None, limit=25, return_type='pandas'):
        """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            limit (int): Number of results to return. Defaults to 25.
            return_type (str): Type of the result to return. Can be either 'pandas' or 'arrow'. Defaults to 'pandas'.

        Returns:
            A table or pandas dataframe containing the results.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.get_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
        img = self._check_imgs_or_idxs(img, idx)
        similar = self.query(img, limit=limit)

        if return_type == 'pandas':
            return similar.to_pandas()
        elif return_type == 'arrow':
            return similar

    def plot_similar(self, img=None, idx=None, limit=25, labels=True):
        """
        Plot the similar images. Accepts images or indexes.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            idx (int or list): Index of the image in the table or a list of indexes.
            labels (bool): Whether to plot the labels or not.
            limit (int): Number of results to return. Defaults to 25.

        Returns:
            PIL Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            similar = exp.plot_similar(img='https://ultralytics.com/images/zidane.jpg')
            ```
        """
        similar = self.get_similar(img, idx, limit, return_type='arrow')
        img = plot_similar_images(similar, plot_labels=labels)
        img = Image.fromarray(img)
        return img

    def similarity_index(self, max_dist=0.2, top_k=None, force=False):
        """
        Calculate the similarity index of all the images in the table. Here, the index will contain the data points that
        are max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of the closest data points to consider when counting. Used to apply limit when running
                            vector search. Defaults to 0.01.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            A pandas dataframe containing the similarity index.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            sim_idx = exp.similarity_index()
            ```
        """
        if self.table is None:
            raise ValueError('Table is not created. Please create the table first.')
        sim_idx_table_name = f'{self.sim_idx_base_name}_thres_{max_dist}_top_{top_k}'.lower()
        if sim_idx_table_name in self.connection.table_names() and not force:
            LOGGER.info('Similarity matrix already exists. Reusing it. Pass force=True to overwrite it.')
            return self.connection.open_table(sim_idx_table_name).to_pandas()

        if top_k and not (1.0 >= top_k >= 0.0):
            raise ValueError(f'top_k must be between 0.0 and 1.0. Got {top_k}')
        if max_dist < 0.0:
            raise ValueError(f'max_dist must be greater than 0. Got {max_dist}')

        top_k = int(top_k * len(self.table)) if top_k else len(self.table)
        top_k = max(top_k, 1)
        features = self.table.to_lance().to_table(columns=['vector', 'im_file']).to_pydict()
        im_files = features['im_file']
        embeddings = features['vector']

        sim_table = self.connection.create_table(sim_idx_table_name, schema=get_sim_index_schema(), mode='overwrite')

        def _yield_sim_idx():
            for i in tqdm(range(len(embeddings))):
                sim_idx = self.table.search(embeddings[i]).limit(top_k).to_pandas().query(f'_distance <= {max_dist}')
                yield [{
                    'idx': i,
                    'im_file': im_files[i],
                    'count': len(sim_idx),
                    'sim_im_files': sim_idx['im_file'].tolist()}]

        sim_table.add(_yield_sim_idx())
        self.sim_index = sim_table

        return sim_table.to_pandas()

    def plot_similarity_index(self, max_dist=0.2, top_k=None, force=False):
        """
        Plot the similarity index of all the images in the table. Here, the index will contain the data points that are
        max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of closest data points to consider when counting. Used to apply limit when
                running vector search. Defaults to 0.01.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            PIL Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            exp.plot_similarity_index()
            ```
        """
        sim_idx = self.similarity_index(max_dist=max_dist, top_k=top_k, force=force)
        sim_count = sim_idx['count'].tolist()
        sim_count = np.array(sim_count)

        indices = np.arange(len(sim_count))

        # Create the bar plot
        plt.bar(indices, sim_count)

        # Customize the plot (optional)
        plt.xlabel('data idx')
        plt.ylabel('Count')
        plt.title('Similarity Count')
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Use Pillow to open the image from the buffer
        image = Image.open(buffer)
        return image

    def _check_imgs_or_idxs(self, img, idx):
        if img is None and idx is None:
            raise ValueError('Either img or idx must be provided.')
        if img is not None and idx is not None:
            raise ValueError('Only one of img or idx must be provided.')
        if idx is not None:
            idx = idx if isinstance(idx, list) else [idx]
            img = self.table.to_lance().take(idx, columns=['im_file']).to_pydict()['im_file']

        img = img if isinstance(img, list) else [img]
        return img

    def visualize(self, result):
        """
        Visualize the results of a query.

        Args:
            result (arrow table): Arrow table containing the results of a query.
        """
        # TODO:
        pass

    def generate_report(self, result):
        """Generate a report of the dataset."""
        pass
