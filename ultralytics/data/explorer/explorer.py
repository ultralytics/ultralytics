from pathlib import Path
from typing import List

import cv2
import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import logger
from ultralytics.utils.checks import check_requirements

from .utils import sanitize_batch

check_requirements('lancedb')
import lancedb


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
        self.connection = lancedb.connect(uri)
        self.table_name = Path(data).name
        self.model = YOLO(model)
        self.data = data  # None
        self.choice_set = None

        self.table = None

    def create_embeddings_table(self, force=False, split='train', verbose=False):
        if (self.table is not None and not force):
            logger.info('Table already exists. Reusing it. Pass force=True to overwrite it.')
            return
        if self.table_name in self.connection.table_names():
            logger.info(f'Table {self.table_name} already exists. Reusing it. Pass force=True to overwrite it.')
            self.table = self.connection.open_table(self.table_name)
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

        dataset = ExplorerDataset(img_path=choice_set, data=data_info, augment=False, cache=False)

        # Create the table schema
        schema = pa.schema([
            pa.field('im_file', pa.string()),
            pa.field('labels', pa.list_(pa.string())),
            pa.field('bboxes', pa.list_(pa.list_(pa.float32()))),
            pa.field('cls', pa.list_(pa.int32())),
            pa.field('vector', pa.list_(pa.float32(),
                                        self.model.embed(dataset[0]['im_file'])[0].shape[0])), ])
        table = self.connection.create_table(self.table_name, schema=schema, mode='overwrite')
        table.add(
            self._yeild_batches(dataset,
                                data_info,
                                self.model,
                                exclude_keys=['img', 'ratio_pad', 'resized_shape', 'ori_shape', 'batch_idx']))

        self.table = table

    @staticmethod
    def _yeild_batches(dataset, data_info, model, exclude_keys: List):
        # Implement Batching
        for i in tqdm(range(len(dataset))):
            batch = dataset[i]
            for k in exclude_keys:
                batch.pop(k, None)
            batch = sanitize_batch(batch, data_info)
            batch['vector'] = model.embed(batch['im_file'], verbose=False)[0].detach().tolist()
            yield [batch]

    def query(self, img, limit=25):
        """
        Query the table for similar images. Accepts a single image or a list of images.

        Args:
            img (str or list): Path to the image or a list of paths to the images.
            limit (int): Number of results to return.

        Returns:
            An arrow table containing the results.
        """
        if self.table is None:
            raise ValueError('Table is not created. Please create the table first.')
        if isinstance(img, str):
            img = [img]
        elif isinstance(img, list):
            pass
        else:
            raise ValueError(f'img must be a string or a list of strings. Got {type(img)}')
        embeds = self.model.embed(img)
        # Get avg if multiple images are passed (len > 1)
        embeds = torch.mean(torch.stack(embeds))

        query = self.table.query(embeds).limit(limit).to_arrow()
        return query

    def sql_query(self, query):
        """
        Run a SQL-Like query on the table. Utilizes LanceDB predicate pushdown.

        Args:
            query (str): SQL query to run.

        Returns:
            An arrow table containing the results.
        """
        if self.table is None:
            raise ValueError('Table is not created. Please create the table first.')

        return self.table.to_lance.to_table(filter=query).to_arrow()
