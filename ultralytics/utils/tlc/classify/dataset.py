# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from multiprocessing.pool import ThreadPool
from pathlib import Path
from itertools import repeat

from ultralytics.data.dataset import ClassificationDataset
from ultralytics.data.utils import verify_image
from ultralytics.utils import LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.tlc.engine.dataset import TLCDatasetMixin


class TLCClassificationDataset(TLCDatasetMixin, ClassificationDataset):
    """
    Initialize 3LC classification dataset for use in YOLO classification.

    Args:
        table (tlc.Table): The 3LC table with classification data. Needs columns 'image' and 'label'.
        args (Namespace): See parent.
        augment (bool): See parent.
        prefix (str): See parent.
    
    """
    def __init__(
            self,
            table,
            args,
            augment=False,
            prefix="",
            image_column_name=tlc.IMAGE,
            label_column_name=tlc.LABEL,
        ):
        # Populate self.samples with image paths and labels
        # Each is a tuple of (image_path, label)
        assert isinstance(table, tlc.Table)
        self.table = table
        self.root = table.url

        self.verify_schema(image_column_name, label_column_name)

        self.samples = []
        self.example_ids = []

        for example_id, row in enumerate(self.table.table_rows):
            self.example_ids.append(example_id)
            image_path = Path(tlc.Url(row[image_column_name]).to_absolute().to_str())
            self.samples.append((image_path, row[label_column_name]))

        # Initialize attributes (calls self.verify_images())
        self._init_attributes(args, augment, prefix)

        # Call mixin
        self._post_init()

    def verify_schema(self, image_column_name, label_column_name):
        """ Verify that the provided Table has the desired entries """

        # Check for data in columns
        assert len(self.table) > 0, f"Table {self.root.to_str()} has no rows."
        first_row = self.table.table_rows[0]
        assert isinstance(first_row[image_column_name], str), f"First value in image column '{image_column_name}' in table {self.root.to_str()} is not a string."
        assert isinstance(first_row[label_column_name], int), f"First value in label column '{label_column_name}' in table {self.root.to_str()} is not an integer."

    def verify_images(self):
        """ Verify all images in the dataset."""

        # Skip verification if the dataset has already been scanned
        if self._is_scanned():
            return self.samples
 
        desc = f"{self.prefix}Scanning images in {self.table.url.to_str()}..."
        # Run scan if the marker does not exist
        nf, nc, msgs, samples, example_ids = 0, 0, [], [], []
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(enumerate(results), desc=desc, total=len(self.samples))
            for i, (sample, nf_f, nc_f, msg) in pbar:
                if nf_f:
                    example_ids.append(self.example_ids[i])
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f"{desc} {nf} images, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))

        # If no problems are found, create the marker
        if nc == 0:
            self._write_scanned_marker()

        self._example_ids = example_ids
        return samples
