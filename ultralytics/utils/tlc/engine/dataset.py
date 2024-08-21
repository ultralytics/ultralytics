# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import json
import tlc
import numpy as np

from ultralytics.utils import LOGGER

# Responsible for any generic 3LC dataset handling, such as using sampling weights
# Assume there is an attribute self.table that is a tlc.Table, and self._example_ids
class TLCDatasetMixin:
    def _post_init(self, sampling_weights=False):
        self.display_name = self.table.dataset_name
        # Checks
        if sampling_weights and tlc.SAMPLE_WEIGHT not in self.table.table_rows[0]:
            raise ValueError("Cannot use sampling weights with no sample weight column in the table.")

        assert hasattr(self, "table") and isinstance(self.table, tlc.Table), "TLCDatasetMixin requires an attribute `table` which is a tlc.Table."
        # Assume instance has self._indices (live sampled indices of the dataset)
        # Assume instance has self._example_ids (index -> example_id mapping)

        if not hasattr(self, "_indices"):
            self._indices = np.arange(len(self.example_ids))

        sample_weights = [
            self.table.table_rows[example_id][tlc.SAMPLE_WEIGHT]
            for example_id in self.example_ids
        ]
        self._sample_probabilities = np.array(sample_weights) / np.sum(sample_weights)

    def resample_indices(self):
        # Sample from available indices
        self._indices[:] = np.random.choice(self.example_ids, len(self.example_ids), p=self._sample_probabilities)

    def __getitem__(self, index):
        i = self._indices[index]
        return super().__getitem__(i)
    
    def __len__(self):
        return len(self._indices)

    def _get_enumerated_table_rows(self, exclude_zero_weight):
        if exclude_zero_weight and tlc.SAMPLE_WEIGHT not in self.table.table_rows[0]:
            raise ValueError("Cannot exclude zero weight samples with no sample weight column in the table.")

        if exclude_zero_weight:
            return ((i, row) for i, row in enumerate(self.table.table_rows) if row[tlc.SAMPLE_WEIGHT] > 0)
        else:
            return enumerate(self.table.table_rows)
        
    def _is_scanned(self):
        """ Check if the dataset has been scanned. """
        verified_marker_url = self.table.url / "cache.yolo"
        
        if verified_marker_url.exists():
            # Only skip scan if full scan was done or zero weight sample exclusion is the same as for scan
            content = json.loads(verified_marker_url.read(mode="s"))
            # If zero_excluded is not in the marker, we assume it is True
            if not content.get("zero_excluded", True):
                LOGGER.info(f"{self.prefix}Images in {self.table.url.to_str()} already verified, skipping scan.")
                return True
            elif content.get("zero_excluded", True) and self._exclude_zero_weight:
                LOGGER.info(f"{self.prefix}Images in {self.table.url.to_str()} already verified (excluding zero weight samples), skipping scan.")
                return True
            else:
                LOGGER.info(f"{self.prefix}Images in {self.table.url.to_str()} already verified, but scan was not on all images. Re-scanning.")
        
        return False
    
    def _write_scanned_marker(self):
        verified_marker_url = self.table.url / "cache.yolo"
        possible_subset_str = "All non-zero weight images" if self._exclude_zero_weight else "All images"
        LOGGER.info(f"{self.prefix}{possible_subset_str} in {self.table.url.to_str()} are verified. Writing marker file to {verified_marker_url.to_str()} to skip future verification.")
        verified_marker_url.write(
            content=json.dumps({"verified": True, "zero_excluded": self._exclude_zero_weight}),
            mode="s",
            if_exists="overwrite",
        )