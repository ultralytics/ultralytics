# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import json
import tlc
import numpy as np

from ultralytics.utils import LOGGER


# Responsible for any generic 3LC dataset handling, such as scanning, caching and adding example ids to each sample
# Assume there is an attribute self.table that is a tlc.Table
class TLCDatasetMixin:

    def _post_init(self):
        self.display_name = self.table.dataset_name

        assert hasattr(self, "table") and isinstance(
            self.table, tlc.Table), "TLCDatasetMixin requires an attribute `table` which is a tlc.Table."
        if not hasattr(self, "example_ids"):
            self.example_ids = np.arange(len(self.table))

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample[tlc.EXAMPLE_ID] = self.example_ids[index]  # Add example id to the sample dict
        return sample

    def __len__(self):
        return len(self.example_ids)

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
            elif content.get("zero_excluded", True):
                LOGGER.info(
                    f"{self.prefix}Images in {self.table.url.to_str()} only verified for zero weight samples, need to rescan."
                )
            else:
                LOGGER.info(
                    f"{self.prefix}Images in {self.table.url.to_str()} already verified, but scan was not on all images. Re-scanning."
                )

        return False

    def _write_scanned_marker(self):
        verified_marker_url = self.table.url / "cache.yolo"
        LOGGER.info(
            f"{self.prefix}Images in {self.table.url.to_str()} are verified. Writing marker file to {verified_marker_url.to_str()} to skip future verification."
        )
        verified_marker_url.write(
            content=json.dumps({"verified": True}),
            mode="s",
            if_exists="overwrite",
        )
