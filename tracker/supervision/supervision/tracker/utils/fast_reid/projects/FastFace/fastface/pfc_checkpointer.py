# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
from typing import Any, Dict

import torch

from supervision.tracker.utils.fast_reid.fastreid.engine.hooks import PeriodicCheckpointer
from supervision.tracker.utils.fast_reid.fastreid.utils import comm
from supervision.tracker.utils.fast_reid.fastreid.utils.checkpoint import Checkpointer
from supervision.tracker.utils.fast_reid.fastreid.utils.file_io import PathManager


class PfcPeriodicCheckpointer(PeriodicCheckpointer):

    def step(self, epoch: int, **kwargs: Any):
        rank = comm.get_rank()
        if (epoch + 1) % self.period == 0 and epoch < self.max_epoch - 1:
            self.checkpointer.save(
                f"softmax_weight_{epoch:04d}_rank_{rank:02d}"
            )
        if epoch >= self.max_epoch - 1:
            self.checkpointer.save(f"softmax_weight_{rank:02d}", )


class PfcCheckpointer(Checkpointer):
    def __init__(self, model, save_dir, *, save_to_disk=True, **checkpointables):
        super().__init__(model, save_dir, save_to_disk=save_to_disk, **checkpointables)
        self.rank = comm.get_rank()

    def save(self, name: str, **kwargs: Dict[str, str]):
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        data["model"] = {
            "weight": self.model.weight.data,
            "momentum": self.model.weight_mom,
        }
        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving partial fc weights")
        with PathManager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def _load_model(self, checkpoint: Any):
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)
        self.model.weight.data.copy_(checkpoint_state_dict.pop("weight"))
        self.model.weight_mom.data.copy_(checkpoint_state_dict.pop("momentum"))

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, f"last_weight_{self.rank:02d}")
        return PathManager.exists(save_file)

    def get_checkpoint_file(self):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_weight_{self.rank:02d}")
        try:
            with PathManager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str):
        save_file = os.path.join(self.save_dir, f"last_weight_{self.rank:02d}")
        with PathManager.open(save_file, "w") as f:
            f.write(last_filename_basename)
