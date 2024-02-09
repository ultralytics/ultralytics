# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.plotting import feature_visualization


class TLCDetectionModel(DetectionModel):
    activations = None
    """ YOLO (You Only Look Once) object detection model with 3LC integration. """

    def _predict_once(self, x: torch.Tensor, profile: bool = False, visualize: bool = False, embed: list | None = None):
        """
        Perform a forward pass through the network.

        :param x: The input tensor to the model.
        :param profile: Print the computation time of each layer if True, defaults to False.
        :param visualize: Save the feature maps of the model if True, defaults to False.
        :param embed: A list of feature vectors/embeddings to return.
        :return: The last output of the model.
        """
        y, dt, embeddings = [], [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if m.type == 'ultralytics.nn.modules.block.SPPF':
                TLCDetectionModel.activations = nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(
                    -1)  # flatten
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if embed and m.i in embed:
                embeddings.append(nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))  # flatten
                if m.i == max(embed):
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x
