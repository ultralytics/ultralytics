# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc
import torch
import weakref

from ultralytics.data import build_dataloader
from ultralytics.models import yolo
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME, CLASSIFY_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.classify.dataset import TLCClassificationDataset
from ultralytics.utils.tlc.classify.utils import tlc_check_cls_dataset
from ultralytics.utils.tlc.engine.validator import TLCValidatorMixin
from ultralytics.utils.tlc.utils import create_sampler

class TLCClassificationValidator(TLCValidatorMixin, yolo.classify.ClassificationValidator):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = CLASSIFY_LABEL_COLUMN_NAME

    def check_dataset(self, *args, **kwargs):
        return tlc_check_cls_dataset(*args, **kwargs)

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader with given parameters."""
        dataset = self.build_dataset(dataset_path)
        
        sampler = create_sampler(dataset.table, mode="val", settings=self._settings)
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1, sampler=sampler)

    def build_dataset(self, table):
        return TLCClassificationDataset(
            table=table,
            args=self.args,
            augment=False,
            prefix=self.args.split,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
        )

    def _get_metrics_schemas(self):
        class_names=[value['internal_name'] for value in self.dataloader.dataset.table.get_value_map(self._label_column_name).values()]
        column_schemas = {
            "loss": tlc.Schema("Loss", "Cross Entropy Loss", writable=False, value=tlc.Float32Value()),
            "predicted": tlc.CategoricalLabelSchema(class_names=class_names, display_name="Predicted", description="The highest confidence class predicted by the model."),
            "confidence": tlc.Schema("Confidence", "The confidence of the prediction", value=tlc.Float32Value(value_min=0.0, value_max=1.0)),
            "top1_accuracy": tlc.Schema("Top-1 Accuracy", "The correctness of the prediction", value= tlc.Float32Value()),
        }

        if len(class_names) > 5:
            column_schemas["top5_accuracy"] = tlc.Schema(
                "Top-5 Accuracy",
                "The correctness of any of the top five confidence predictions",
                value=tlc.Float32Value()
            )

        return column_schemas

    def _compute_3lc_metrics(self, preds, batch):
        """ Compute 3LC classification metrics for each sample. """
        confidence, predicted = preds.max(dim=1)

        batch_metrics = {
            "loss": torch.nn.functional.nll_loss(torch.log(preds), batch["cls"], reduction="none"), #nll since preds are normalized
            "predicted": predicted,
            "confidence": confidence,
            "top1_accuracy": (torch.argmax(preds, dim=1) == batch["cls"]).to(torch.float32),
        }

        if len(self.dataloader.dataset.table.get_value_map(self._label_column_name)) > 5:
            _, top5_pred = torch.topk(preds, 5, dim=1)
            labels_expanded = batch["cls"].view(-1, 1).expand_as(top5_pred)
            top5_correct = torch.any(top5_pred == labels_expanded, dim=1)
            batch_metrics["top5_accuracy"] = top5_correct.to(torch.float32)

        return batch_metrics

    def _add_embeddings_hook(self, model):
        """ Add a hook to extract embeddings from the model, and infer the activation size. For a classification model, this
        amounts to finding the linear layer and extracting the input size."""
        
        # Find index of the linear layer
        has_linear_layer = False
        for index, module in enumerate(model.modules()):
            if isinstance(module, torch.nn.Linear):
                has_linear_layer = True
                activation_size = module.in_features
                break

        if not has_linear_layer:
            raise ValueError("No linear layer found in model, cannot collect embeddings.")

        weak_self = weakref.ref(self) # Avoid circular reference (self <-> hook_fn)
        def hook_fn(module, input, output):
            # Store embeddings
            self_ref = weak_self()
            embeddings = output.detach().cpu().numpy()
            self_ref.embeddings = embeddings

        # Add forward hook to collect embeddings
        for i, module in enumerate(model.modules()):
            if i == index - 1:
                self._hook_handles.append(module.register_forward_hook(hook_fn))

        return activation_size
    
    def _infer_batch_size(self, preds, batch) -> int:
        return preds.size(0)