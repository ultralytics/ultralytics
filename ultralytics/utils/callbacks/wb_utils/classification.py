from typing import Any, Optional

import numpy as np
import wandb as wb

from ultralytics.engine.results import Results
from ultralytics.models.yolo.classify import ClassificationPredictor


def plot_classification_predictions(
    result: Results, model_name: str, table: Optional[wb.Table] = None
):
    """Plot classification prediction results to a `wandb.Table` if the table is passed otherwise return the data."""
    result = result.to("cpu")
    probabilities = result.probs
    probabilities_list = probabilities.data.numpy().tolist()
    class_id_to_label = {int(k): str(v) for k, v in result.names.items()}
    table_row = [
        model_name,
        wb.Image(result.orig_img),
        class_id_to_label[int(probabilities.top1)],
        probabilities.top1conf,
        [class_id_to_label[int(class_idx)] for class_idx in list(probabilities.top5)],
        [probabilities_list[int(class_idx)] for class_idx in list(probabilities.top5)],
        {
            class_id_to_label[int(class_idx)]: probability
            for class_idx, probability in enumerate(probabilities_list)
        },
        result.speed,
    ]
    if table is not None:
        table.add_data(*table_row)
        return table
    return class_id_to_label, table_row
