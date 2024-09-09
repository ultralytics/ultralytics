"""
What do we want to test here?

# End-to-end

## Detection training coco128
## Detection metrics collection only coco128
## Classify training

# Settings

## collection epoch start + interval
## embeddings
## sampling weights
## zero weight exclusion
## val only collection

# Features
## 3LC YOLO YAML
## passing tables directly

"""
import pandas as pd
import pytest
import tlc

from ultralytics.utils.tlc import Settings, TLCYOLO
from ultralytics.models.yolo import YOLO

from tests import TMP

TMP_PROJECT_ROOT_URL = tlc.Url(TMP / "3LC")
tlc.Configuration.instance().project_root_url = TMP_PROJECT_ROOT_URL
tlc.TableIndexingTable.instance().add_scan_url(
    {
        "url": tlc.Url(TMP_PROJECT_ROOT_URL),
        "layout": "project",
        "object_type": "table",
        "static": True,
    }
)

def test_detect_training() -> None:
    # End-to-end test of detection
    overrides = {"data": "coco8.yaml", "epochs": 1, "batch": 4, "device": "cpu", "save": False, "plots": False}

    # Compare results from 3LC with ultralytics
    model_ultralytics = YOLO("yolov8n.pt")
    results_ultralytics = model_ultralytics.train(**overrides)

    settings = Settings(
        collection_epoch_start=1,
        image_embeddings_dim=2,
        image_embeddings_reducer="pacmap",
        project_name="test_detect_project",
        run_name="test_detect",
        run_description="Test detection training",
    )

    model_3lc = TLCYOLO("yolov8n.pt")
    results_3lc = model_3lc.train(**overrides, settings=settings)
    assert results_3lc, "Detection training failed"

    # Compare 3LC integration with ultralytics results
    assert results_ultralytics.results_dict == results_3lc.results_dict, "Results validation metrics 3LC different from Ultralytics"
    assert results_ultralytics.names == results_3lc.names, "Results validation names"

    # Get 3LC run and inspect the results
    run_url = TMP_PROJECT_ROOT_URL / settings.project_name / "runs" / settings.run_name
    run = tlc.Run.from_url(run_url)

    assert run.project_name == settings.project_name, "Project name mismatch"
    assert run.description == settings.run_description, "Description mismatch"
    # Check that hyperparameters and overrides are saved
    for key, value in overrides.items():
        assert run.constants["parameters"][key] == value, f"Parameter {key} mismatch, {run.constants['parameters'][key]} != {value}"
    
    # Check that there is a per-epoch value written
    assert len(run.constants["outputs"]) > 0, "No outputs written"

    # Check that the desired metrics were written
    metrics_df = pd.concat([metrics_table.to_pandas() for metrics_table in run.metrics_tables], ignore_index=True)
    
    embeddings_column_name = f"embeddings_{settings.image_embeddings_reducer}"
    assert embeddings_column_name in metrics_df.columns, "Expected embeddings column missing"
    assert len(metrics_df[embeddings_column_name][0]) == settings.image_embeddings_dim, "Embeddings dimension mismatch"

    assert 0 in metrics_df["Training Phase"], "Expected metrics from during training"
    assert 1 in metrics_df["Training Phase"], "Expected metrics from after training"

def test_detect_metrics_collection() -> None:
    overrides = {"device": "cpu"}
    model = TLCYOLO("yolov8n.pt")
    settings = Settings()

    splits = ("train", "val")
    results_dict = model.collect(data="coco8.yaml", splits=splits, settings=settings, **overrides)
    assert all(results_dict[split] for split in splits), "Metrics collection failed"
    # TODO: Test run created and looks as expected

def test_illegal_reducer() -> None:
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="illegal_reducer")
    with pytest.raises(Exception):
        settings.verify(training=False)

def test_missing_reducer() -> None:
    # umap-learn not installed in the test env, so using it should fail
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="umap")
    with pytest.raises(Exception):
        settings.verify(training=False)

@pytest.mark.parametrize(
    "start,interval,epochs,disable,expected",
    [
        (1, 1, 10, False, list(range(1,11))), # Start at 1, interval 1, 10 epochs
        (1, 2, 10, False, [1, 3, 5, 7, 9]),   # Start at 1, interval 2, 5 epochs
        (None, 2, 10, False, []),             # No start means no collection
        (0, 1, 10, False, ValueError),        # Start must be positive
        (1, 0, 10, False, ValueError),        # Interval must be positive
        (1, 1, 10, True, []),                 # Disable collection, no mc
    ],
)
def test_get_metrics_collection_epochs(start, interval, epochs, disable, expected) -> None:
    settings = Settings(
        collection_epoch_start=start,
        collection_epoch_interval=interval,
        collection_disable=disable
    )
    if isinstance(expected, list):
        collection_epochs = settings.get_metrics_collection_epochs(epochs)
        assert collection_epochs == expected, f"Expected {expected}, got {collection_epochs}"
    else:
        with pytest.raises(expected):
            settings.get_metrics_collection_epochs(epochs)
