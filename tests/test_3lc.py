from collections import defaultdict
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import tlc

from ultralytics.utils.tlc import Settings, TLCYOLO, TLCClassificationTrainer, TLCDetectionTrainer
from ultralytics.utils.tlc.classify.utils import tlc_check_cls_dataset
from ultralytics.utils.tlc.detect.utils import tlc_check_det_dataset
from ultralytics.utils.tlc.utils import check_tlc_dataset
from ultralytics.models.yolo import YOLO

from tests import TMP
from ultralytics.utils.tlc.constants import (
    DEFAULT_COLLECT_RUN_DESCRIPTION,
    MAP,
    MAP50_95,
    NUM_IMAGES,
    NUM_INSTANCES,
    PER_CLASS_METRICS_STREAM_NAME,
    PRECISION,
    RECALL,
    TRAINING_PHASE,
)

TMP_PROJECT_ROOT_URL = tlc.Url(TMP / "3LC")
tlc.Configuration.instance().project_root_url = TMP_PROJECT_ROOT_URL
tlc.TableIndexingTable.instance().add_scan_url({
    "url": tlc.Url(TMP_PROJECT_ROOT_URL),
    "layout": "project",
    "object_type": "table",
    "static": True, })

TASK2DATASET = {"detect": "coco8.yaml", "classify": "imagenet10"}
TASK2MODEL = {"detect": "yolo11n.pt", "classify": "yolo11n-cls.pt"}
TASK2LABEL_COLUMN_NAME = {"detect": "bbs.bb_list.label", "classify": "label"}
TASK2PREDICTED_LABEL_COLUMN_NAME = {"detect": "bbs_predicted.bb_list.label", "classify": "predicted"}

try:
    import umap

    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False


def get_metrics_tables_from_run(run: tlc.Run) -> dict[str, list[tlc.Table]]:
    """Return metrics tables grouped by stream name"""
    metrics_infos = run.metrics
    metrics_tables = defaultdict(list)
    for metrics_info in metrics_infos:
        metrics_table = tlc.Table.from_url(tlc.Url(metrics_info["url"]).to_absolute(run.url))
        metrics_tables[metrics_info["stream_name"]].append(metrics_table)
    return metrics_tables


def test_detect_training() -> None:
    # End-to-end test of detection
    overrides = {
        "data": TASK2DATASET["detect"],
        "epochs": 1,
        "batch": 4,
        "device": "cpu",
        "save": False,
        "plots": False, }

    # Compare results from 3LC with ultralytics
    model_ultralytics = YOLO(TASK2MODEL["detect"])
    results_ultralytics = model_ultralytics.train(**overrides)

    settings = Settings(
        collection_epoch_start=1,
        collect_loss=True,
        image_embeddings_dim=2,
        image_embeddings_reducer="pacmap",
        project_name="test_detect_project",
        run_name="test_detect",
        run_description="Test detection training",
    )

    model_3lc = TLCYOLO(TASK2MODEL["detect"])
    results_3lc = model_3lc.train(**overrides, settings=settings)
    assert results_3lc, "Detection training failed"

    # Compare 3LC integration with ultralytics results
    assert (results_ultralytics.results_dict == results_3lc.results_dict
            ), "Results validation metrics 3LC different from Ultralytics"
    assert results_ultralytics.names == results_3lc.names, "Results validation names"

    # Get 3LC run and inspect the results
    run = _get_run_from_settings(settings)

    assert run.status == tlc.RUN_STATUS_COMPLETED, "Run status not set to completed after training"

    assert run.project_name == settings.project_name, "Project name mismatch"
    assert run.description == settings.run_description, "Description mismatch"
    # Check that hyperparameters and overrides are saved
    for key, value in overrides.items():
        assert (run.constants["parameters"][key] == value
                ), f"Parameter {key} mismatch, {run.constants['parameters'][key]} != {value}"

    # Check that confidence-recall-precision-f1 data is written
    assert "3LC/Precision" in run.constants["parameters"]
    assert "3LC/Recall" in run.constants["parameters"]
    assert "3LC/F1_Score" in run.constants["parameters"]
    
    # Check that there is a per-epoch value written
    assert len(run.constants["outputs"]) > 0, "No outputs written"
    metrics_tables = get_metrics_tables_from_run(run)

    # Check that the desired metrics were written
    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )

    embeddings_column_name = f"embeddings_{settings.image_embeddings_reducer}"
    assert embeddings_column_name in metrics_df.columns, "Expected embeddings column missing"
    assert len(metrics_df[embeddings_column_name][0]) == settings.image_embeddings_dim, "Embeddings dimension mismatch"

    assert "loss" in metrics_df.columns, "Expected loss column to be present, but it is missing"
    assert 0 in metrics_df[TRAINING_PHASE], "Expected metrics from during training"
    assert 1 in metrics_df[TRAINING_PHASE], "Expected metrics from after training"

    # model.predict() should work and be the same as vanilla ultralytics
    assert all(model_ultralytics.predict(imgsz=320)[0].boxes.cls == model_3lc.predict(
        imgsz=320)[0].boxes.cls), "Predictions mismatch"

    per_class_metrics_tables = metrics_tables[PER_CLASS_METRICS_STREAM_NAME]
    assert len(per_class_metrics_tables) == 4, "Expected 4 per-class metrics tables to be written"
    per_class_metrics_df = pd.concat(
        [m.to_pandas() for m in per_class_metrics_tables],
        ignore_index=True,
    )
    assert TRAINING_PHASE in per_class_metrics_df.columns, "Expected training phase column in per-class metrics"
    assert tlc.EPOCH in per_class_metrics_df.columns, "Expected epoch column in per-class metrics"
    assert tlc.FOREIGN_TABLE_ID in per_class_metrics_df.columns, "Expected foreign_table_id column in per-class metrics"
    assert tlc.LABEL in per_class_metrics_df.columns, "Expected label column in per-class metrics"
    assert PRECISION in per_class_metrics_df.columns, "Expected precision column in per-class metrics"
    assert RECALL in per_class_metrics_df.columns, "Expected recall column in per-class metrics"
    assert MAP in per_class_metrics_df.columns, "Expected mAP column in per-class metrics"
    assert MAP50_95 in per_class_metrics_df.columns, "Expected mAP50-95 column in per-class metrics"
    assert NUM_IMAGES in per_class_metrics_df.columns, "Expected num_images column in per-class metrics"
    assert NUM_INSTANCES in per_class_metrics_df.columns, "Expected num_instances column in per-class metrics"


def test_classify_training() -> None:
    model = TASK2MODEL["classify"]
    overrides = {"data": TASK2DATASET["classify"], "device": "cpu", "epochs": 3, "batch": 64, "imgsz": 32}

    # Compare results from 3LC with ultralytics
    model_ultralytics = YOLO(model)
    results_ultralytics = model_ultralytics.train(**overrides)

    model_3lc = TLCYOLO(model)

    settings = Settings(
        image_embeddings_dim=3,
        collection_epoch_start=2,
        collection_epoch_interval=1,
        project_name="test_classify_project",
        run_name="test_classify",
    )
    results_3lc = model_3lc.train(**overrides, settings=settings)

    assert results_3lc, "Classification training failed"

    assert (results_ultralytics.results_dict == results_3lc.results_dict
            ), "Results validation metrics 3LC different from Ultralytics"

    run = _get_run_from_settings(settings)

    assert run.status == tlc.RUN_STATUS_COMPLETED, "Run status not set to completed after training"
    assert not run.description, "Description mismatch, default should be empty string"

    assert len(run.metrics_tables) == 6  # Two passes after epochs 2 and 3, and after training

    # Check that the desired metrics were written
    metrics_df = pd.concat(
        [metrics_table.to_pandas() for metrics_table in run.metrics_tables],
        ignore_index=True,
    )

    assert 0 in metrics_df[TRAINING_PHASE], "Expected metrics from during training"
    assert 1 in metrics_df[TRAINING_PHASE], "Expected metrics from after training"

    # Aggregate per-sample metrics should match the output aggregate metrics
    val_after_metrics_df = run.metrics_tables[-1].to_pandas()  # Val metrics after training should be written last

    metrics_top1_accuracy = val_after_metrics_df["top1_accuracy"].mean()
    metrics_top5_accuracy = val_after_metrics_df["top5_accuracy"].mean()

    assert np.isclose(metrics_top1_accuracy, results_3lc.top1)
    assert np.isclose(metrics_top5_accuracy, results_3lc.top5)

    embeddings_column_name = f"embeddings_{settings.image_embeddings_reducer}"
    assert embeddings_column_name in metrics_df.columns, "Expected embeddings column missing"
    assert len(metrics_df[embeddings_column_name][0]) == settings.image_embeddings_dim, "Embeddings dimension mismatch"

    # Test metrics collection only here with the same weights (since there are no readily available pretrained weights for the ten-class case)
    best = results_3lc.save_dir / "weights" / "best.pt"
    best_model = TLCYOLO(best)
    results_dict = best_model.collect(
        data=TASK2DATASET["classify"],
        splits=("train", "val"),
        settings=settings,
    )

    assert (results_dict["val"].results_dict == results_ultralytics.results_dict
            ), "Results validation metrics collection onlywith  3LC different from Ultralytics"

    # model.predict() should work and be the same as vanilla ultralytics
    preds_3lc = model_3lc.predict(imgsz=320)
    preds_ultralytics = model_ultralytics.predict(imgsz=320)

    assert preds_3lc[0].probs.top5 == preds_ultralytics[0].probs.top5, "Predictions mismatch"


@pytest.mark.parametrize("task", ["detect"])
def test_metrics_collection_only(task) -> None:
    overrides = {"device": "cpu"}
    settings = Settings(project_name=f"test_{task}_collect", run_name=f"test_{task}_collect", collect_loss=True)
    splits = ("train", "val")

    model = TLCYOLO(TASK2MODEL[task])
    results_dict = model.collect(data=TASK2DATASET[task], splits=splits, settings=settings, **overrides)
    assert all(results_dict[split] for split in splits), "Metrics collection failed"

    run_urls = [results_dict[split].run_url for split in splits]
    assert run_urls[0] == run_urls[1], "Expected same run URL for both splits"

    run = tlc.Run.from_url(run_urls[0])
    metrics_tables = get_metrics_tables_from_run(run)

    metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables["default_stream"]],
        ignore_index=True,
    )
    assert "loss" not in metrics_df.columns, "Expected no loss column"
    assert run.status == tlc.RUN_STATUS_COMPLETED, "Run status not set to completed after training"
    assert run.description == DEFAULT_COLLECT_RUN_DESCRIPTION, "Description mismatch"
    assert len(metrics_tables[PER_CLASS_METRICS_STREAM_NAME]) == 2, "Expected 2 per-class metrics tables (train, val)"

    per_class_metrics_df = pd.concat(
        [m.to_pandas() for m in metrics_tables[PER_CLASS_METRICS_STREAM_NAME]],
        ignore_index=True,
    )
    assert TRAINING_PHASE not in per_class_metrics_df.columns, "Expected no training phase column"
    assert tlc.EPOCH not in per_class_metrics_df.columns, "Expected no epoch column"


def test_train_collection_val_only() -> None:
    task = "classify"
    model_arg = TASK2MODEL[task]
    overrides = {"data": TASK2DATASET[task], "device": "cpu", "epochs": 1, "batch": 4, "imgsz": 224}

    model = TLCYOLO(model_arg)

    settings = Settings(
        collection_val_only=True,
        project_name="test_train_collect_val_only",
        run_name="test_train_collect_val_only",
    )
    model.train(**overrides, settings=settings)

    # Ensure that only validation metrics are collected after training
    run = _get_run_from_settings(settings)
    assert len(run.metrics_tables) == 1, "Expected only validation metrics to be collected after training"


def test_train_collection_disabled() -> None:
    task = "classify"
    model_arg = TASK2MODEL[task]
    overrides = {"data": TASK2DATASET[task], "device": "cpu", "epochs": 1, "batch": 4, "imgsz": 224}

    model = TLCYOLO(model_arg)

    settings = Settings(
        collection_disable=True,
        project_name="test_train_collection_disabled",
        run_name="test_train_collection_disabled",
    )
    model.train(**overrides, settings=settings)

    # Ensure that only validation metrics are collected after training
    run = _get_run_from_settings(settings)
    assert len(run.metrics_tables) == 0, "Expected no metrics tables to be written"


def test_invalid_tables() -> None:
    # Test that an error is raised if the tables are not formatted as desired
    for model_arg in TASK2MODEL.values():
        model = TLCYOLO(model_arg)
        table = tlc.Table.from_dict({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError):
            model.train(tables={"train": table, "val": table})


def test_table_resolving() -> None:
    # Check that repeated runs with 'data' resolve to the same tables, or the latest
    settings = Settings(project_name="test_table_resolving")
    trainer = TLCDetectionTrainer(overrides={"data": TASK2DATASET["detect"], "settings": settings})

    # Create initial tables
    train_table = trainer.trainset

    # Create an edited version of the train table
    train_table_edited = tlc.NullOverlay(
        train_table.url.create_sibling("peter").create_unique(),
        input_table_url=train_table,
    )

    # A new trainer should now use the edited table since it gets latest
    new_trainer = TLCDetectionTrainer(overrides={"data": TASK2DATASET["detect"], "settings": settings})
    assert new_trainer.trainset.url == train_table_edited.url, "Table not resolved correctly"

    # A new trainer should not be able to take the tables directly
    tables = {"train": train_table_edited.url, "val": new_trainer.testset.url}
    trainer_from_tables = TLCDetectionTrainer(overrides={"tables": tables, "settings": settings})
    trainer_from_tables.trainset.url == train_table_edited.url, "Table passed directly not resolved correctly"


def test_sampling_weights() -> None:
    # Test that sampling weights are correctly applied, with worker processes enabled
    settings = Settings(project_name="test_sampling_weights", sampling_weights=True)
    trainer = TLCDetectionTrainer(overrides={"data": TASK2DATASET["detect"], "settings": settings, "workers": 4})
    epochs = 1000

    # Create edited table where one sample has weight increased to 2
    edited_table = tlc.EditedTable(
        url=trainer.trainset.url.create_sibling("jonas"),
        input_table_url=trainer.trainset,
        edits={tlc.SAMPLE_WEIGHT: {
            "runs_and_values": [[0], 2.0]}},
    )

    dataloader = trainer.get_dataloader(edited_table, batch_size=2, rank=-1)

    sampled_example_ids = []
    for epoch in range(epochs):
        for batch in dataloader:
            sampled_example_ids.extend(batch["example_id"])

    # Check other samples are sampled within [0.45, 0.55] of the time of the first
    counts = np.bincount(sampled_example_ids)
    relative_counts = counts[1:] / counts[0]
    assert np.allclose(
        relative_counts,
        np.full_like(relative_counts, 0.5),
        atol=0.05,
    ), f"First sample should be sampled twice as often as others, got {counts}"
    assert len(sampled_example_ids) == len(edited_table) * epochs, "Expected no change in the number of samples"


def test_exclude_zero_weight_training() -> None:
    # Test that sampling weights are correctly applied, with worker processes enabled
    settings = Settings(project_name="test_exclude_zero_weight_training", exclude_zero_weight_training=True)
    trainer = TLCDetectionTrainer(overrides={"data": TASK2DATASET["detect"], "settings": settings, "workers": 4})

    # Create edited table where one sample has weight increased to 2
    edited_table = tlc.EditedTable(
        url=trainer.trainset.url.create_sibling("jonas"),
        input_table_url=trainer.trainset,
        edits={tlc.SAMPLE_WEIGHT: {
            "runs_and_values": [[0], 0.0]}},
    )

    dataloader = trainer.get_dataloader(edited_table, batch_size=2, rank=-1)
    sampled_example_ids = []
    for batch in dataloader:
        sampled_example_ids.extend(batch["example_id"])

    assert 0 not in sampled_example_ids, "Sample with zero weight should not be included in training"
    assert len(sampled_example_ids) == len(edited_table) - 1, "Expected one sample to be excluded"


@pytest.mark.parametrize("task,trainer_class", [("detect", TLCDetectionTrainer),
                                                ("classify", TLCClassificationTrainer)])
def test_exclude_zero_weight_collection(task, trainer_class) -> None:
    # Test that sampling weights are correctly applied during metrics collection
    settings = Settings(project_name=f"test_sampling_weights_collection_{task}", exclude_zero_weight_collection=True)
    trainer = trainer_class(overrides={"data": TASK2DATASET[task], "settings": settings, "workers": 2})

    # Classification trainer needs a model object to create dataloader
    if task == "classify":
        trainer.model = Mock()

    # Create edited table where several samples have weight 0
    edited_table = tlc.EditedTable(
        url=trainer.trainset.url.create_sibling(f"erna_{task}"),
        input_table_url=trainer.trainset,
        edits={tlc.SAMPLE_WEIGHT: {
            "runs_and_values": [[0, 3], 0.0]}},
    )

    dataloader = trainer.get_dataloader(edited_table, batch_size=2, rank=-1, mode="val")
    sampled_example_ids = []
    for batch in dataloader:
        sampled_example_ids.extend(batch["example_id"])

    assert 0 not in sampled_example_ids, "Sample with zero weight should not be included in collection"
    assert 3 not in sampled_example_ids, "Sample with zero weight should not be included in collection"
    assert len(sampled_example_ids) == len(edited_table) - 2, "Expected two samples to be excluded"


def test_illegal_reducer() -> None:
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="illegal_reducer")
    with pytest.raises(Exception):
        settings.verify(training=False)


@pytest.mark.skipif(UMAP_AVAILABLE, reason="Test assumes umap is not installed")
def test_missing_reducer() -> None:
    # umap-learn not installed in the test env, so using it should fail
    settings = Settings(image_embeddings_dim=2, image_embeddings_reducer="umap")
    with pytest.raises(Exception):
        settings.verify(training=False)


@pytest.mark.parametrize(
    "start,interval,epochs,disable,expected",
    [
        (1, 1, 10, False, list(range(1, 11))),  # Start at 1, interval 1, 10 epochs
        (1, 2, 10, False, [1, 3, 5, 7, 9]),  # Start at 1, interval 2, 5 epochs
        (None, 2, 10, False, []),  # No start means no collection
        (0, 1, 10, False, ValueError),  # Start must be positive
        (1, 0, 10, False, ValueError),  # Interval must be positive
        (1, 1, 10, True, []),  # Disable collection, no mc
    ],
)
def test_get_metrics_collection_epochs(start, interval, epochs, disable, expected) -> None:
    settings = Settings(collection_epoch_start=start, collection_epoch_interval=interval, collection_disable=disable)
    if isinstance(expected, list):
        collection_epochs = settings.get_metrics_collection_epochs(epochs)
        assert collection_epochs == expected, f"Expected {expected}, got {collection_epochs}"
    else:
        with pytest.raises(expected):
            settings.get_metrics_collection_epochs(epochs)


@pytest.mark.parametrize("task", ["detect", "classify"])
def test_arbitrary_class_indices(task) -> None:
    # Test that arbitrary class indices can be used
    settings = Settings(
        project_name=f"test_arbitrary_class_indices_{task}",
        run_name=f"test_arbitrary_class_indices_{task}",
    )

    label_column_name = TASK2LABEL_COLUMN_NAME[task]
    predicted_label_column_name = TASK2PREDICTED_LABEL_COLUMN_NAME[task]

    if task == "detect":
        data_dict = tlc_check_det_dataset(
            data=TASK2DATASET["detect"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
        )
    elif task == "classify":
        data_dict = tlc_check_cls_dataset(
            data=TASK2DATASET["classify"],
            tables=None,
            image_column_name="image",
            label_column_name=label_column_name,
            project_name=settings.project_name,
        )

    # Create edited tables where class indices are changed
    edited_tables = {}
    for split in ("train", "val"):
        table = data_dict[split]
        table_value_map = table.get_value_map(label_column_name)
        label_map = {k: -k ** 2 for k in table_value_map.keys()}  # 0, 1, 2, ... -> 0, -1, -4, ...
        edited_value_map = {label_map[k]: v for k, v in table_value_map.items()}
        edited_schema_table = table.set_value_map(label_column_name, edited_value_map)

        if task == "detect":
            bbs_edits = []
            for i, row in enumerate(edited_schema_table.table_rows):
                bb_list_override = []
                for bb in row["bbs"]["bb_list"]:
                    bb_list_override.append({**bb, "label": label_map[bb["label"]]})

                bbs_edits.append([i])
                bbs_edits.append({**row["bbs"], "bb_list": bb_list_override})

            edited_tables[split] = tlc.EditedTable(
                url=edited_schema_table.url.create_sibling(f"edited_value_map_and_values_{task}"),
                input_table_url=edited_schema_table,
                edits={"bbs": {
                    "runs_and_values": bbs_edits}},
            )
        elif task == "classify":
            edits = []
            for i, row in enumerate(edited_schema_table.table_rows):
                edits.append([i])
                edits.append(label_map[row[label_column_name]])

            edited_tables[split] = tlc.EditedTable(
                url=edited_schema_table.url.create_sibling(f"edited_value_map_and_values_{task}"),
                input_table_url=edited_schema_table,
                edits={label_column_name: {
                    "runs_and_values": edits}},
            )

    # Check that the edited table can be used for training and validation
    model = TLCYOLO(TASK2MODEL[task])
    results = model.train(tables=edited_tables, settings=settings, epochs=1, device="cpu")

    assert results, f"{task} training with arbitrary class indices failed"

    run = _get_run_from_settings(settings)

    # Verify metrics have the expected class indices
    sample_metrics_tables = [
        m for m in run.metrics_tables if TASK2PREDICTED_LABEL_COLUMN_NAME[task].split(".")[0] in m.columns]
    metrics_df = pd.concat(
        [metrics_table.to_pandas() for metrics_table in sample_metrics_tables],
        ignore_index=True,
    )

    if task == "detect":
        for i in range(len(metrics_df)):
            assert all(bb["label"] <= 0 for bb in metrics_df["bbs_predicted"][i]["bb_list"])

        # Verify that a giraffe is predicted in the second image
        predicted_label = np.sqrt(-metrics_df["bbs_predicted"][1]["bb_list"][0]["label"])
        assert table_value_map[predicted_label]["internal_name"] == "giraffe"
    elif task == "classify":
        assert all(label <= 0 for label in metrics_df[predicted_label_column_name]), "Predicted label indices mismatch"

    # Verify that the metrics schema is correct
    label_value_map = edited_tables["train"].get_value_map(label_column_name)
    predicted_label_value_map = sample_metrics_tables[0].get_value_map(predicted_label_column_name)
    assert label_value_map == predicted_label_value_map, "Predicted label value map mismatch"


def test_check_tlc_dataset_different_categories() -> None:
    # Test that an error is raised if the categories of the tables are different
    project_name = "test_check_tlc_dataset_different_categories"

    train_structure = {"image": tlc.ImagePath("image"), "label": tlc.CategoricalLabel("label", classes=["a", "b", "c"])}
    val_structure = {"image": tlc.ImagePath("image"), "label": tlc.CategoricalLabel("label", classes=["a", "b", "d"])}

    train_table = tlc.Table.from_dict(
        {
            "image": ["a.jpg", "b.jpg"],
            "label": [0, 1]},
        structure=train_structure,
        project_name=project_name,
        dataset_name="train",
    )
    val_table = tlc.Table.from_dict(
        {
            "image": ["c.jpg", "d.jpg"],
            "label": [0, 1]},
        structure=val_structure,
        project_name=project_name,
        dataset_name="val",
    )

    with pytest.raises(ValueError):
        check_tlc_dataset(
            data="",
            tables={
                "train": train_table,
                "val": val_table, },
            image_column_name="image",
            label_column_name="label",
        )


def test_check_tlc_dataset_bad_tables() -> None:
    # Test that an error is raised if tables or urls are not provided properly
    tables = {"train": [1, 2, 3], "val": [4, 5, 6]}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b")


def test_check_tlc_dataset_bad_url() -> None:
    # Test that an error is raised if a non-valid url is provided
    tables = {"train": "some_url", "val": "some_other_url"}

    with pytest.raises(ValueError):
        check_tlc_dataset(data="", tables=tables, image_column_name="a", label_column_name="b")


# HELPERS
def _get_run_from_settings(settings: Settings) -> tlc.Run:
    run_url = TMP_PROJECT_ROOT_URL / settings.project_name / "runs" / settings.run_name
    return tlc.Run.from_url(run_url)
