# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc
import torch
import yaml

from packaging import version
from pathlib import Path

from .constants import TRAINING_PHASE

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.tlc.constants import TLC_COLORSTR, TLC_REQUIRED_VERSION, TLC_PREFIX
from ultralytics.utils.tlc.settings import Settings

from typing import Callable, Literal


def check_tlc_dataset(
    data: str,
    tables: dict[str, tlc.Table | tlc.Url | str] | None,
    image_column_name: str,
    label_column_name: str,
    dataset_checker: Callable[[str], dict[str, object]] | None = None,
    table_creator: Callable[[str, dict[str, object], str, str, str, str, str], tlc.Table] | None = None,
    table_checker: Callable[[str, tlc.Table], bool] | None = None,
    project_name: str | None = None,
    check_backwards_compatible_table_name: bool = False,
) -> dict[str, tlc.Table | dict[float, str] | int]:
    """ Get or create tables for YOLOv8 datasets. data is ignored when tables is provided.

    :param data: Path to a dataset
    :param tables: Dictionary of tables, if already created
    :param image_column_name: Name of the column containing image paths
    :param label_column_name: Name of the column containing labels
    :param dataset_checker: Function to check the dataset (yolo implementation, download and checks)
    :param table_creator: Function to create the tables for the YOLO dataset
    :param table_checker: Function to check that a table is compatible with the current task
    :param project_name: Name of the project
    :param check_backwards_compatible_table_name: Whether to check for a backwards compatible table name
    :return: Dictionary of tables and class names
    """
    # If the data starts with the 3LC prefix, parse the YAML file and populate `tables`
    if tables is None and data.startswith(TLC_PREFIX):
        LOGGER.info(f"{TLC_COLORSTR}Parsing 3LC YAML file data={data} and populating tables")
        tables = parse_3lc_yaml_file(data)

    if tables is None:
        tables = {}

        data_dict = dataset_checker(data)

        # Get or create tables
        LOGGER.info(f"{TLC_COLORSTR}Creating or reusing tables from data={data}")

        for key in ("train", "val", "test", "minival"):
            if data_dict.get(key):
                name = Path(data).stem
                dataset_name = f"{name}-{key}"
                table_name = "initial"

                if project_name is None:
                    project_name = f"{name}-YOLOv8"

                # Previously the table name was "original" and now it is "initial", so we need to check for backwards compatibility
                if check_backwards_compatible_table_name:
                    table_url_backcompatible = tlc.Table._resolve_table_url(
                        table_url=None,
                        root_url=None,
                        project_name=project_name,
                        dataset_name=dataset_name,
                        table_name="original",
                    )

                    if table_url_backcompatible.exists():
                        table_name = "original"

                table = table_creator(key,
                                      data_dict,
                                      image_column_name=image_column_name,
                                      label_column_name=label_column_name,
                                      project_name=project_name,
                                      dataset_name=dataset_name,
                                      table_name=table_name)

                # Get the latest version when inferring
                tables[key] = table.latest()

                if tables[key] != table:
                    LOGGER.info(f"   {colorstr(key)}: Using latest version of table {table.url} -> {tables[key].url}")
                else:
                    LOGGER.info(f"   {colorstr(key)}: Using initial version of table {tables[key].url}")

    else:
        LOGGER.info(f"{TLC_COLORSTR}Using data directly from tables")
        for key, table in tables.items():
            if isinstance(table, (str, Path, tlc.Url)):
                try:
                    table_url = tlc.Url(table)
                    tables[key] = tlc.Table.from_url(table_url)
                except Exception as e:
                    raise ValueError(
                        f"Error loading table from {table} for split '{key}' provided through `tables`.") from e
            elif isinstance(table, tlc.Table):
                tables[key] = table
            else:
                raise ValueError(
                    f"Invalid type {type(table)} for split {key} provided through `tables`."
                    "Must be a tlc.Table object or a location (string, pathlib.Path or tlc.Url) of a tlc.Table.")

            # Check that the table is compatible with the current task
            if table_checker is not None:
                table_checker(tables[key], image_column_name, label_column_name)

            LOGGER.info(f"   - {key}: {tables[key].url}")

    first_split = next(iter(tables.keys()))
    value_map = tables[first_split].get_value_map(label_column_name)
    names = {int(k): v['internal_name'] for k, v in value_map.items()}

    return {**tables, "names": names, "nc": len(names)}


def training_phase_schema() -> tlc.Schema:
    """Create a 3LC schema for the training phase.

    :returns: The training phase schema.
    """
    return tlc.Schema(
        display_name=TRAINING_PHASE,
        description=("'During' metrics are collected with EMA during training, "
                     "'After' is with the final model weights after completed training."),
        display_importance=tlc.DISPLAY_IMPORTANCE_EPOCH - 1,  # Right hand side of epoch in the Dashboard
        writable=False,
        computable=False,
        value=tlc.Int32Value(
            value_min=0,
            value_max=1,
            value_map={
                float(0): tlc.MapElement(display_name='During'),
                float(1): tlc.MapElement(display_name='After'), },
        ))


def image_embeddings_schema(activation_size=512) -> dict[str, tlc.Schema]:
    """ Create a 3LC schema for YOLOv8 image embeddings.

    :param activation_size: The size of the activation tensor.
    :returns: The YOLO image embeddings schema.
    """
    embedding_schema = tlc.Schema('Embedding',
                                  'Large NN embedding',
                                  writable=False,
                                  computable=False,
                                  value=tlc.Float32Value(number_role=tlc.NUMBER_ROLE_NN_EMBEDDING),
                                  size0=tlc.DimensionNumericValue(value_min=activation_size,
                                                                  value_max=activation_size,
                                                                  enforce_min=True,
                                                                  enforce_max=True))
    return embedding_schema


def reduce_embeddings(
    run: tlc.Run,
    method: str,
    n_components: int,
    foreign_table_url: tlc.Url | None = None,
):
    """Reduce image embeddings by a foreign table URL."""
    if foreign_table_url is None:
        foreign_table_url = tlc.Url(tlc.active_run().constants['inputs'][0]['input_table_url']).to_absolute(
            tlc.active_run().url)

    LOGGER.info(TLC_COLORSTR +
                f"Reducing image embeddings to {n_components}D with {method}, this may take a few minutes...")
    run.reduce_embeddings_by_foreign_table_url(
        foreign_table_url=foreign_table_url,
        method=method,
        n_components=n_components,
    )


def check_tlc_version():
    """Check the 3LC version."""
    installed_version = version.parse(tlc.__version__)
    if installed_version < version.parse(TLC_REQUIRED_VERSION):
        raise ValueError(
            f"3LC version {tlc.__version__} is too old to use the YOLOv8 integration. "
            f"Please upgrade to version {TLC_REQUIRED_VERSION} or later by running 'pip install --upgrade 3lc'.")


def parse_3lc_yaml_file(data_file: str) -> dict[str, tlc.Table]:
    """ Parse a 3LC YAML file and return the corresponding tables.
    
    :param data_file: The path to the 3LC YAML file.
    :returns: The tables pointed to by the YAML file.
    """
    # Read the YAML file, removing the prefix
    if not (data_file_url := tlc.Url(data_file.replace(TLC_PREFIX, ""))).exists():
        raise FileNotFoundError(f"Could not find YAML file {data_file_url}")

    data_config = yaml.safe_load(data_file_url.read())

    path = data_config.get("path")
    splits = [key for key in data_config if key != "path"]

    tables = {}
    for split in splits:
        # Handle :latest at the end
        if data_config[split].endswith(":latest"):
            latest = True
            split_path = data_config[split][:-len(":latest")]
        else:
            latest = False
            split_path = data_config[split]

        if split_path.startswith("./"):
            LOGGER.debug(f"{TLC_COLORSTR}{split} split path starts with './', removing it.")
            split_path = split_path[2:]

        table_url = tlc.Url(path) / split_path if path else tlc.Url(split_path)

        table = tlc.Table.from_url(table_url)

        if latest:
            table = table.latest()

        tables[split] = table

    return tables


def create_sampler(table: tlc.Table,
                   mode: Literal["train", "val"],
                   settings: Settings,
                   distributed: bool = False) -> torch.utils.data.Sampler | None:
    """Get the sampler for the dataset.
    
    :param table: The table to get the sampler for.
    :param mode: The mode of the sampler.
    :param settings: The settings for the run.
    :param distributed: Whether training is distributed.
    :returns: The sampler for the dataset.
    """
    sampler = None

    if mode == "train":
        if settings.sampling_weights or settings.exclude_zero_weight_training:
            if distributed:
                raise NotImplementedError("Distributed training and using 3LC weights is not yet supported.")

            try:
                sampler = table.create_sampler(
                    exclude_zero_weights=settings.exclude_zero_weight_training,
                    weighted=settings.sampling_weights,
                    shuffle=True,
                )
            except Exception as e:
                raise ValueError(f"Error creating sampler for table {table.url}") from e

    elif mode == "val":
        if distributed:
            raise NotImplementedError("Distributed validation and exclusion by weight is not yet supported.")

        # Exclude zero weight is handled in the dataset for validation
        return None
    return sampler
