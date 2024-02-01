# YOLOv5 🚀 AGPL-3.0 license
"""
3LC utils
"""
from __future__ import annotations

import importlib
import os
from difflib import get_close_matches
from pathlib import Path
from typing import Callable, TypeVar

import tlc
import yaml
from tlc.client.torch.metrics.metrics_collectors.bounding_box_metrics_collector import (_TLCPredictedBoundingBox,
                                                                                        _TLCPredictedBoundingBoxes)
from tlc.core.builtins.constants.paths import _ROW_CACHE_FILE_NAME
from tlc.core.objects.tables.from_url.utils import get_hash

from ultralytics.data.utils import check_file
from ultralytics.utils import LOGGER
from ultralytics.utils.tlc.detect.constants import TLC_COLORSTR, TLC_PREFIX, TLC_SUPPORTED_ENV_VARS, TRAINING_PHASE

T = TypeVar('T', bound=Callable)  # Generic type for environment variable parsing functions


def yolo_predicted_bounding_box_schema(categories: dict[int, str]) -> tlc.Schema:
    """ Create a 3LC bounding box schema for YOLOv5

    :param categories: Categories for the current dataset.
    :returns: The YOLO bounding box schema for predicted boxes.
    """
    label_value_map = {float(i): tlc.MapElement(class_name) for i, class_name in categories.items()}

    bounding_box_schema = tlc.BoundingBoxListSchema(
        label_value_map=label_value_map,
        x0_number_role=tlc.NUMBER_ROLE_BB_CENTER_X,
        x1_number_role=tlc.NUMBER_ROLE_BB_SIZE_X,
        y0_number_role=tlc.NUMBER_ROLE_BB_CENTER_Y,
        y1_number_role=tlc.NUMBER_ROLE_BB_SIZE_Y,
        x0_unit=tlc.UNIT_RELATIVE,
        y0_unit=tlc.UNIT_RELATIVE,
        x1_unit=tlc.UNIT_RELATIVE,
        y1_unit=tlc.UNIT_RELATIVE,
        description='Predicted Bounding Boxes',
        writable=False,
        is_prediction=True,
        include_segmentation=False,
    )

    return bounding_box_schema


def yolo_loss_schemas() -> dict[str, tlc.Schema]:
    """ Create a 3LC schema for YOLOv5 loss metrics.

    :returns: The YOLO loss schemas.
    """
    schemas = {}
    schemas['loss'] = tlc.Schema(description='Sample loss',
                                 writable=False,
                                 value=tlc.Float32Value(),
                                 display_importance=3003)
    schemas['box_loss'] = tlc.Schema(description='Box loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3004)
    schemas['obj_loss'] = tlc.Schema(description='Object loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3005)
    schemas['cls_loss'] = tlc.Schema(description='Classification loss',
                                     writable=False,
                                     value=tlc.Float32Value(),
                                     display_importance=3006)
    return schemas


def yolo_image_embeddings_schema(activation_size=512) -> dict[str, tlc.Schema]:
    """ Create a 3LC schema for YOLOv5 image embeddings.

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
    return {'embeddings': embedding_schema}


def construct_bbox_struct(
    predicted_annotations: list[dict[str, int | float | dict[str, float]]],
    image_width: int,
    image_height: int,
    inverse_label_mapping: dict[int, int] | None = None,
) -> _TLCPredictedBoundingBoxes:
    """Construct a 3LC bounding box struct from a list of bounding boxes.

    :param predicted_annotations: A list of predicted bounding boxes.
    :param image_width: The width of the image.
    :param image_height: The height of the image.
    :param inverse_label_mapping: A mapping from predicted label to category id.
    """

    bbox_struct = _TLCPredictedBoundingBoxes(
        bb_list=[],
        image_width=image_width,
        image_height=image_height,
    )

    for pred in predicted_annotations:
        bbox, label, score, iou = pred['bbox'], pred['category_id'], pred['score'], pred['iou']
        label_val = inverse_label_mapping[label] if inverse_label_mapping is not None else label
        bbox_struct['bb_list'].append(
            _TLCPredictedBoundingBox(
                label_predicted=label_val,
                confidence=score,
                iou=iou,
                x0=bbox[0],
                y0=bbox[1],
                x1=bbox[2],
                y1=bbox[3],
            ))

    return bbox_struct


def get_metrics_collection_epochs(start: int, epochs: int, interval: int, disable: bool) -> list[int]:
    """ Compute the epochs to collect metrics for.

    :param start: The starting epoch.
    :param epochs: The total number of epochs.
    :param interval: How frequently to collect metrics. 1 means every epoch, 2 means every other epoch, and so on.
    :param disable: Whether metrics collection is disabled.
    """
    if disable:
        return []

    if start >= epochs:
        return []

    # If start is less than zero, we don't collect during training
    if start < 0:
        return []

    if interval <= 0:
        raise ValueError(f'Invalid interval {interval}, must be non-zero')
    else:
        return list(range(start, epochs, interval))


def create_tlc_info_string_before_training(metrics_collection_epochs: list[int]) -> str:
    """ Creates a 3LC info string to print before training.

    :param metrics_collection_epochs: The epochs to collect metrics for.

    :returns: The 3LC info string.
    """
    if not metrics_collection_epochs:
        tlc_mc_string = 'Metrics collection disabled for this run.'
    else:
        plural_epochs = len(metrics_collection_epochs) > 1
        mc_epochs_str = ','.join(map(str, metrics_collection_epochs))
        tlc_mc_string = f'Collecting metrics for epoch{"s" if plural_epochs else ""} {mc_epochs_str}'

    return tlc_mc_string


def resolve_table_url(
    paths: list[str],
    dataset_name: str | None,
    project_name: str | None,
    prefix: str | None,
) -> tlc.Url:
    """Resolves a unique table url from the given parameters.

    :param paths: A list of paths to use to create the hash.
    :param dataset_name: The name of the dataset.
    :param project_name: The name of the project.
    :param prefix: The prefix to use for the table name.

    :returns: The unique table url.
    """
    _hash = get_hash(paths, dataset_name)
    table_name = f'{prefix or ""}{_hash}'
    table_url = tlc.Url.create_table_url(project_name=project_name, dataset_name=dataset_name, table_name=table_name)
    return table_url


def get_cache_file_name(table_url: tlc.Url) -> tlc.Url:
    """Returns the name of the cache file for the given table url.

    :param table_url: The table url.
    :returns: The cache file url.
    """
    return tlc.Url(f'./{_ROW_CACHE_FILE_NAME}.parquet')


def get_or_create_tlc_table(yolo_yaml_file: tlc.Url | str | None = None,
                            split: str | None = None,
                            revision_url: str | None = '',
                            root_url: tlc.Url | str | None = None) -> tlc.Table:
    """Get or create a 3LC Table for the given inputs.

    :param yolo_yaml_file: The path to the YOLO YAML file.
    :param split: The split to use.
    :param revision_url: The revision url to use.
    :param root_url: The root url to use.
    """

    if not yolo_yaml_file and not revision_url:
        raise ValueError('Either yolo_yaml_file or revision_url must be specified')

    if not split and not revision_url:
        raise ValueError('split must be specified if revision_url is not specified')

    # Ensure complete index before resolving any Tables
    tlc.TableIndexingTable.instance().ensure_fully_defined()

    if yolo_yaml_file:
        # Infer dataset and project names
        dataset_name_base = Path(yolo_yaml_file).stem
        dataset_name = dataset_name_base + '-' + split
        project_name = 'yolov8-' + dataset_name_base

        # if yolo_yaml_file:  # review this
        yolo_yaml_file = str(Path(yolo_yaml_file).resolve())  # Ensure absolute path for resolving Table Url

        # Resolve a unique Table name using dataset_name, yaml file path, yaml file size (and optionally root_url path and size), and split to create a deterministic url
        # The Table Url will be <3LC Table root> / <dataset_name> / <key><unique name>.json
        table_url_from_yaml = resolve_table_url([yolo_yaml_file, root_url if root_url else '', split],
                                                dataset_name,
                                                project_name,
                                                prefix='yolo_')

    # If revision_url is specified as an argument, use that Table
    if revision_url:
        try:
            table = tlc.Table.from_url(revision_url)
        except FileNotFoundError:
            raise ValueError(f'Could not find Table {revision_url} for {split} split')

        # If YAML file (--data argument) is also set, write appropriate log messages
        if yolo_yaml_file:
            try:
                root_table = tlc.Table.from_url(table_url_from_yaml)
                if not table.is_descendant_of(root_table):
                    LOGGER.info(
                        f"{TLC_COLORSTR}Revision URL is not a descendant of the Table corresponding to the YAML file's {split} split. Ignoring YAML file."
                    )
            except FileNotFoundError:
                LOGGER.warning(
                    f'{TLC_COLORSTR}Ignoring YAML file {yolo_yaml_file} because --tlc-{split}{"-" if split else ""}revision-url is set'
                )
        try:
            check_table_compatibility(table)
        except AssertionError as e:
            raise ValueError(f'Table {revision_url} is not compatible with YOLOv5') from e

        LOGGER.info(f'{TLC_COLORSTR}Using {split} revision {revision_url}')
    else:

        try:
            table = tlc.Table.from_url(table_url_from_yaml)
            initial_url = table.url
            table = table.latest()
            latest_url = table.url
            if initial_url != latest_url:
                LOGGER.info(f'{TLC_COLORSTR}Using latest version of {split} table: {latest_url.to_str()}')
            else:
                LOGGER.info(f'{TLC_COLORSTR}Using root {split} table: {initial_url.to_str()}')
        except FileNotFoundError:
            cache_url = get_cache_file_name(table_url_from_yaml)
            table = tlc.TableFromYolo(
                url=table_url_from_yaml,
                row_cache_url=cache_url,
                input_url=yolo_yaml_file,
                root_url=root_url,
                split=split,
            )
            table.get_rows_as_binary()  # Force immediate creation of row cache
            LOGGER.info(f'{TLC_COLORSTR}Using {split} table {table.url}')

        try:
            check_table_compatibility(table)
        except AssertionError as e:
            raise ValueError(f'Table {table_url_from_yaml.to_str()} is not compatible with YOLOv5') from e

    table.ensure_fully_defined()
    return table


def check_table_compatibility(table: tlc.Table) -> bool:
    """Check that the 3LC Table is compatible with YOLOv5.

    :param table: The 3LC Table to check.
    :returns: True if the Table is compatible, False otherwise.
    """

    row_schema = table.row_schema.values
    assert tlc.IMAGE in row_schema
    assert tlc.WIDTH in row_schema
    assert tlc.HEIGHT in row_schema
    assert tlc.BOUNDING_BOXES in row_schema
    assert tlc.BOUNDING_BOX_LIST in row_schema[tlc.BOUNDING_BOXES].values
    assert tlc.SAMPLE_WEIGHT in row_schema
    assert tlc.LABEL in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.X0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.Y0 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.X1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values
    assert tlc.Y1 in row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values

    X0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X0]
    Y0 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y0]
    X1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.X1]
    Y1 = row_schema[tlc.BOUNDING_BOXES].values[tlc.BOUNDING_BOX_LIST].values[tlc.Y1]

    assert X0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_X
    assert Y0.value.number_role == tlc.NUMBER_ROLE_BB_CENTER_Y
    assert X1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_X
    assert Y1.value.number_role == tlc.NUMBER_ROLE_BB_SIZE_Y

    return True


def _parse_boolean_env_var(name: str, default: str) -> bool:
    """Parse a boolean environment variable. Supported values:
    - true/false (case insensitive)
    - y/n (case insensitive)
    - 1/0
    - yes/no (case insensitive)

    :param name: The name of the environment variable.
    :param default: The value of the environment variable.
    :returns: The parsed boolean value.
    :raises: ValueError if the value is not a valid boolean.
    """
    value = os.getenv(name, default)
    if value.lower() in ('y', 'yes', '1', 'true'):
        return True
    elif value.lower() in ('n', 'no', '0', 'false'):
        return False
    else:
        raise ValueError(f'Invalid value {value} for environment variable {name}, '
                         'should be a boolean on the form y/n, yes/no, 1/0 or true/false.')


def _parse_env_var(name: str, default: str, var_type: type[T]) -> T:
    """Generic function to parse an environment variable and cast it to a specific type.

    :param name: The name of the environment variable.
    :param default: The default value of the environment variable if not set.
    :param var_type: The type to which the value should be cast.
    :returns: The parsed value of the specified type.
    :raises: ValueError if the value is not a valid type.
    """
    value = os.getenv(name, default)

    if var_type == bool:
        return _parse_boolean_env_var(name, default)
    elif var_type == list:
        return value.split(',')
    else:
        try:
            return var_type(value)
        except ValueError:
            raise ValueError(f'Invalid value {value} for environment variable {name}, should be a {var_type.__name__}.')


def _supported_env_vars_str(sep: str = '\n  - ') -> str:
    """ Print all supported environment variables.

    :param sep: The separator to use between each variable.
    :returns: A string sep-separated with all supported environment variables.

    """
    lines = [f'{var["name"]}: {var["description"]}. Default: {var["default"]}.' for var in TLC_SUPPORTED_ENV_VARS]
    return f'Supported environment variables:{sep}{sep.join(lines)}'


def _handle_unsupported_environment_variables():
    """ Handle unsupported environment variables by issuing warnings and suggestions.

    """
    env_var_names = os.environ.keys()
    tlc_env_var_names = [var['name'] for var in TLC_SUPPORTED_ENV_VARS]
    unsupported = [name for name in env_var_names if name.startswith('TLC_') and name not in tlc_env_var_names]

    # Output all environment variables if there are any unsupported ones
    if len(unsupported) > 1:
        LOGGER.warning(f'{TLC_COLORSTR}Found unsupported environment variables: '
                       f'{", ".join(unsupported)}.\n{_supported_env_vars_str()}')

    # If there is only one, look for the most similar one
    elif len(unsupported) == 1:
        closest_match = get_close_matches(unsupported[0], tlc_env_var_names, n=1, cutoff=0.4)
        if closest_match:
            LOGGER.warning(f'{TLC_COLORSTR}Found unsupported environment variable: {unsupported[0]}. '
                           f'Did you mean {closest_match[0]}?')
        else:
            LOGGER.warning(f'{TLC_COLORSTR}Found unsupported environment variable: {unsupported[0]}.'
                           f'\n{_supported_env_vars_str()}')


def parse_environment_variables() -> dict[str, int | float | bool | str | list[str]]:
    """ Parse and validate 3LC integration specific environment variables.

    :returns: A dictionary of parsed environment variables.
    :raises: ValueError if any environment variables are invalid.
    """
    config = {}

    # Warn about unsupported environment variables
    _handle_unsupported_environment_variables()

    # Read all supported environment variables
    for var in TLC_SUPPORTED_ENV_VARS:
        config[var['internal_name']] = _parse_env_var(var['name'], var['default'], var['type'])

    # Check for valid values
    if config['CONF_THRES'] < 0.0 or config['CONF_THRES'] > 1.0:
        raise ValueError(f'Invalid TLC_CONF_THRES={config["CONF_THRES"]}, must satisfy 0 <= TLC_CONF_THRES <= 1.')

    if config['MAX_DET'] < 1:
        raise ValueError(f'Invalid TLC_MAX_DET={config["MAX_DET"]}, must be > 0.')

    # Embeddings
    if config['IMAGE_EMBEDDINGS_DIM'] in (2, 3):
        umap_spec = importlib.util.find_spec('pacmap')
        if umap_spec is None:
            raise ValueError('Missing PaCMAP dependency, run `pip install pacmap` to enable embeddings collection.')
    elif config['IMAGE_EMBEDDINGS_DIM'] != 0:
        raise ValueError(f'Invalid TLC_IMAGE_EMBEDDINGS_DIM={config["IMAGE_EMBEDDINGS_DIM"]}, must be 0, 2 or 3.')

    # Collection interval
    if config['COLLECTION_EPOCH_INTERVAL'] < 1:
        raise ValueError(f'Invalid TLC_COLLECTION_EPOCH_INTERVAL={config["COLLECTION_EPOCH_INTERVAL"]}, must be >= 1.')

    return config


def write_3lc_yaml(data_file: str, tables: dict[str, tlc.Table]):
    """ Write a 3LC YAML file for the given tables.

    :param data_file: The path to the original YOLO YAML file.
    :param tables: The 3LC tables.
    """
    new_yaml_url = tlc.Url(data_file.replace('.yaml', '_3lc.yaml'))
    if new_yaml_url.exists():
        LOGGER.info(f'{TLC_COLORSTR}3LC YAML file already exists: {str(new_yaml_url)}. To use this file,'
                    f' add a 3LC prefix: --data 3LC://{str(new_yaml_url)}.')
        return

    # Common path for train, val, test tables:
    #                                        v           <--          <--          *
    # projects / yolov5-<dataset_name> / datasets / <dataset_name> / tables / <table_url> / files
    path = tables['train'].url.parent.parent.parent

    # Get relative paths for each table to write to YAML file
    split_paths = {split: str(tlc.Url.relative_from(tables[split].url, path).apply_aliases()) for split in tables}

    # Add :latest to each
    split_paths_latest = {split: f'{path}:latest' for split, path in split_paths.items()}

    # Create 3LC yaml file
    data_config = {'path': str(path), **split_paths_latest}
    new_yaml_url.write(yaml.dump(data_config, sort_keys=False, encoding='utf-8'))

    LOGGER.info(f'{TLC_COLORSTR}Created 3LC YAML file: {str(new_yaml_url)}. To use this file,'
                f' add a 3LC prefix: --data 3LC://{str(new_yaml_url)}.')


def tlc_check_dataset(data_file: str, get_splits: tuple | list = ('train', 'val')) -> dict[str, tlc.Table]:
    """ Parse the data file and get or create corresponding 3LC tables. If no 3LC YAML exists,
    create one.

    :param data_file: The path to the original YOLO YAML file.
    :param get_splits: The splits to get tables for.
    :returns: The 3LC tables.
    :raises: FileNotFoundError if the YAML file does not exist.
    """
    # Regular YAML file
    if not data_file.startswith(TLC_PREFIX):
        data_file = check_file(data_file)

        if not (data_file_url := tlc.Url(data_file)).exists():
            raise FileNotFoundError(f'Could not find YAML file {data_file_url}')

        data_file_content = yaml.safe_load(data_file_url.read())
        splits = [
            key for key in data_file_content if key not in ('path', 'names', 'download') and data_file_content[key]]

        # Create 3LC tables, get root table if already registered
        tables = {split: get_or_create_tlc_table(data_file, split=split) for split in splits}

        # Write all tables to the 3LC YAML file
        write_3lc_yaml(data_file, tables)

        # Remove any tables that are not in get_splits
        tables = {split: table for split, table in tables.items() if split in get_splits}

    # 3LC YAML file
    else:

        # Read the YAML file, removing the prefix
        if not (data_file_url := tlc.Url(data_file.replace(TLC_PREFIX, ''))).exists():
            raise FileNotFoundError(f'Could not find YAML file {data_file_url}')

        data_config = yaml.safe_load(data_file_url.read())

        path = data_config.get('path')
        splits = [key for key in data_config if key != 'path']

        tables = {}
        for split in splits:
            if split not in get_splits:
                continue

            split_path = data_config[split].split(':')[0]
            latest = data_config[split].endswith(':latest')

            if split_path.count(':') > 1:
                raise ValueError(f'Found more than one : in the split path {split_path} for split {split}')
            url = tlc.Url(path) / split_path if path else tlc.Url(split_path)

            table = get_or_create_tlc_table(split=split, revision_url=url)

            # Use latest revision if :latest is specified
            if latest:
                prev_url = url
                table = table.latest()
                if prev_url != table.url:
                    LOGGER.info(f'{TLC_COLORSTR}Using latest revision for {split} set: {table.url}.')

            tables[split] = table

    # Check that the tables have the same bounding box value maps
    value_maps = [table.get_value_map_for_column(tlc.BOUNDING_BOXES) for table in tables.values()]
    assert all(value_maps[0] == value_maps[i] for i in range(1, len(value_maps)))

    return tables


def tlc_task_map(task, key):
    if task != 'detect':
        LOGGER.info("3LC enabled, but currently only supports detect task. Defaulting to non-3LC mode.")
        return None

    from ultralytics.utils.tlc.detect.trainer import TLCDetectionTrainer
    from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator

    if key == "trainer":
        return TLCDetectionTrainer
    elif key == "validator":
        return TLCDetectionValidator
    else:
        return None


def training_phase_schema() -> tlc.Schema:
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
