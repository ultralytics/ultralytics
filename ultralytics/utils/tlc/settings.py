# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field, fields
from difflib import get_close_matches
from typing import Any

from ultralytics.utils import LOGGER
from ultralytics.utils.tlc.constants import TLC_COLORSTR


@dataclass
class Settings:
    """ Settings dataclass for the 3LC integration. Defines and handles user settings for the 3LC integration.

    Supports parsing values from environment variables.
    """
    conf_thres: float = field(default=0.1, metadata={'description': 'Confidence threshold for detections'})
    max_det: int = field(default=300, metadata={'description': 'Maximum number of detections collected per image'})
    project_name: str | None = field(default=None, metadata={'description': 'The name of the 3LC project'})
    run_name: str | None = field(default=None, metadata={'description': 'The name of the 3LC run'})
    run_description: str | None = field(default=None, metadata={'description': 'The description of the 3LC run'})
    image_embeddings_dim: int = field(
        default=0,
        metadata={
            'description':
            'Image embeddings dimension. 0 means no embeddings, 2 means 2D embeddings, 3 means 3D embeddings'})
    image_embeddings_reducer: str = field(
        default='pacmap',
        metadata={
            'description':
            'Reduction algorithm for image embeddings. Options: pacmap and umap. Only used if IMAGE_EMBEDDINGS_DIM > 0'
        })
    sampling_weights: bool = field(default=False, metadata={'description': 'Whether to use 3LC Sampling Weights'})
    exclude_zero_weight_training: bool = field(
        default=False, metadata={"description": "Whether to exclude zero-weighted samples in training."}
    )
    exclude_zero_weight_collection: bool = field(
        default=False, metadata={"description": "Whether to exclude zero-weighted samples in metrics collection."}
    )
    # TODO: Restore when loss is supported.
    # collect_loss: bool = field(default=False,
    #                            metadata={'description': 'Whether to collect loss values during training'})
    collection_val_only: bool = field(default=False,
                                      metadata={'description': 'Whether to collect metrics only on the val set'})
    collection_disable: bool = field(default=False,
                                     metadata={'description': 'Whether to disable 3LC metrics collection entirely'})
    collection_epoch_start: int | None = field(default=None,
                                               metadata={'description': 'Start epoch for collection during training (1 is after the first epoch)'})
    collection_epoch_interval: int = field(default=1,
                                           metadata={
                                               'description':
                                               'Epoch interval for collection. Only used if a starting epoch is set'})


    @classmethod
    def from_env(cls) -> Settings:
        """ Create a Settings instance from environment variables.

        :returns: A Settings instance with values parsed from environment variables.
        """
        cls._handle_unsupported_env_vars()  # Warn about unsupported environment variables

        kwargs = {}

        for _field in fields(cls):
            env_var = cls._field_to_env_var(_field)
            env_value = os.getenv(env_var, None)
            if env_value is not None:
                value = Settings._parse_env_var(env_var, env_value, _field.type)
                kwargs[_field.name] = value

        instance = cls(**kwargs)

        instance._from_env = True  # Mark the instance as created from environment variables

        return instance

    def __post_init__(self) -> None:
        # Mark as not created from environment variables
        if not hasattr(self, '_from_env'):
            self._from_env = False

    def verify(self, training: bool = True) -> None:
        """ Verify that the settings are valid.

        :param training: Whether the settings are for training or validation.

        :raises: AssertionError if the settings are invalid.
        """
        # Checks
        assert 0.0 <= self.conf_thres <= 1.0, f'Confidence threshold {self.conf_thres} is not in [0, 1].'
        assert self.max_det > 0, f'Maximum number of detections {self.max_det} is not positive.'
        assert self.image_embeddings_dim in (0, 2,
                                             3), f'Invalid image embeddings dimension {self.image_embeddings_dim}.'
        assert self.image_embeddings_reducer in (
            'pacmap', 'umap'), f'Invalid image embeddings reducer {self.image_embeddings_reducer}.'
        if self.image_embeddings_dim > 0:
            self._check_reducer_available()

        # Train / collect specific settings
        self._verify_training() if training else self._verify_collection()

    def get_metrics_collection_epochs(self, epochs: int) -> list[int]:
        if self.collection_disable:
            return []

        if self.collection_epoch_start is None:
            return []

        if self.collection_epoch_start >= epochs:
            return []

        # If start is less than one, we don't collect during training
        if self.collection_epoch_start < 1:
            return []

        if self.collection_epoch_interval <= 0:
            raise ValueError(f'Invalid interval {self.collection_epoch_interval}, must be non-zero')
        else:
            return list(range(self.collection_epoch_start, epochs, self.collection_epoch_interval))

    def _verify_training(self) -> None:
        """ Verify that the settings are valid for training.

        :param opt: The argparse namespace containing YOLOv8 settings.

        :raises: AssertionError if the settings are invalid.
        """
        # Can't collect things when collection is disabled
        cases = [
            (self.collection_val_only, 'collect only on val set'),
            # (self.collect_loss, 'collect loss values'), TODO: Restore when loss is supported.
            (self.image_embeddings_dim > 0, 'collect image embeddings'),
            (self.collection_epoch_start, 'collect metrics during training'), ]

        for setting, description in cases:
            assert not (self.collection_disable and setting), f'Cannot disable collection and {description}.'

        # Collection epoch settings
        assert self.collection_epoch_start is None or self.collection_epoch_start >= 0, (
            f'Invalid collection start epoch {self.collection_epoch_start}.')
        assert self.collection_epoch_interval > 0, (
            f'Invalid collection epoch interval {self.collection_epoch_interval}.')

    def _verify_collection(self) -> None:
        """ Verify that the settings are valid for metrics collection only (val.py --task collect).

        :raises: AssertionError if the settings are invalid.
        """
        pass

    def _check_reducer_available(self) -> None:
        """ Check that the selected reducer is available.

        :raises: ValueError if the selected reducer is not available.
        """
        reducer_spec = importlib.util.find_spec(self.image_embeddings_reducer)
        if reducer_spec is None:
            reducer_to_package = {'pacmap': 'pacmap', 'umap': 'umap-learn'}
            package = reducer_to_package[self.image_embeddings_reducer]
            raise ValueError(
                f"Embeddings collection enabled, but missing {self.image_embeddings_reducer} dependency. "
                f"Run `pip install {package}` to enable embeddings collection."
            )

    @staticmethod
    def _field_to_env_var(_field: field) -> None:
        """Return the environment variable name for a given field.

        :param _field: The field to get the environment variable for.
        :returns: The environment variable name.
        """
        return f'TLC_{_field.name.upper()}'

    def _handle_unsupported_env_vars(self) -> None:
        """Handle environment variables starting with TLC which are not supported.

        Appropriate warnings are logged when unsupported environment variables are encountered.
        """
        supported_env_vars = [self._field_to_env_var(_field) for _field in fields(Settings)]
        unsupported_env_vars = [var for var in os.environ if var.startswith('TLC_') and var not in supported_env_vars]

        # Output all environment variables if there are any unsupported ones
        if len(unsupported_env_vars) > 1:
            LOGGER.warning(f'{TLC_COLORSTR}Found unsupported environment variables: '
                           f'{", ".join(unsupported_env_vars)}.\n{self._supported_env_vars_str()}')

        # If there is only one, look for the most similar one
        elif len(unsupported_env_vars) == 1:
            closest_match = get_close_matches(unsupported_env_vars[0], supported_env_vars, n=1, cutoff=0.4)
            if closest_match:
                LOGGER.warning(f'{TLC_COLORSTR}Found unsupported environment variable: {unsupported_env_vars[0]}. '
                               f'Did you mean {closest_match[0]}?')
            else:
                LOGGER.warning(f'{TLC_COLORSTR}Found unsupported environment variable: {unsupported_env_vars[0]}.'
                               f'\n{self._supported_env_vars_str()}')

    def _supported_env_vars_str(self, sep: str = '\n  - ') -> str:
        """ Print all supported environment variables.

        :param sep: The separator to use between each variable.
        :returns: A sep-separated string with all supported environment variables.

        """
        default_settings_instance = Settings()  # Create an instance to get the default values

        # Display defaults differently for environment variables as they are provided differently
        if self._from_env:
            formatter = lambda x: x if not isinstance(x, list) else ','.join(x)
        else:
            formatter = lambda x: x

        field_info_list = [{
            'name': self._field_to_env_var(_field),
            'description': _field.metadata['description'],
            'default': formatter(getattr(default_settings_instance, _field.name))} for _field in fields(Settings)]

        # Display each line as TLC_<FIELD_NAME>: <DESCRIPTION>. Default: '<DEFAULT>'
        lines = [
            f"{field_info['name']}: {field_info['description']}. Default: '{field_info['default']}'."
            for field_info in field_info_list]
        return f'Supported environment variables:{sep}{sep.join(lines)}'

    @staticmethod
    def _parse_env_var(name: str, value: Any, var_type: str) -> Any:
        """ Parse an environment variable.

        :param name: The name of the environment variable.
        :param value: The value of the environment variable.
        :param var_type: The expected type of the environment variable as defined in the dataclass.
        """
        if var_type == 'bool':
            return Settings._parse_boolean_env_var(name, value)
        elif var_type == 'list':
            return value.split(',')
        elif var_type == 'int':
            return int(value)
        elif var_type == 'float':
            return float(value)
        elif var_type == 'str':
            return value
        else:
            raise ValueError(f'Unsupported type {var_type} for environment variable {name}.')

    @staticmethod
    def _parse_boolean_env_var(name: str, value: Any) -> bool:
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
        if value.lower() in ('y', 'yes', '1', 'true'):
            return True
        elif value.lower() in ('n', 'no', '0', 'false'):
            return False
        else:
            raise ValueError(f'Invalid value {value} for environment variable {name}, '
                             'should be a boolean on the form y/n, yes/no, 1/0 or true/false.')
