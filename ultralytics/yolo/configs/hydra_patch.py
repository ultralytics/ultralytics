# Ultralytics YOLO 🚀, GPL-3.0 license

import sys
from difflib import get_close_matches
from textwrap import dedent

import hydra
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf, open_dict  # noqa
from omegaconf.errors import ConfigAttributeError, ConfigKeyError, OmegaConfBaseException  # noqa

from ultralytics.yolo.utils import LOGGER, colorstr


def override_config(overrides, cfg):
    override_keys = [override.key_or_group for override in overrides]
    check_config_mismatch(override_keys, cfg.keys())
    for override in overrides:
        if override.package is not None:
            raise ConfigCompositionException(f"Override {override.input_line} looks like a config group"
                                             f" override, but config group '{override.key_or_group}' does not exist.")

        key = override.key_or_group
        value = override.value()
        try:
            if override.is_delete():
                config_val = OmegaConf.select(cfg, key, throw_on_missing=False)
                if config_val is None:
                    raise ConfigCompositionException(f"Could not delete from config. '{override.key_or_group}'"
                                                     " does not exist.")
                elif value is not None and value != config_val:
                    raise ConfigCompositionException("Could not delete from config. The value of"
                                                     f" '{override.key_or_group}' is {config_val} and not"
                                                     f" {value}.")

                last_dot = key.rfind(".")
                with open_dict(cfg):
                    if last_dot == -1:
                        del cfg[key]
                    else:
                        node = OmegaConf.select(cfg, key[:last_dot])
                        del node[key[last_dot + 1:]]

            elif override.is_add():
                if OmegaConf.select(cfg, key, throw_on_missing=False) is None or isinstance(value, (dict, list)):
                    OmegaConf.update(cfg, key, value, merge=True, force_add=True)
                else:
                    assert override.input_line is not None
                    raise ConfigCompositionException(
                        dedent(f"""\
                    Could not append to config. An item is already at '{override.key_or_group}'.
                    Either remove + prefix: '{override.input_line[1:]}'
                    Or add a second + to add or override '{override.key_or_group}': '+{override.input_line}'
                    """))
            elif override.is_force_add():
                OmegaConf.update(cfg, key, value, merge=True, force_add=True)
            else:
                try:
                    OmegaConf.update(cfg, key, value, merge=True)
                except (ConfigAttributeError, ConfigKeyError) as ex:
                    raise ConfigCompositionException(f"Could not override '{override.key_or_group}'."
                                                     f"\nTo append to your config use +{override.input_line}") from ex
        except OmegaConfBaseException as ex:
            raise ConfigCompositionException(f"Error merging override {override.input_line}").with_traceback(
                sys.exc_info()[2]) from ex


def check_config_mismatch(overrides, cfg):
    mismatched = [option for option in overrides if option not in cfg and 'hydra.' not in option]

    for option in mismatched:
        LOGGER.info(f"{colorstr(option)} is not a valid key. Similar keys: {get_close_matches(option, cfg, 3, 0.6)}")
    if mismatched:
        exit()


hydra._internal.config_loader_impl.ConfigLoaderImpl._apply_overrides_to_config = override_config
