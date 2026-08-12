---
title: cfg.__init__ API Reference
description: Explore the methods for managing and validating YOLO configurations in the Ultralytics configuration module. Enhance your YOLO experience.
keywords: Ultralytics, YOLO, configuration, cfg2dict, get_cfg, check_cfg, save_dir, deprecation, merge_args, yolo, settings, explorer
---

# Reference for `ultralytics/cfg/__init__.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/\_\_init\_\_.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`cfg2dict`](#ultralytics.cfg.__init__.cfg2dict)
        - [`get_cfg`](#ultralytics.cfg.__init__.get_cfg)
        - [`check_cfg`](#ultralytics.cfg.__init__.check_cfg)
        - [`get_save_dir`](#ultralytics.cfg.__init__.get_save_dir)
        - [`_handle_deprecation`](#ultralytics.cfg.__init__._handle_deprecation)
        - [`check_dict_alignment`](#ultralytics.cfg.__init__.check_dict_alignment)
        - [`merge_equals_args`](#ultralytics.cfg.__init__.merge_equals_args)
        - [`handle_yolo_login`](#ultralytics.cfg.__init__.handle_yolo_login)
        - [`handle_yolo_settings`](#ultralytics.cfg.__init__.handle_yolo_settings)
        - [`handle_yolo_solutions`](#ultralytics.cfg.__init__.handle_yolo_solutions)
        - [`parse_key_value_pair`](#ultralytics.cfg.__init__.parse_key_value_pair)
        - [`smart_value`](#ultralytics.cfg.__init__.smart_value)
        - [`entrypoint`](#ultralytics.cfg.__init__.entrypoint)
        - [`copy_default_cfg`](#ultralytics.cfg.__init__.copy_default_cfg)


## Function `ultralytics.cfg.cfg2dict` {#ultralytics.cfg.\_\_init\_\_.cfg2dict}

```python
def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict
```

Convert a configuration object to a dictionary.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str \| Path \| dict \| SimpleNamespace` | Configuration object to be converted. Can be a file path, a string, a dictionary, or a SimpleNamespace object. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Configuration object in dictionary format. |

**Examples**

Convert a YAML file path to a dictionary:

```python
>>> config_dict = cfg2dict("config.yaml")
```

Convert a SimpleNamespace to a dictionary:

```python
>>> from types import SimpleNamespace
>>> config_sn = SimpleNamespace(param1="value1", param2="value2")
>>> config_dict = cfg2dict(config_sn)
```

Pass through an already existing dictionary:

```python
>>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})
```

!!! note "Notes"

    - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
    - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
    - If cfg is already a dictionary, it's returned unchanged.

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L314-L345">View on GitHub</a>
```python
def cfg2dict(cfg: str | Path | dict | SimpleNamespace) -> dict:
    """Convert a configuration object to a dictionary.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted. Can be a file path, a string, a
            dictionary, or a SimpleNamespace object.

    Returns:
        (dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, STR_OR_PATH):
        cfg = YAML.load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.get_cfg` {#ultralytics.cfg.\_\_init\_\_.get\_cfg}

```python
def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_DICT, overrides: dict | None = None
) -> SimpleNamespace
```

Load and merge configuration data from a file or dictionary, with optional overrides.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `str \| Path \| dict \| SimpleNamespace` | Configuration data source. Can be a file path, dictionary, or SimpleNamespace object. | `DEFAULT_CFG_DICT` |
| `overrides` | `dict \| None` | Dictionary containing key-value pairs to override the base configuration. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `SimpleNamespace` | Namespace containing the merged configuration arguments. |

**Examples**

```python
>>> from ultralytics.cfg import get_cfg
>>> config = get_cfg()  # Load default configuration
>>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch": 16})
```

!!! note "Notes"

    - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
    - Special handling ensures alignment and correctness of the configuration, such as converting numeric
      `project` and `name` to strings and validating configuration keys and values.
    - The function performs type and value checks on the configuration data.

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L348-L392">View on GitHub</a>
```python
def get_cfg(
    cfg: str | Path | dict | SimpleNamespace = DEFAULT_CFG_DICT, overrides: dict | None = None
) -> SimpleNamespace:
    """Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        overrides (dict | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch": 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], FLOAT_OR_INT):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = Path(str(cfg.get("model") or "")).stem
        LOGGER.warning(f"'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.check_cfg` {#ultralytics.cfg.\_\_init\_\_.check\_cfg}

```python
def check_cfg(cfg: dict, hard: bool = True) -> None
```

Check configuration argument types and values for the Ultralytics library.

This function validates the types and values of configuration arguments, ensuring correctness and converting them if necessary. It checks for specific key types defined in global variables such as `CFG_FLOAT_KEYS`, `CFG_FRACTION_KEYS`, `CFG_INT_KEYS`, and `CFG_BOOL_KEYS`.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `cfg` | `dict` | Configuration dictionary to validate. | *required* |
| `hard` | `bool` | If True, raises exceptions for invalid types and values; if False, attempts to convert them. | `True` |

**Examples**

```python
>>> config = {
...     "epochs": 50,  # valid integer
...     "lr0": 0.01,  # valid float
...     "momentum": 0.937,  # valid float
...     "save": "true",  # invalid bool
... }
>>> check_cfg(config, hard=False)
>>> print(config)
{'epochs': 50, 'lr0': 0.01, 'momentum': 0.937, 'save': True}
```

!!! note "Notes"

    - The function modifies the input dictionary in-place.
    - None values are ignored as they may be from optional arguments.
    - Fraction keys use [0.0, 1.0], except dataset fraction, which uses (0.0, 1.0].

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L395-L499">View on GitHub</a>
```python
def check_cfg(cfg: dict, hard: bool = True) -> None:
    """Check configuration argument types and values for the Ultralytics library.

    This function validates the types and values of configuration arguments, ensuring correctness and converting them if
    necessary. It checks for specific key types defined in global variables such as `CFG_FLOAT_KEYS`,
    `CFG_FRACTION_KEYS`, `CFG_INT_KEYS`, and `CFG_BOOL_KEYS`.

    Args:
        cfg (dict): Configuration dictionary to validate.
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.

    Examples:
        >>> config = {
        ...     "epochs": 50,  # valid integer
        ...     "lr0": 0.01,  # valid float
        ...     "momentum": 0.937,  # valid float
        ...     "save": "true",  # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 0.937, 'save': True}

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys use [0.0, 1.0], except dataset fraction, which uses (0.0, 1.0].
    """
    typed_keys = CFG_FLOAT_KEYS | CFG_FRACTION_KEYS | CFG_INT_KEYS | CFG_BOOL_KEYS | CFG_STR_KEYS | {"scale", "compile"}
    for k, v in cfg.items():
        if v is None and DEFAULT_CFG_DICT.get(k) is not None and k in typed_keys and k != "auto_augment":
            raise TypeError(f"'{k}=None' is invalid. '{k}' must not be None.")
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, FLOAT_OR_INT):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k == "scale":
                if isinstance(v, (list, tuple)):
                    if len(v) != 2 or not all(isinstance(x, (int, float)) for x in v):
                        if hard:
                            raise TypeError(
                                f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int, float, or a tuple/list of two floats (i.e. '{k}=(0.5, 2.0)')"
                            )
                        continue
                    continue
                elif not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, FLOAT_OR_INT):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0) or (k == "fraction" and v == 0.0):
                    raise ValueError(f"'{k}={v}' is invalid. Use (0.0, 1.0] for fraction; [0.0, 1.0] otherwise.")
            elif k in CFG_INT_KEYS:
                if not isinstance(v, int):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be an int (i.e. '{k}=8')"
                        )
                    cfg[k] = v = int(v)
                if k in CFG_INT_MIN and v < CFG_INT_MIN[k]:
                    raise ValueError(f"'{k}={v}' is an invalid value. '{k}' must be >= {CFG_INT_MIN[k]}.")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)
            elif k in CFG_STR_KEYS and not isinstance(v, str):
                if hard:
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. '{k}' must be a str.")
                cfg[k] = str(v)
            elif k == "compile" and not isinstance(v, (bool, str)):  # False=off, True="default", or a mode string
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool or str (i.e. '{k}=True' or '{k}=max-autotune')"
                    )
                cfg[k] = bool(v)
            elif k == "quantize":  # canonicalize 8/16/32 or w-notation to a scheme (unset stays None for FP32)
                scheme = QUANTIZE_ALIASES.get(str(v).lower())
                if scheme is None:
                    if hard:
                        raise ValueError(
                            f"'{k}={v}' is invalid. Valid '{k}' values are {QUANTIZE_VALID_VALUES}. "
                            f"See {QUANTIZE_DOCS_URL}"
                        )
                else:
                    cfg[k] = scheme
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.get_save_dir` {#ultralytics.cfg.\_\_init\_\_.get\_save\_dir}

```python
def get_save_dir(args: SimpleNamespace, name: str | None = None) -> Path
```

Return the directory path for saving outputs, derived from arguments or default settings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `SimpleNamespace` | Namespace object containing configurations such as 'project', 'name', 'task', 'mode', and 'save_dir'. | *required* |
| `name` | `str \| None` | Optional name for the output directory. If not provided, it defaults to 'args.name' or the 'args.mode'. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `Path` | Directory path where outputs should be saved. |

**Examples**

```python
>>> from types import SimpleNamespace
>>> args = SimpleNamespace(project="my_project", name="exp", task="detect", mode="train", exist_ok=True)
>>> get_save_dir(args).parts[-3:]
('detect', 'my_project', 'exp')
```

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L502-L535">View on GitHub</a>
```python
def get_save_dir(args: SimpleNamespace, name: str | None = None) -> Path:
    """Return the directory path for saving outputs, derived from arguments or default settings.

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task', 'mode',
            and 'save_dir'.
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name' or the
            'args.mode'.

    Returns:
        (Path): Directory path where outputs should be saved.

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", name="exp", task="detect", mode="train", exist_ok=True)
        >>> get_save_dir(args).parts[-3:]
        ('detect', 'my_project', 'exp')
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or ""
        if not Path(project).is_absolute():
            base = ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else Path(SETTINGS["runs_dir"])
            worker = os.environ.get("PYTEST_XDIST_WORKER")
            if worker and TESTS_RUNNING:  # isolate parallel pytest-xdist workers
                base = base / worker
            project = base / args.task / project
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir).resolve()  # resolve to display full path in console
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg._handle_deprecation` {#ultralytics.cfg.\_\_init\_\_.\_handle\_deprecation}

```python
def _handle_deprecation(custom: dict) -> dict
```

Handle deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `custom` | `dict` | Configuration dictionary potentially containing deprecated keys. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Updated configuration dictionary with deprecated keys replaced. |

**Examples**

```python
>>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
>>> _handle_deprecation(custom_config)
{'show_boxes': True, 'show_labels': False, 'line_width': 2}
```

!!! note "Notes"

    This function modifies the input dictionary in-place, replacing deprecated keys with their current
    equivalents. It also handles value conversions where necessary, such as inverting boolean values for
    'hide_labels' and 'hide_conf'.

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L538-L588">View on GitHub</a>
```python
def _handle_deprecation(custom: dict) -> dict:
    """Handle deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

    Args:
        custom (dict): Configuration dictionary potentially containing deprecated keys.

    Returns:
        (dict): Updated configuration dictionary with deprecated keys replaced.

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        {'show_boxes': True, 'show_labels': False, 'line_width': 2}

    Notes:
        This function modifies the input dictionary in-place, replacing deprecated keys with their current
        equivalents. It also handles value conversions where necessary, such as inverting boolean values for
        'hide_labels' and 'hide_conf'.
    """
    deprecated_mappings = {
        "boxes": ("show_boxes", lambda v: v),
        "hide_labels": ("show_labels", lambda v: not bool(v)),
        "hide_conf": ("show_conf", lambda v: not bool(v)),
        "line_thickness": ("line_width", lambda v: v),
    }
    removed_keys = {"label_smoothing", "save_hybrid", "crop_fraction"}

    # Forward the deprecated precision flags onto the unified `quantize` scheme (int8 wins over half). The value is read
    # as a bool so quoted/string 'False' disables it while a bare CLI flag (empty string) enables it; an explicit false
    # flag maps to None to clear any inherited quantize. An explicit `quantize=` always wins over the legacy flags.
    int8 = custom.pop("int8", None)
    half = custom.pop("half", None)
    if (int8 is not None or half is not None) and "quantize" not in custom:
        int8_on = int8 is not None and str(int8).strip().lower() not in {"none", "false", "0"}
        half_on = half is not None and str(half).strip().lower() not in {"none", "false", "0"}
        custom["quantize"] = 8 if int8_on else 16 if half_on else None  # False/0 clears precision back to FP32
        deprecation_warn("int8" if int8 is not None else "half", "quantize")

    for old_key, (new_key, transform) in deprecated_mappings.items():
        if old_key not in custom:
            continue
        deprecation_warn(old_key, new_key)
        custom[new_key] = transform(custom.pop(old_key))

    for key in removed_keys:
        if key not in custom:
            continue
        deprecation_warn(key)
        custom.pop(key)

    return custom
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.check_dict_alignment` {#ultralytics.cfg.\_\_init\_\_.check\_dict\_alignment}

```python
def check_dict_alignment(
    base: dict, custom: dict, e: Exception | None = None, allowed_custom_keys: set | None = None
) -> None
```

Check alignment between custom and base configuration dictionaries, handling deprecated keys and providing error

messages for mismatched keys.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `base` | `dict` | The base configuration dictionary containing valid keys. | *required* |
| `custom` | `dict` | The custom configuration dictionary to be checked for alignment. | *required* |
| `e` | `Exception \| None` | Optional error instance passed by the calling function. | `None` |
| `allowed_custom_keys` | `set \| None` | Optional set of additional keys that are allowed in the custom dictionary. | `None` |

**Examples**

```python
>>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch": 16}
>>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch": 32}
>>> try:
...     check_dict_alignment(base_cfg, custom_cfg)
... except SyntaxError:
...     print("Mismatched keys found")
Mismatched keys found
```

!!! note "Notes"

    - Suggests corrections for mismatched keys based on similarity to valid keys.
    - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
    - Prints detailed error messages for each mismatched key to help users correct their configurations.

**Raises**

| Type | Description |
| --- | --- |
| `SyntaxError` | If mismatched keys are found between the custom and base dictionaries. |

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L591-L634">View on GitHub</a>
```python
def check_dict_alignment(
    base: dict, custom: dict, e: Exception | None = None, allowed_custom_keys: set | None = None
) -> None:
    """Check alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.

    Args:
        base (dict): The base configuration dictionary containing valid keys.
        custom (dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function.
        allowed_custom_keys (set | None): Optional set of additional keys that are allowed in the custom dictionary.

    Raises:
        SyntaxError: If mismatched keys are found between the custom and base dictionaries.

    Examples:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SyntaxError:
        ...     print("Mismatched keys found")
        Mismatched keys found

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (frozenset(x.keys()) for x in (base, custom))
    # Allow 'augmentations' as a valid custom parameter for custom Albumentations transforms
    if allowed_custom_keys is None:
        allowed_custom_keys = {"augmentations", "save_dir"}
    if mismatched := [k for k in custom_keys if k not in base_keys and k not in allowed_custom_keys]:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.merge_equals_args` {#ultralytics.cfg.\_\_init\_\_.merge\_equals\_args}

```python
def merge_equals_args(args: list[str]) -> list[str]
```

Merge arguments around isolated '=' in a list of strings and join fragments with brackets.

This function handles the following cases:
    1. ['arg', '=', 'val'] becomes ['arg=val']
    2. ['arg=', 'val'] becomes ['arg=val']
    3. ['arg', '=val'] becomes ['arg=val']
    4. Joins fragments with brackets, e.g., ['imgsz=[3,', '640,', '640]'] becomes ['imgsz=[3,640,640]']

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `list[str]` | A list of strings where each element represents an argument or fragment. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[str]` | A list of strings where the arguments around isolated '=' are merged and fragments with brackets are joined. |

**Examples**

```python
>>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
>>> merge_equals_args(args)
['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
```

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L637-L693">View on GitHub</a>
```python
def merge_equals_args(args: list[str]) -> list[str]:
    """Merge arguments around isolated '=' in a list of strings and join fragments with brackets.

    This function handles the following cases:
        1. ['arg', '=', 'val'] becomes ['arg=val']
        2. ['arg=', 'val'] becomes ['arg=val']
        3. ['arg', '=val'] becomes ['arg=val']
        4. Joins fragments with brackets, e.g., ['imgsz=[3,', '640,', '640]'] becomes ['imgsz=[3,640,640]']

    Args:
        args (list[str]): A list of strings where each element represents an argument or fragment.

    Returns:
        (list[str]): A list of strings where the arguments around isolated '=' are merged and fragments with brackets
            are joined.

    Examples:
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_equals_args(args)
        ['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle equals sign merging
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
            i += 1
            continue

        # Handle bracket joining
        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    # Append any remaining current string
    if current:
        new_args.append(current)

    return new_args
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.handle_yolo_login` {#ultralytics.cfg.\_\_init\_\_.handle\_yolo\_login}

```python
def handle_yolo_login(args: list[str]) -> None
```

Log in to Ultralytics Platform with an API key or remove the saved key.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `list[str]` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L696-L724">View on GitHub</a>
```python
def handle_yolo_login(args: list[str]) -> None:
    """Log in to Ultralytics Platform with an API key or remove the saved key."""
    if args[0] == "logout":
        SETTINGS["api_key"] = ""
        LOGGER.info("Logged out ✅. To log in again, use 'yolo login API_KEY'.")
        return

    api_key_url = f"{PLATFORM_URL}/settings?tab=api-keys"
    if len(args) < 2:
        LOGGER.info(f"Get an API key from {api_key_url} and then run 'yolo login API_KEY'.")
        return

    import requests  # scoped as slow import

    try:
        response = requests.get(
            f"{PLATFORM_URL}/api/settings",
            headers={"Authorization": f"Bearer {args[1]}"},
            timeout=30,
        )
        if response.status_code == 200:
            SETTINGS["api_key"] = args[1]
            LOGGER.info("New authentication successful ✅")
        elif response.status_code == 401:
            LOGGER.warning("Invalid API key")
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        LOGGER.warning(f"Authentication request failed, check your connection: {e}")
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.handle_yolo_settings` {#ultralytics.cfg.\_\_init\_\_.handle\_yolo\_settings}

```python
def handle_yolo_settings(args: list[str]) -> None
```

Handle YOLO settings command-line interface (CLI) commands.

This function processes YOLO settings CLI commands such as reset and updating individual settings. It should be called when executing a script with arguments related to YOLO settings management.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `list[str]` | A list of command line arguments for YOLO settings management. | *required* |

**Examples**

```python
>>> handle_yolo_settings(["reset"])  # Reset YOLO settings
>>> handle_yolo_settings(["runs_dir=path/to/dir"])  # Update a specific setting
```

!!! note "Notes"

    - If no arguments are provided, the function will display the current settings.
    - The 'reset' command will delete the existing settings file and create new default settings.
    - Other arguments are treated as key-value pairs to update specific settings.
    - The function will check for alignment between the provided settings and the existing ones.
    - After processing, the updated settings will be displayed.
    - For more information on handling YOLO settings, visit:
      https://docs.ultralytics.com/quickstart/#ultralytics-settings

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L727-L766">View on GitHub</a>
```python
def handle_yolo_settings(args: list[str]) -> None:
    """Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset and updating individual settings. It should be
    called when executing a script with arguments related to YOLO settings management.

    Args:
        args (list[str]): A list of command line arguments for YOLO settings management.

    Examples:
        >>> handle_yolo_settings(["reset"])  # Reset YOLO settings
        >>> handle_yolo_settings(["runs_dir=path/to/dir"])  # Update a specific setting

    Notes:
        - If no arguments are provided, the function will display the current settings.
        - The 'reset' command will delete the existing settings file and create new default settings.
        - Other arguments are treated as key-value pairs to update specific settings.
        - The function will check for alignment between the provided settings and the existing ones.
        - After processing, the updated settings will be displayed.
        - For more information on handling YOLO settings, visit:
          https://docs.ultralytics.com/quickstart/#ultralytics-settings
    """
    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"  # help URL
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_FILE.unlink()  # delete the settings file
                SETTINGS.reset()  # create new settings
                LOGGER.info("Settings reset successfully")  # inform the user that settings have been reset
            else:  # save a new setting
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)
                for k, v in new.items():
                    LOGGER.info(f"✅ Updated '{k}={v}'")

        LOGGER.info(SETTINGS)  # print the current settings
        LOGGER.info(f"💡 Learn more about Ultralytics Settings at {url}")
    except Exception as e:
        LOGGER.warning(f"settings error: '{e}'. Please see {url} for help.")
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.handle_yolo_solutions` {#ultralytics.cfg.\_\_init\_\_.handle\_yolo\_solutions}

```python
def handle_yolo_solutions(args: list[str]) -> None
```

Process YOLO solutions arguments and run the specified computer vision solutions pipeline.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `list[str]` | Command-line arguments for configuring and running the Ultralytics YOLO solutions. | *required* |

**Examples**

Run people counting solution with default settings:

```python
>>> handle_yolo_solutions(["count"])
```

Run analytics with custom configuration:

```python
>>> handle_yolo_solutions(["analytics", "conf=0.25", "source=path/to/video.mp4"])
```

Run inference with custom configuration, requires Streamlit version 1.29.0 or higher.

```python
>>> handle_yolo_solutions(["inference", "model=yolo26n.pt"])
```

!!! note "Notes"

    - Arguments can be provided in the format 'key=value' or as boolean flags
    - Available solutions are defined in SOLUTION_MAP with their respective classes and methods
    - If an invalid solution is provided, defaults to 'count' solution
    - Output videos are saved in 'runs/solutions/exp' directory
    - For 'analytics' solution, frame numbers are tracked for generating analytical graphs
    - Video processing can be interrupted by pressing 'q'
    - Processes video frames sequentially and saves output in .avi format
    - If no source is specified, downloads and uses a default sample video
    - The inference solution will be launched using the 'streamlit run' command.
    - The Streamlit app file is located in the Ultralytics package directory.

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L769-L876">View on GitHub</a>
```python
def handle_yolo_solutions(args: list[str]) -> None:
    """Process YOLO solutions arguments and run the specified computer vision solutions pipeline.

    Args:
        args (list[str]): Command-line arguments for configuring and running the Ultralytics YOLO solutions.

    Examples:
        Run people counting solution with default settings:
        >>> handle_yolo_solutions(["count"])

        Run analytics with custom configuration:
        >>> handle_yolo_solutions(["analytics", "conf=0.25", "source=path/to/video.mp4"])

        Run inference with custom configuration, requires Streamlit version 1.29.0 or higher.
        >>> handle_yolo_solutions(["inference", "model=yolo26n.pt"])

    Notes:
        - Arguments can be provided in the format 'key=value' or as boolean flags
        - Available solutions are defined in SOLUTION_MAP with their respective classes and methods
        - If an invalid solution is provided, defaults to 'count' solution
        - Output videos are saved in 'runs/solutions/exp' directory
        - For 'analytics' solution, frame numbers are tracked for generating analytical graphs
        - Video processing can be interrupted by pressing 'q'
        - Processes video frames sequentially and saves output in .avi format
        - If no source is specified, downloads and uses a default sample video
        - The inference solution will be launched using the 'streamlit run' command.
        - The Streamlit app file is located in the Ultralytics package directory.
    """
    from ultralytics.solutions.config import SolutionConfig

    full_args_dict = vars(SolutionConfig())  # arguments dictionary
    overrides = {}

    # check dictionary alignment
    for arg in merge_equals_args(args):
        arg = arg.lstrip("-").rstrip(",")
        if "=" in arg:
            try:
                k, v = parse_key_value_pair(arg)
                overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {arg: ""}, e)
        elif arg in full_args_dict and isinstance(full_args_dict.get(arg), bool):
            overrides[arg] = True
    check_dict_alignment(full_args_dict, overrides)  # dict alignment

    # Get solution name
    if not args:
        LOGGER.warning("No solution name provided. i.e `yolo solutions count`. Defaulting to 'count'.")
        args = ["count"]
    if args[0] == "help":
        LOGGER.info(SOLUTIONS_HELP_MSG)
        return  # Early return for 'help' case
    elif args[0] in SOLUTION_MAP:
        solution_name = args.pop(0)  # Extract the solution name directly
    else:
        LOGGER.warning(
            f"❌ '{args[0]}' is not a valid solution. 💡 Defaulting to 'count'.\n"
            f"🚀 Available solutions: {', '.join(list(SOLUTION_MAP.keys())[:-1])}\n"
        )
        solution_name = "count"  # Default for invalid solution

    if solution_name == "inference":
        checks.check_requirements("streamlit>=1.29.0")
        LOGGER.info("💡 Loading Ultralytics live inference app...")
        subprocess.run(
            [  # Run subprocess with Streamlit custom argument
                "streamlit",
                "run",
                str(ROOT / "solutions/streamlit_inference.py"),
                "--server.headless",
                "true",
                overrides.pop("model", "yolo26n.pt"),
            ],
            check=False,
        )
    else:
        import cv2  # Only needed for cap and vw functionality

        from ultralytics import solutions

        solution = getattr(solutions, SOLUTION_MAP[solution_name])(is_cli=True, **overrides)  # class i.e. ObjectCounter

        cap = cv2.VideoCapture(solution.CFG["source"])  # read the video file
        if solution_name != "crop":
            # extract width, height and fps of the video file, create save directory and initialize video writer
            w, h, fps = (
                int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
            )
            if solution_name == "analytics":  # analytical graphs follow fixed shape for output i.e w=1280, h=720
                w, h = 1280, 720
            save_dir = get_save_dir(SimpleNamespace(task="solutions", name="exp", exist_ok=False, project=None))
            save_dir.mkdir(parents=True, exist_ok=True)  # create the output directory i.e. runs/solutions/exp
            vw = cv2.VideoWriter(str(save_dir / f"{solution_name}.avi"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        try:  # Process video frames
            f_n = 0  # frame number, required for analytical graphs
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                results = solution(frame, f_n := f_n + 1) if solution_name == "analytics" else solution(frame)
                if solution_name != "crop":
                    vw.write(results.plot_im)
                if solution.CFG["show"] and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.parse_key_value_pair` {#ultralytics.cfg.\_\_init\_\_.parse\_key\_value\_pair}

```python
def parse_key_value_pair(pair: str = "key=value") -> tuple
```

Parse a key-value pair string into separate key and value components.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `pair` | `str` | A string containing a key-value pair in the format "key=value". | `"key=value"` |

**Returns**

| Type | Description |
| --- | --- |
| `key (str)` | The parsed key. |
| `value (str)` | The parsed value. |

**Examples**

```python
>>> key, value = parse_key_value_pair("model=yolo26n.pt")
>>> print(f"Key: {key}, Value: {value}")
Key: model, Value: yolo26n.pt
```

```python
>>> key, value = parse_key_value_pair("epochs=100")
>>> print(f"Key: {key}, Value: {value}")
Key: epochs, Value: 100
```

!!! note "Notes"

    - The function splits the input string on the first '=' character.
    - Leading and trailing whitespace is removed from both key and value.
    - An assertion error is raised if the value is empty after stripping.

**Raises**

| Type | Description |
| --- | --- |
| `AssertionError` | If the value is missing or empty. |

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L879-L909">View on GitHub</a>
```python
def parse_key_value_pair(pair: str = "key=value") -> tuple:
    """Parse a key-value pair string into separate key and value components.

    Args:
        pair (str): A string containing a key-value pair in the format "key=value".

    Returns:
        key (str): The parsed key.
        value (str): The parsed value.

    Raises:
        AssertionError: If the value is missing or empty.

    Examples:
        >>> key, value = parse_key_value_pair("model=yolo26n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo26n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    Notes:
        - The function splits the input string on the first '=' character.
        - Leading and trailing whitespace is removed from both key and value.
        - An assertion error is raised if the value is empty after stripping.
    """
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.smart_value` {#ultralytics.cfg.\_\_init\_\_.smart\_value}

```python
def smart_value(v: str) -> Any
```

Convert a string representation of a value to its appropriate Python type.

This function attempts to convert a given string into a Python object of the most appropriate type. It handles conversions to None, bool, int, float, and other types that can be evaluated safely.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `v` | `str` | The string representation of the value to be converted. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `Any` | The converted value. The type can be None, bool, int, float, or the original string if no conversion is applicable. |

**Examples**

```python
>>> smart_value("42")
42
>>> smart_value("3.14")
3.14
>>> smart_value("True")
True
>>> print(smart_value("None"))
None
>>> smart_value("some_string")
'some_string'
```

!!! note "Notes"

    - The function uses a case-insensitive comparison for boolean and None values.
    - For other types, it attempts to use Python's ast.literal_eval() function for safe evaluation.
    - If no conversion is possible, the original string is returned.

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L912-L958">View on GitHub</a>
```python
def smart_value(v: str) -> Any:
    """Convert a string representation of a value to its appropriate Python type.

    This function attempts to convert a given string into a Python object of the most appropriate type. It handles
    conversions to None, bool, int, float, and other types that can be evaluated safely.

    Args:
        v (str): The string representation of the value to be converted.

    Returns:
        (Any): The converted value. The type can be None, bool, int, float, or the original string if no conversion is
            applicable.

    Examples:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> print(smart_value("None"))
        None
        >>> smart_value("some_string")
        'some_string'

    Notes:
        - The function uses a case-insensitive comparison for boolean and None values.
        - For other types, it attempts to use Python's ast.literal_eval() function for safe evaluation.
        - If no conversion is possible, the original string is returned.
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return ast.literal_eval(v)
        except Exception:
            name, _, attr = v.rpartition(".")
            if (module := sys.modules.get(name)) and attr.isupper():
                value = getattr(module, attr, None)
                if isinstance(value, (int, float)):
                    return value
            return v
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.entrypoint` {#ultralytics.cfg.\_\_init\_\_.entrypoint}

```python
def entrypoint(debug: str = "") -> None
```

Ultralytics entrypoint function for parsing and executing command-line arguments.

This function serves as the main entry point for the Ultralytics CLI, parsing command-line arguments and executing the corresponding tasks such as training, validation, prediction, exporting models, and more.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `debug` | `str` | Space-separated string of command-line arguments for debugging purposes. | `""` |

**Examples**

Train a detection model for 10 epochs with an initial learning_rate of 0.01:

```python
>>> entrypoint("train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01")
```

Predict a YouTube video using a pretrained segmentation model at image size 320:

```python
>>> entrypoint("predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")
```

Validate a pretrained detection model at batch-size 1 and image size 640:

```python
>>> entrypoint("val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640")
```

!!! note "Notes"

    - If no arguments are passed, the function will display the usage help message.
    - For a list of all available commands and their arguments, see the provided help messages and the
      Ultralytics documentation at https://docs.ultralytics.com.

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L961-L1130">View on GitHub</a>
```python
def entrypoint(debug: str = "") -> None:
    """Ultralytics entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the Ultralytics CLI, parsing command-line arguments and executing
    the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str): Space-separated string of command-line arguments for debugging purposes.

    Examples:
        Train a detection model for 10 epochs with an initial learning_rate of 0.01:
        >>> entrypoint("train data=coco8.yaml model=yolo26n.pt epochs=10 lr0=0.01")

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        >>> entrypoint("predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        Validate a pretrained detection model at batch-size 1 and image size 640:
        >>> entrypoint("val model=yolo26n.pt data=coco8.yaml batch=1 imgsz=640")

    Notes:
        - If no arguments are passed, the function will display the usage help message.
        - For a list of all available commands and their arguments, see the provided help messages and the
          Ultralytics documentation at https://docs.ultralytics.com.
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: YAML.print(DEFAULT_CFG_PATH),
        "login": lambda: handle_yolo_login(args),
        "logout": lambda: handle_yolo_login(args),
        "copy-cfg": copy_default_cfg,
        "solutions": lambda: handle_yolo_solutions(args[1:]),
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # custom.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in YAML.load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in {"half", "int8"}:
            overrides[a] = True  # deprecated bare precision flags, forwarded to quantize by _handle_deprecation
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"'mode' argument is missing. Valid modes are {list(MODES)}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {list(MODES)}.\n{CLI_HELP_MSG}")

    # Task
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            if task == "track":
                LOGGER.warning(
                    f"invalid 'task=track', setting 'task=detect' and 'mode=track'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}."
                )
                task, mode = "detect", "track"
            else:
                raise ValueError(f"Invalid 'task={task}'. Valid tasks are {list(TASKS)}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # Model
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolo26n.pt"
        LOGGER.warning(f"'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # guess architecture
        from ultralytics import RTDETR

        model = RTDETR(model)  # no task argument
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam_" in stem or "sam2_" in stem or "sam2.1_" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
        if "yoloe" in stem or "world" in stem:
            cls_list = overrides.pop("classes", DEFAULT_CFG.classes)
            if cls_list is not None and isinstance(cls_list, str):
                model.set_classes([c.strip() for c in cls_list.split(",")])  # "person, bus" -> ['person', 'bus']
    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(
                f"conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # Mode
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = (
            "https://ultralytics.com/images/boats.jpg" if task == "obb" else DEFAULT_CFG.source or ASSETS
        )
        LOGGER.warning(f"'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"'format' argument is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")

    # Recommend VS Code extension
    if IS_VSCODE and SETTINGS.get("vscode_msg", True):
        LOGGER.info(vscode_msg())
```
</details>


<br><br><hr><br>

## Function `ultralytics.cfg.copy_default_cfg` {#ultralytics.cfg.\_\_init\_\_.copy\_default\_cfg}

```python
def copy_default_cfg() -> None
```

Copy the default configuration file and create a new one with '_copy' appended to its name.

This function duplicates the existing default configuration file (DEFAULT_CFG_PATH) and saves it with '_copy' appended to its name in the current working directory. It provides a convenient way to create a custom configuration file based on the default settings.

**Examples**

```python
>>> copy_default_cfg()
```

!!! note "Notes"

    - The new configuration file is created in the current working directory.
    - After copying, the function prints a message with the new file's location and an example
      YOLO command demonstrating how to use the new configuration file.
    - This function is useful for users who want to modify the default configuration without
      altering the original file.

<details>
<summary>Source code in <code>ultralytics/cfg/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/__init__.py#L1134-L1156">View on GitHub</a>
```python
def copy_default_cfg() -> None:
    """Copy the default configuration file and create a new one with '_copy' appended to its name.

    This function duplicates the existing default configuration file (DEFAULT_CFG_PATH) and saves it with '_copy'
    appended to its name in the current working directory. It provides a convenient way to create a custom configuration
    file based on the default settings.

    Examples:
        >>> copy_default_cfg()

    Notes:
        - The new configuration file is created in the current working directory.
        - After copying, the function prints a message with the new file's location and an example
          YOLO command demonstrating how to use the new configuration file.
        - This function is useful for users who want to modify the default configuration without
          altering the original file.
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )
```
</details>

<br><br>
