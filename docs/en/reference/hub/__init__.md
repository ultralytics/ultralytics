---
description: Explore Ultralytics Platform API functions for login, logout, model reset, export, and dataset checks. Enhance your YOLO workflows with these essential utilities.
keywords: Ultralytics Platform API, login, logout, reset model, export model, check dataset, YOLO, machine learning
---

# Reference for `ultralytics/hub/__init__.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/\_\_init\_\_.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`login`](#ultralytics.hub.__init__.login)
        - [`logout`](#ultralytics.hub.__init__.logout)
        - [`reset_model`](#ultralytics.hub.__init__.reset_model)
        - [`export_fmts_hub`](#ultralytics.hub.__init__.export_fmts_hub)
        - [`export_model`](#ultralytics.hub.__init__.export_model)
        - [`get_export`](#ultralytics.hub.__init__.get_export)
        - [`check_dataset`](#ultralytics.hub.__init__.check_dataset)


## Function `ultralytics.hub.login` {#ultralytics.hub.\_\_init\_\_.login}

```python
def login(api_key: str | None = None, save: bool = True) -> bool
```

Log in to the Ultralytics HUB API using the provided API key.

The session is not stored; a new session is created when needed using the saved SETTINGS or the HUB_API_KEY environment variable if successfully authenticated.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `api_key` | `str, optional` | API key to use for authentication. If not provided, it will be retrieved from SETTINGS<br>    or HUB_API_KEY environment variable. | `None` |
| `save` | `bool, optional` | Whether to save the API key to SETTINGS if authentication is successful. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `bool` | True if authentication is successful, False otherwise. |

<details>
<summary>Source code in <code>ultralytics/hub/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py#L25-L65"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def login(api_key: str | None = None, save: bool = True) -> bool:
    """Log in to the Ultralytics HUB API using the provided API key.

    The session is not stored; a new session is created when needed using the saved SETTINGS or the HUB_API_KEY
    environment variable if successfully authenticated.

    Args:
        api_key (str, optional): API key to use for authentication. If not provided, it will be retrieved from SETTINGS
            or HUB_API_KEY environment variable.
        save (bool, optional): Whether to save the API key to SETTINGS if authentication is successful.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    checks.check_requirements("hub-sdk>=0.0.12")
    from hub_sdk import HUBClient

    api_key_url = f"{HUB_WEB_ROOT}/settings?tab=api+keys"  # set the redirect URL
    saved_key = SETTINGS.get("api_key")
    active_key = api_key or saved_key
    credentials = {"api_key": active_key} if active_key and active_key != "" else None  # set credentials

    client = HUBClient(credentials)  # initialize HUBClient

    if client.authenticated:
        # Successfully authenticated with HUB

        if save and client.api_key != saved_key:
            SETTINGS.update({"api_key": client.api_key})  # update settings with valid API key

        # Set message based on whether key was provided or retrieved from settings
        log_message = (
            "New authentication successful ✅" if client.api_key == api_key or not credentials else "Authenticated ✅"
        )
        LOGGER.info(f"{PREFIX}{log_message}")

        return True
    else:
        # Failed to authenticate with HUB
        LOGGER.info(f"{PREFIX}Get API key from {api_key_url} and then run 'yolo login API_KEY'")
        return False
```
</details>


<br><br><hr><br>

## Function `ultralytics.hub.logout` {#ultralytics.hub.\_\_init\_\_.logout}

```python
def logout()
```

Log out of Ultralytics HUB by removing the API key from the settings file.

<details>
<summary>Source code in <code>ultralytics/hub/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py#L68-L71"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def logout():
    """Log out of Ultralytics HUB by removing the API key from the settings file."""
    SETTINGS["api_key"] = ""
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo login'.")
```
</details>


<br><br><hr><br>

## Function `ultralytics.hub.reset_model` {#ultralytics.hub.\_\_init\_\_.reset\_model}

```python
def reset_model(model_id: str = "")
```

Reset a trained model to an untrained state.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_id` | `str` |  | `""` |

<details>
<summary>Source code in <code>ultralytics/hub/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py#L74-L82"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def reset_model(model_id: str = ""):
    """Reset a trained model to an untrained state."""
    import requests  # scoped as slow import

    r = requests.post(f"{HUB_API_ROOT}/model-reset", json={"modelId": model_id}, headers={"x-api-key": Auth().api_key})
    if r.status_code == 200:
        LOGGER.info(f"{PREFIX}Model reset successfully")
        return
    LOGGER.warning(f"{PREFIX}Model reset failure {r.status_code} {r.reason}")
```
</details>


<br><br><hr><br>

## Function `ultralytics.hub.export_fmts_hub` {#ultralytics.hub.\_\_init\_\_.export\_fmts\_hub}

```python
def export_fmts_hub()
```

Return a list of HUB-supported export formats.

<details>
<summary>Source code in <code>ultralytics/hub/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py#L85-L89"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def export_fmts_hub():
    """Return a list of HUB-supported export formats."""
    from ultralytics.engine.exporter import export_formats

    return [*list(export_formats()["Argument"][1:]), "ultralytics_tflite", "ultralytics_coreml"]
```
</details>


<br><br><hr><br>

## Function `ultralytics.hub.export_model` {#ultralytics.hub.\_\_init\_\_.export\_model}

```python
def export_model(model_id: str = "", format: str = "torchscript")
```

Export a model to a specified format for deployment via the Ultralytics HUB API.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_id` | `str` | The ID of the model to export. An empty string will use the default model. | `""` |
| `format` | `str` | The format to export the model to. Must be one of the supported formats returned by<br>    export_fmts_hub(). | `"torchscript"` |

**Examples**

```python
>>> from ultralytics import hub
>>> hub.export_model(model_id="your_model_id", format="torchscript")
```

**Raises**

| Type | Description |
| --- | --- |
| `AssertionError` | If the specified format is not supported or if the export request fails. |

<details>
<summary>Source code in <code>ultralytics/hub/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py#L92-L114"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def export_model(model_id: str = "", format: str = "torchscript"):
    """Export a model to a specified format for deployment via the Ultralytics HUB API.

    Args:
        model_id (str): The ID of the model to export. An empty string will use the default model.
        format (str): The format to export the model to. Must be one of the supported formats returned by
            export_fmts_hub().

    Raises:
        AssertionError: If the specified format is not supported or if the export request fails.

    Examples:
        >>> from ultralytics import hub
        >>> hub.export_model(model_id="your_model_id", format="torchscript")
    """
    import requests  # scoped as slow import

    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export", json={"format": format}, headers={"x-api-key": Auth().api_key}
    )
    assert r.status_code == 200, f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")
```
</details>


<br><br><hr><br>

## Function `ultralytics.hub.get_export` {#ultralytics.hub.\_\_init\_\_.get\_export}

```python
def get_export(model_id: str = "", format: str = "torchscript")
```

Retrieve an exported model in the specified format from Ultralytics HUB using the model ID.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_id` | `str` | The ID of the model to retrieve from Ultralytics HUB. | `""` |
| `format` | `str` | The export format to retrieve. Must be one of the supported formats returned by export_fmts_hub(). | `"torchscript"` |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | JSON response containing the exported model information. |

**Examples**

```python
>>> from ultralytics import hub
>>> result = hub.get_export(model_id="your_model_id", format="torchscript")
```

**Raises**

| Type | Description |
| --- | --- |
| `AssertionError` | If the specified format is not supported or if the API request fails. |

<details>
<summary>Source code in <code>ultralytics/hub/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py#L117-L143"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def get_export(model_id: str = "", format: str = "torchscript"):
    """Retrieve an exported model in the specified format from Ultralytics HUB using the model ID.

    Args:
        model_id (str): The ID of the model to retrieve from Ultralytics HUB.
        format (str): The export format to retrieve. Must be one of the supported formats returned by export_fmts_hub().

    Returns:
        (dict): JSON response containing the exported model information.

    Raises:
        AssertionError: If the specified format is not supported or if the API request fails.

    Examples:
        >>> from ultralytics import hub
        >>> result = hub.get_export(model_id="your_model_id", format="torchscript")
    """
    import requests  # scoped as slow import

    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
    )
    assert r.status_code == 200, f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()
```
</details>


<br><br><hr><br>

## Function `ultralytics.hub.check_dataset` {#ultralytics.hub.\_\_init\_\_.check\_dataset}

```python
def check_dataset(path: str, task: str) -> None
```

Check HUB dataset Zip file for errors before upload.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `path` | `str` | Path to data.zip (with data.yaml inside data.zip). | *required* |
| `task` | `str` | Dataset task. Options are 'detect', 'segment', 'pose', 'classify', 'obb'. | *required* |

**Examples**

```python
>>> from ultralytics.hub import check_dataset
>>> check_dataset("path/to/coco8.zip", task="detect")  # detect dataset
>>> check_dataset("path/to/coco8-seg.zip", task="segment")  # segment dataset
>>> check_dataset("path/to/coco8-pose.zip", task="pose")  # pose dataset
>>> check_dataset("path/to/dota8.zip", task="obb")  # OBB dataset
>>> check_dataset("path/to/imagenet10.zip", task="classify")  # classification dataset
```

!!! note "Notes"

    Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
    i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.

<details>
<summary>Source code in <code>ultralytics/hub/__init__.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/hub/__init__.py#L146-L166"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def check_dataset(path: str, task: str) -> None:
    """Check HUB dataset Zip file for errors before upload.

    Args:
        path (str): Path to data.zip (with data.yaml inside data.zip).
        task (str): Dataset task. Options are 'detect', 'segment', 'pose', 'classify', 'obb'.

    Examples:
        >>> from ultralytics.hub import check_dataset
        >>> check_dataset("path/to/coco8.zip", task="detect")  # detect dataset
        >>> check_dataset("path/to/coco8-seg.zip", task="segment")  # segment dataset
        >>> check_dataset("path/to/coco8-pose.zip", task="pose")  # pose dataset
        >>> check_dataset("path/to/dota8.zip", task="obb")  # OBB dataset
        >>> check_dataset("path/to/imagenet10.zip", task="classify")  # classification dataset

    Notes:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
        i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
    """
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(f"Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.")
```
</details>

<br><br>
