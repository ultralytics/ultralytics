# Ultralytics YOLO 🚀, AGPL-3.0 license

import requests

from ultralytics.data.utils import HUBDatasetStats
from ultralytics.hub.auth import Auth
from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX
from ultralytics.utils import LOGGER, SETTINGS, checks


def login(api_key: str = None, save=True) -> bool:
    """
    Log in to the Ultralytics HUB API using the provided API key.

    The session is not stored; a new session is created when needed using the saved SETTINGS or the HUB_API_KEY
    environment variable if successfully authenticated.

    Args:
        api_key (str, optional): API key to use for authentication.
            If not provided, it will be retrieved from SETTINGS or HUB_API_KEY environment variable.
        save (bool, optional): Whether to save the API key to SETTINGS if authentication is successful.

    Returns:
        (bool): True if authentication is successful, False otherwise.
    """
    checks.check_requirements("hub-sdk>=0.0.6")
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
        LOGGER.info(f"{PREFIX}Get API key from {api_key_url} and then run 'yolo hub login API_KEY'")
        return False


def logout():
    """
    Log out of Ultralytics HUB by removing the API key from the settings file. To log in again, use 'yolo hub login'.

    Example:
        ```python
        from ultralytics import hub

        hub.logout()
        ```
    """
    SETTINGS["api_key"] = ""
    SETTINGS.save()
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo hub login'.")


def reset_model(model_id=""):
    """Reset a trained model to an untrained state."""
    r = requests.post(f"{HUB_API_ROOT}/model-reset", json={"modelId": model_id}, headers={"x-api-key": Auth().api_key})
    if r.status_code == 200:
        LOGGER.info(f"{PREFIX}Model reset successfully")
        return
    LOGGER.warning(f"{PREFIX}Model reset failure {r.status_code} {r.reason}")


def export_fmts_hub():
    """Returns a list of HUB-supported export formats."""
    from ultralytics.engine.exporter import export_formats

    return list(export_formats()["Argument"][1:]) + ["ultralytics_tflite", "ultralytics_coreml"]


def export_model(model_id="", format="torchscript"):
    """Export a model to all formats."""
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export", json={"format": format}, headers={"x-api-key": Auth().api_key}
    )
    assert r.status_code == 200, f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(model_id="", format="torchscript"):
    """Get an exported model dictionary with download URL."""
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
    )
    assert r.status_code == 200, f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()


def check_dataset(path="", task="detect"):
    """
    Function for error-checking HUB dataset Zip file before upload. It checks a dataset for errors before it is uploaded
    to the HUB. Usage examples are given below.

    Args:
        path (str, optional): Path to data.zip (with data.yaml inside data.zip). Defaults to ''.
        task (str, optional): Dataset task. Options are 'detect', 'segment', 'pose', 'classify'. Defaults to 'detect'.

    Example:
        ```python
        from ultralytics.hub import check_dataset

        check_dataset('path/to/coco8.zip', task='detect')  # detect dataset
        check_dataset('path/to/coco8-seg.zip', task='segment')  # segment dataset
        check_dataset('path/to/coco8-pose.zip', task='pose')  # pose dataset
        ```
    """
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(f"Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.")
