# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import requests

from ultralytics.data.utils import HUBDatasetStats
from ultralytics.hub.auth import Auth
from ultralytics.hub.session import HUBTrainingSession
from ultralytics.hub.utils import HUB_API_ROOT, HUB_WEB_ROOT, PREFIX, events
from ultralytics.utils import LOGGER, SETTINGS, checks

__all__ = (
    "PREFIX",
    "HUB_WEB_ROOT",
    "HUBTrainingSession",
    "login",
    "logout",
    "reset_model",
    "export_fmts_hub",
    "export_model",
    "get_export",
    "check_dataset",
    "events",
)


def login(api_key: str = None, save: bool = True) -> bool:
    """
    Log in to the Ultralytics HUB API using the provided API key.

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


def logout():
    """
    Log out of Ultralytics HUB by removing the API key from the settings file. To log in again, use 'yolo login'.

    Examples:
        >>> from ultralytics import hub
        >>> hub.logout()
    """
    SETTINGS["api_key"] = ""
    LOGGER.info(f"{PREFIX}logged out ✅. To log in again, use 'yolo login'.")


def reset_model(model_id: str = ""):
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


def export_model(model_id: str = "", format: str = "torchscript"):
    """
    Export a model to a specified format for deployment via the Ultralytics HUB API.

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
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/v1/models/{model_id}/export", json={"format": format}, headers={"x-api-key": Auth().api_key}
    )
    assert r.status_code == 200, f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(model_id: str = "", format: str = "torchscript"):
    """
    Retrieve an exported model in the specified format from Ultralytics HUB using the model ID.

    Args:
        model_id (str): The ID of the model to retrieve from Ultralytics HUB.
        format (str): The export format to retrieve. Must be one of the supported formats returned by export_fmts_hub().

    Raises:
        AssertionError: If the specified format is not supported or if the API request fails.

    Examples:
        >>> from ultralytics import hub
        >>> hub.get_export(model_id="your_model_id", format="torchscript")
    """
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    r = requests.post(
        f"{HUB_API_ROOT}/get-export",
        json={"apiKey": Auth().api_key, "modelId": model_id, "format": format},
        headers={"x-api-key": Auth().api_key},
    )
    assert r.status_code == 200, f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()


def check_dataset(path: str, task: str) -> None:
    """
    Check HUB dataset Zip file for errors before upload.

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

    Note:
        Download *.zip files from https://github.com/ultralytics/hub/tree/main/example_datasets
        i.e. https://github.com/ultralytics/hub/raw/main/example_datasets/coco8.zip for coco8.zip.
    """
    HUBDatasetStats(path=path, task=task).get_json()
    LOGGER.info(f"Checks completed correctly ✅. Upload this dataset to {HUB_WEB_ROOT}/datasets/.")
