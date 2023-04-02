# Ultralytics YOLO 🚀, GPL-3.0 license

import requests

from ultralytics.hub.session import HUBTrainingSession
from ultralytics.hub.utils import PREFIX, split_key
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import LOGGER


def login(api_key=''):
    """
    Log in to the Ultralytics HUB API using the provided API key.

    Args:
        api_key (str, optional): May be an API key or a combination API key and model ID, i.e. key_id

    Example:
        from ultralytics import hub
        hub.login('your_api_key')
    """
    from ultralytics.hub.auth import Auth
    Auth(api_key)


def start(model=''):
    """
    --- DEPRECATED ---
    Start training models with Ultralytics HUB.

    Args:
        model (str, optional): A string containing either the API key and model ID combination (apikey_modelid),
                               or the full model URL (https://hub.ultralytics.com/models/apikey_modelid).

    Usage:
        from ultralytics import hub
        hub.start('API_KEY')

    This function initializes a HUBTrainingSession with the provided model string and trains a YOLO model using
    the session's model file and training arguments.
    """
    session = HUBTrainingSession(model)
    model = YOLO(model=session.model_file, session=session)
    model.train(**session.train_args)


def reset_model(key=''):
    # Reset a trained model to an untrained state
    api_key, model_id = split_key(key)
    r = requests.post('https://api.ultralytics.com/model-reset', json={'apiKey': api_key, 'modelId': model_id})

    if r.status_code == 200:
        LOGGER.info(f'{PREFIX}Model reset successfully')
        return
    LOGGER.warning(f'{PREFIX}Model reset failure {r.status_code} {r.reason}')


def export_fmts_hub():
    # Returns a list of HUB-supported export formats
    from ultralytics.yolo.engine.exporter import export_formats
    return list(export_formats()['Argument'][1:]) + ['ultralytics_tflite', 'ultralytics_coreml']


def export_model(key='', format='torchscript'):
    # Export a model to all formats
    assert format in export_fmts_hub(), f"Unsupported export format '{format}', valid formats are {export_fmts_hub()}"
    api_key, model_id = split_key(key)
    r = requests.post('https://api.ultralytics.com/export',
                      json={
                          'apiKey': api_key,
                          'modelId': model_id,
                          'format': format})
    assert r.status_code == 200, f'{PREFIX}{format} export failure {r.status_code} {r.reason}'
    LOGGER.info(f'{PREFIX}{format} export started ✅')


def get_export(key='', format='torchscript'):
    # Get an exported model dictionary with download URL
    assert format in export_fmts_hub, f"Unsupported export format '{format}', valid formats are {export_fmts_hub}"
    api_key, model_id = split_key(key)
    r = requests.post('https://api.ultralytics.com/get-export',
                      json={
                          'apiKey': api_key,
                          'modelId': model_id,
                          'format': format})
    assert r.status_code == 200, f'{PREFIX}{format} get_export failure {r.status_code} {r.reason}'
    return r.json()


if __name__ == '__main__':
    start()
