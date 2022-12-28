import requests

from ultralytics import __version__
from ultralytics.hub.auth import Auth
from ultralytics.hub.utils import PREFIX, split_key
from ultralytics.yolo.utils import LOGGER, emojis, is_colab
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.torch_utils import select_device

# from .trainer import Trainer


def checks(verbose=True):
    import os
    import shutil

    check_requirements(('psutil', 'IPython'))
    import psutil
    from IPython import display  # to display images and clear console output

    if is_colab():
        shutil.rmtree('sample_data', ignore_errors=True)  # remove colab /sample_data directory

    if verbose:
        # System info
        # gb = 1 / 1000 ** 3  # bytes to GB
        gib = 1 / 1024 ** 3  # bytes to GiB
        ram = psutil.virtual_memory().total
        total, used, free = shutil.disk_usage("/")
        display.clear_output()
        s = f'({os.cpu_count()} CPUs, {ram * gib:.1f} GB RAM, {(total - free) * gib:.1f}/{total * gib:.1f} GB disk)'
    else:
        s = ''

    select_device(newline=False, version=__version__)
    print(emojis(f'Setup complete ✅ {s}'))


def start(key=''):
    # Start training models with Ultralytics HUB. Usage: from src.ultralytics import start; start('API_KEY')
    def split_key(key: str) -> list:
        return (key.split('_')) if '_' in key else (key, None)

    def request_api_key(attempts=0):
        """Prompt the user to input their API key"""
        import getpass

        max_attempts = 3
        tries = f"Attempt {str(attempts + 1)} of {max_attempts}" if attempts > 0 else ""
        LOGGER.info(f"{PREFIX}Login. {tries}")
        input_key = getpass.getpass("Enter your Ultralytics HUB API key:\n")
        authCtrl.api_key, model_id = split_key(input_key)
        if not authCtrl.authenticate():
            attempts += 1
            LOGGER.warning(emojis(f"{PREFIX}Invalid API key ⚠️\n"))
            if attempts < max_attempts:
                return request_api_key(attempts)
            raise Exception(emojis(f"{PREFIX}Failed to authenticate ❌"))
        else:
            return model_id

    try:
        api_key, model_id = split_key(key)
        authCtrl = Auth(api_key)  # attempts cookie login if no api key is present
        attempts = 1 if len(key) else 0
        if not authCtrl.get_state():
            if len(key):
                LOGGER.warning(emojis(f"{PREFIX}Invalid API key ⚠️\n"))
            model_id = request_api_key(attempts)
        LOGGER.info(emojis(f"{PREFIX}Authenticated ✅"))
        if not model_id:
            raise Exception(emojis('Connecting with global API key is not currently supported. ❌'))
        '''
        TODO:
        trainer = Trainer(model_id=model_id, auth=authCtrl)
        if trainer.model is not None:
            trainer.start()
        '''
    except Exception as e:
        LOGGER.warning(f"{PREFIX}{e}")


def reset_model(key=''):
    # Reset a trained model to an untrained state
    api_key, model_id = split_key(key)
    r = requests.post('https://api.ultralytics.com/model-reset', json={"apiKey": api_key, "modelId": model_id})

    if r.status_code == 200:
        LOGGER.info(f"{PREFIX}model reset successfully")
        return
    LOGGER.warning(f"{PREFIX}model reset failure {r.status_code} {r.reason}")


def export_model(key='', format='torchscript'):
    # Export a model to all formats
    api_key, model_id = split_key(key)
    formats = ('torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs',
               'ultralytics_tflite', 'ultralytics_coreml')
    assert format in formats, f"ERROR: Unsupported export format '{format}' passed, valid formats are {formats}"

    r = requests.post('https://api.ultralytics.com/export',
                      json={
                          "apiKey": api_key,
                          "modelId": model_id,
                          "format": format})
    assert r.status_code == 200, f"{PREFIX}{format} export failure {r.status_code} {r.reason}"
    LOGGER.info(f"{PREFIX}{format} export started ✅")


def get_export(key='', format='torchscript'):
    # Get an exported model dictionary with download URL
    api_key, model_id = split_key(key)
    formats = ('torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs',
               'ultralytics_tflite', 'ultralytics_coreml')
    assert format in formats, f"ERROR: Unsupported export format '{format}' passed, valid formats are {formats}"

    r = requests.post('https://api.ultralytics.com/get-export',
                      json={
                          "apiKey": api_key,
                          "modelId": model_id,
                          "format": format})
    assert r.status_code == 200, f"{PREFIX}{format} get_export failure {r.status_code} {r.reason}"
    return r.json()
