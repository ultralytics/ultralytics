# Ultralytics YOLO 🚀, GPL-3.0 license

import requests

from ultralytics.hub.auth import Auth
from ultralytics.hub.session import HubTrainingSession
from ultralytics.hub.utils import PREFIX, split_key
from ultralytics.yolo.utils import LOGGER, emojis
from ultralytics.yolo.v8.detect import DetectionTrainer


def start(key=''):
    # Start training models with Ultralytics HUB. Usage: from src.ultralytics import start; start('API_KEY')
    def request_api_key(attempts=0):
        """Prompt the user to input their API key"""
        import getpass

        max_attempts = 3
        tries = f"Attempt {str(attempts + 1)} of {max_attempts}" if attempts > 0 else ""
        LOGGER.info(f"{PREFIX}Login. {tries}")
        input_key = getpass.getpass("Enter your Ultralytics HUB API key:\n")
        auth.api_key, model_id = split_key(input_key)
        if not auth.authenticate():
            attempts += 1
            LOGGER.warning(f"{PREFIX}Invalid API key ⚠️\n")
            if attempts < max_attempts:
                return request_api_key(attempts)
            raise ConnectionError(emojis(f"{PREFIX}Failed to authenticate ❌"))
        else:
            return model_id

    try:
        api_key, model_id = split_key(key)
        auth = Auth(api_key)  # attempts cookie login if no api key is present
        attempts = 1 if len(key) else 0
        if not auth.get_state():
            if len(key):
                LOGGER.warning(f"{PREFIX}Invalid API key ⚠️\n")
            model_id = request_api_key(attempts)
        LOGGER.info(f"{PREFIX}Authenticated ✅")
        if not model_id:
            raise ConnectionError(emojis('Connecting with global API key is not currently supported. ❌'))
        session = HubTrainingSession(model_id=model_id, auth=auth)
        session.check_disk_space()

        # TODO: refactor, hardcoded for v8
        args = session.model.copy()
        args.pop("id")
        args.pop("status")
        args.pop("weights")
        args["data"] = "coco128.yaml"
        args["model"] = "yolov8n.yaml"
        args["batch_size"] = 16
        args["imgsz"] = 64

        trainer = DetectionTrainer(overrides=args)
        session.register_callbacks(trainer)
        setattr(trainer, 'hub_session', session)
        trainer.train()
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


# temp. For checking
if __name__ == "__main__":
    start(key="b3fba421be84a20dbe68644e14436d1cce1b0a0aaa_HeMfHgvHsseMPhdq7Ylz")
