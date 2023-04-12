from ultralytics import YOLO
from time import sleep


class Buzzer:
    def __init__(self, is_rpi: bool) -> None:
        if is_rpi:
            import RPi.GPIO as GPIO
            self.module = GPIO
            self.rpi = True
            self.module.setmode(GPIO.BCM)
            self._buzzer = 23
            self.module.setup(self._buzzer, GPIO.OUT)
        else:
            import winsound
            self.module = winsound
            self.rpi = False
    
    def buzz(self) -> None:
        if self.rpi:
            self.module.output(self._buzzer, 1)
            sleep(0.5)
            self.module.output(self._buzzer, 0)
        else:
            self.module.Beep(600, 1000)


class CustomPredictor:
    def __init__(self, model_path: str, source: str, is_stream: bool, is_rpi: bool, device: str, save: bool) -> None:
        self._buzzer = Buzzer(is_rpi)
        self._stream = is_stream
        self._save = save
        self._path = model_path
        self._device = device
        self._model = YOLO(self._path)
        self._results = self._model.predict(mode="predict", source=source, stream=self._stream, device=self._device, show=True, save=self._save)
        for result in self._results:
            if result.boxes:
                self._buzz()
    
    def _buzz(self) -> None:
        self._buzzer.buzz()

