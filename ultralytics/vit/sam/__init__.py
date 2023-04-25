from .predictor import Predictor


class sam:

    def __init__(self, model='sam_b.pt') -> None:
        self.predictor = Predictor(overrides={'model': model})

    def predict(self, source):
        return self.predictor(source)

    def train(self, **kwargs):
        raise NotImplementedError("SAM models don't support training")

    def val(self, **kwargs):
        raise NotImplementedError("SAM models don't support validation")
