import torch

from ultralytics.yolo.engine.validator import BaseValidator


class ClassificationValidator(BaseValidator):

    def init_metrics(self, model):
        self.correct = torch.tensor([])

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        targets = batch["cls"]
        correct_in_batch = (targets[:, None] == preds).float()
        self.correct = torch.cat((self.correct, correct_in_batch))

    def get_stats(self):
        acc = torch.stack((self.correct[:, 0], self.correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        return {"top1": top1, "top5": top5, "fitness": top5}
