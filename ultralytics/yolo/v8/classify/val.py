import torch

from ultralytics import yolo


class ClassificationValidator(yolo.BaseValidator):

    def init_metrics(self):
        self.correct = torch.tensor([])

    def update_metrics(self, preds, targets):
        correct_in_batch = (targets[:, None] == preds).float()
        self.correct = torch.cat((self.correct, correct_in_batch))

    def get_stats(self):
        acc = torch.stack((self.correct[:, 0], self.correct.max(1).values), dim=1)  # (top1, top5) accuracy
        top1, top5 = acc.mean(0).tolist()
        return {"top1": top1, "top5": top5, "fitness": top5}
