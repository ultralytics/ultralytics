from copy import deepcopy
from .augment import *
import collections


# TODO mixup
class MixAndRectDataset:
    """A wrapper of multiple images mixed dataset.

    Args:
        dataset (:obj:`BaseDataset`): The dataset to be mixed.
        transforms (Sequence[dict]): config dict to be composed.
    """

    def __init__(self, dataset, hyp=None):
        self.dataset = dataset
        self.img_size = dataset.img_size
        self.transforms = self.build_transforms(hyp)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        labels = deepcopy(self.dataset[index])
        for transform in self.transforms.tolist():
            # mosaic and mixup
            if hasattr(transform, "get_indexes"):
                indexes = transform.get_indexes(self.dataset)
                if not isinstance(indexes, collections.abc.Sequence):
                    indexes = [indexes]
                mix_labels = [deepcopy(self.dataset[index]) for index in indexes]
                labels["mix_labels"] = mix_labels
            if self.dataset.rect and isinstance(transform, LetterBox):
                transform.new_shape = self.dataset.batch_shapes[self.batch[index]]
            labels = transform(labels)
            if "mix_labels" in labels:
                labels.pop("mix_labels")
        return labels

    def build_transforms(self, hyp):
        # TODO: use hyp config to set these augmentations
        mosaic = self.dataset.augment and not self.dataset.rect
        if self.dataset.augment:
            if mosaic:
                transforms = Compose(
                    [
                        Mosaic(img_size=self.img_size, p=1.0, border=[-self.img_size // 2, -self.img_size // 2]),
                        # CopyPaste(p=0.0),  # TODO: something wrong here
                        RandomPerspective(border=[-self.img_size // 2, -self.img_size // 2]),
                        # # MixUp(p=0.0),   # TODO: something wrong here
                        Albumentations(p=1.0),
                        RandomHSV(),
                        RandomFlip(direction="vertical", p=0.5),
                        RandomFlip(direction="horizontal", p=0.5),
                    ]
                )
            else:
                # rect, randomperspective, albumentation, hsv, flipud, fliplr
                transforms = Compose(
                    [
                        LetterBox(new_shape=(self.img_size, self.img_size)),
                        RandomPerspective(border=[0, 0]),
                        Albumentations(p=1.0),
                        RandomHSV(),
                        RandomFlip(direction="vertical", p=0.5),
                        RandomFlip(direction="horizontal", p=0.5),
                    ]
                )
        else:
            transforms = Compose([LetterBox(new_shape=(self.img_size, self.img_size))])
        transforms.append(Format(bbox_format="xyxy", normalize=True, mask=self.dataset.mask, batch_idx=True))
        return transforms
