# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

# sys.path.append('../../specialization_project/')
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
# from src.utils.external_coco_utils.utils import collate_fn

from config import DATASET_MEAN, DATASET_STD


def get_transforms(
    randomCrop=False,
    resize=False,
    randomVerticalFlip=False,
    randomHorizontalFlip=False,
    blur=False,
    medianBlur=False,
    gaussianBlur=False,
    randomBrightnessContrast=False,
):
    # https://www.kaggle.com/code/ankursingh12/data-augmentation-for-object-detection#Mosaic-Augmentation
    # For mer avanserte augmentations med object detection

    bbox_params = A.BboxParams(format="pascal_voc", label_fields=["labels"])
    train_transforms = A.Compose(
        [
            A.Resize(855, 1920),
            A.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2(p=1.0),
        ],
        bbox_params=bbox_params,
    )

    val_transforms = A.Compose(
        [A.Resize(855, 1920), A.Normalize(mean=DATASET_MEAN, std=DATASET_STD), ToTensorV2(p=1.0)],
        bbox_params=bbox_params,
    )

    test_transforms = A.Compose(
        [A.Resize(855, 1920), A.Normalize(mean=DATASET_MEAN, std=DATASET_STD), ToTensorV2(p=1.0)],
        bbox_params=bbox_params,
    )

    return train_transforms, val_transforms, test_transforms


def calculate_mean_and_std(torch_dataset):
    dataloader = DataLoader(
        torch_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        # persistent_workers=True,
    )

    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for inputs, target in tqdm(dataloader):
        inputs = inputs[0]

        psum += inputs.sum(axis=[1, 2])
        psum_sq += (inputs**2).sum(axis=[1, 2])

    count = len(torch_dataset) * 855 * 1920

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    return total_mean, total_std
