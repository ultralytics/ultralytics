from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from supervision.classification.core import Classifications
from supervision.dataset.formats.coco import (
    load_coco_annotations,
    save_coco_annotations,
)
from supervision.dataset.formats.pascal_voc import (
    detections_to_pascal_voc,
    load_pascal_voc_annotations,
)
from supervision.dataset.formats.yolo import (
    load_yolo_annotations,
    save_data_yaml,
    save_yolo_annotations,
)
from supervision.dataset.utils import (
    build_class_index_mapping,
    map_detections_class_id,
    merge_class_lists,
    save_dataset_images,
    train_test_split,
)
from supervision.detection.core import Detections


@dataclass
class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def split(
        self, split_ratio=0.8, random_state=None, shuffle: bool = True
    ) -> Tuple[BaseDataset, BaseDataset]:
        pass


@dataclass
class DetectionDataset(BaseDataset):
    """
    Dataclass containing information about object detection dataset.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Dict[str, np.ndarray]): Dictionary mapping image name to image.
        annotations (Dict[str, Detections]): Dictionary mapping
            image name to annotations.
    """

    classes: List[str]
    images: Dict[str, np.ndarray]
    annotations: Dict[str, Detections]

    def __len__(self) -> int:
        """
        Return the number of images in the dataset.

        Returns:
            int: The number of images.
        """
        return len(self.images)

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray, Detections]]:
        """
        Iterate over the images and annotations in the dataset.

        Yields:
            Iterator[Tuple[str, np.ndarray, Detections]]:
                An iterator that yields tuples containing the image name,
                the image data, and its corresponding annotation.
        """
        for image_name, image in self.images.items():
            yield image_name, image, self.annotations.get(image_name, None)

    def __eq__(self, other):
        if not isinstance(other, DetectionDataset):
            return False

        if set(self.classes) != set(other.classes):
            return False

        for key in self.images:
            if not np.array_equal(self.images[key], other.images[key]):
                return False
            if not self.annotations[key] == other.annotations[key]:
                return False

        return True

    def split(
        self, split_ratio=0.8, random_state=None, shuffle: bool = True
    ) -> Tuple[DetectionDataset, DetectionDataset]:
        """
        Splits the dataset into two parts (training and testing)
            using the provided split_ratio.

        Args:
            split_ratio (float, optional): The ratio of the training
                set to the entire dataset.
            random_state (int, optional): The seed for the random number generator.
                This is used for reproducibility.
            shuffle (bool, optional): Whether to shuffle the data before splitting.

        Returns:
            Tuple[DetectionDataset, DetectionDataset]: A tuple containing
                the training and testing datasets.

        Example:
            ```python
            >>> import supervision as sv

            >>> ds = sv.DetectionDataset(...)
            >>> train_ds, test_ds = ds.split(split_ratio=0.7,
            ...                              random_state=42, shuffle=True)
            >>> len(train_ds), len(test_ds)
            (700, 300)
            ```
        """

        image_names = list(self.images.keys())
        train_names, test_names = train_test_split(
            data=image_names,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_dataset = DetectionDataset(
            classes=self.classes,
            images={name: self.images[name] for name in train_names},
            annotations={name: self.annotations[name] for name in train_names},
        )
        test_dataset = DetectionDataset(
            classes=self.classes,
            images={name: self.images[name] for name in test_names},
            annotations={name: self.annotations[name] for name in test_names},
        )
        return train_dataset, test_dataset

    def as_pascal_voc(
        self,
        images_directory_path: Optional[str] = None,
        annotations_directory_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
        """
        Exports the dataset to PASCAL VOC format. This method saves the images
        and their corresponding annotations in PASCAL VOC format.

        Args:
            images_directory_path (Optional[str]): The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path (Optional[str]): The path to
                the directory where the annotations in PASCAL VOC format should be
                saved. If not provided, annotations will not be saved.
            min_image_area_percentage (float): The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage (float): The maximum percentage
                of detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage (float): The percentage of
                polygon points to be removed from the input polygon,
                in the range [0, 1). Argument is used only for segmentation datasets.
        """
        if images_directory_path:
            save_dataset_images(
                images_directory_path=images_directory_path, images=self.images
            )
        if annotations_directory_path:
            Path(annotations_directory_path).mkdir(parents=True, exist_ok=True)

        for image_path, image in self.images.items():
            detections = self.annotations[image_path]

            if annotations_directory_path:
                annotation_name = Path(image_path).stem
                annotations_path = os.path.join(
                    annotations_directory_path, f"{annotation_name}.xml"
                )
                image_name = Path(image_path).name
                pascal_voc_xml = detections_to_pascal_voc(
                    detections=detections,
                    classes=self.classes,
                    filename=image_name,
                    image_shape=image.shape,
                    min_image_area_percentage=min_image_area_percentage,
                    max_image_area_percentage=max_image_area_percentage,
                    approximation_percentage=approximation_percentage,
                )

                with open(annotations_path, "w") as f:
                    f.write(pascal_voc_xml)

    @classmethod
    def from_pascal_voc(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from PASCAL VOC formatted data.

        Args:
            images_directory_path (str): Path to the directory containing the images.
            annotations_directory_path (str): Path to the directory
                containing the PASCAL VOC XML annotations.
            force_masks (bool, optional): If True, forces masks to
                be loaded for all annotations, regardless of whether they are present.

        Returns:
            DetectionDataset: A DetectionDataset instance containing
                the loaded images and annotations.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("voc")

            >>> ds = sv.DetectionDataset.from_pascal_voc(
            ...     images_directory_path=f"{dataset.location}/train/images",
            ...     annotations_directory_path=f"{dataset.location}/train/labels"
            ... )

            >>> ds.classes
            ['dog', 'person']
            ```
        """

        classes, images, annotations = load_pascal_voc_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            force_masks=force_masks,
        )

        return DetectionDataset(classes=classes, images=images, annotations=annotations)

    @classmethod
    def from_yolo(
        cls,
        images_directory_path: str,
        annotations_directory_path: str,
        data_yaml_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from YOLO formatted data.

        Args:
            images_directory_path (str): The path to the
                directory containing the images.
            annotations_directory_path (str): The path to the directory
                containing the YOLO annotation files.
            data_yaml_path (str): The path to the data
                YAML file containing class information.
            force_masks (bool, optional): If True, forces
                masks to be loaded for all annotations,
                regardless of whether they are present.

        Returns:
            DetectionDataset: A DetectionDataset instance
                containing the loaded images and annotations.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("yolov5")

            >>> ds = sv.DetectionDataset.from_yolo(
            ...     images_directory_path=f"{dataset.location}/train/images",
            ...     annotations_directory_path=f"{dataset.location}/train/labels",
            ...     data_yaml_path=f"{dataset.location}/data.yaml"
            ... )

            >>> ds.classes
            ['dog', 'person']
            ```
        """
        classes, images, annotations = load_yolo_annotations(
            images_directory_path=images_directory_path,
            annotations_directory_path=annotations_directory_path,
            data_yaml_path=data_yaml_path,
            force_masks=force_masks,
        )
        return DetectionDataset(classes=classes, images=images, annotations=annotations)

    def as_yolo(
        self,
        images_directory_path: Optional[str] = None,
        annotations_directory_path: Optional[str] = None,
        data_yaml_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
        """
        Exports the dataset to YOLO format. This method saves the
        images and their corresponding annotations in YOLO format.

        Args:
            images_directory_path (Optional[str]): The path to the
                directory where the images should be saved.
                If not provided, images will not be saved.
            annotations_directory_path (Optional[str]): The path to the
                directory where the annotations in
                YOLO format should be saved. If not provided,
                annotations will not be saved.
            data_yaml_path (Optional[str]): The path where the data.yaml
                file should be saved.
                If not provided, the file will not be saved.
            min_image_area_percentage (float): The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage (float): The maximum percentage
                of detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage (float): The percentage of polygon points to
                be removed from the input polygon, in the range [0, 1).
                This is useful for simplifying the annotations.
                Argument is used only for segmentation datasets.
        """
        if images_directory_path is not None:
            save_dataset_images(
                images_directory_path=images_directory_path, images=self.images
            )
        if annotations_directory_path is not None:
            save_yolo_annotations(
                annotations_directory_path=annotations_directory_path,
                images=self.images,
                annotations=self.annotations,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )
        if data_yaml_path is not None:
            save_data_yaml(data_yaml_path=data_yaml_path, classes=self.classes)

    @classmethod
    def from_coco(
        cls,
        images_directory_path: str,
        annotations_path: str,
        force_masks: bool = False,
    ) -> DetectionDataset:
        """
        Creates a Dataset instance from COCO formatted data.

        Args:
            images_directory_path (str): The path to the
                directory containing the images.
            annotations_path (str): The path to the json annotation files.
            force_masks (bool, optional): If True,
                forces masks to be loaded for all annotations,
                regardless of whether they are present.

        Returns:
            DetectionDataset: A DetectionDataset instance containing
                the loaded images and annotations.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("coco")

            >>> ds = sv.DetectionDataset.from_coco(
            ...     images_directory_path=f"{dataset.location}/train",
            ...     annotations_path=f"{dataset.location}/train/_annotations.coco.json",
            ... )

            >>> ds.classes
            ['dog', 'person']
            ```
        """
        classes, images, annotations = load_coco_annotations(
            images_directory_path=images_directory_path,
            annotations_path=annotations_path,
            force_masks=force_masks,
        )
        return DetectionDataset(classes=classes, images=images, annotations=annotations)

    def as_coco(
        self,
        images_directory_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        min_image_area_percentage: float = 0.0,
        max_image_area_percentage: float = 1.0,
        approximation_percentage: float = 0.0,
    ) -> None:
        """
        Exports the dataset to COCO format. This method saves the
        images and their corresponding annotations in COCO format.

        Args:
            images_directory_path (Optional[str]): The path to the directory
                where the images should be saved.
                If not provided, images will not be saved.
            annotations_path (Optional[str]): The path to COCO annotation file.
            min_image_area_percentage (float): The minimum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            max_image_area_percentage (float): The maximum percentage of
                detection area relative to
                the image area for a detection to be included.
                Argument is used only for segmentation datasets.
            approximation_percentage (float): The percentage of polygon points
                to be removed from the input polygon,
                in the range [0, 1). This is useful for simplifying the annotations.
                Argument is used only for segmentation datasets.
        """
        if images_directory_path is not None:
            save_dataset_images(
                images_directory_path=images_directory_path, images=self.images
            )
        if annotations_path is not None:
            save_coco_annotations(
                annotation_path=annotations_path,
                images=self.images,
                annotations=self.annotations,
                classes=self.classes,
                min_image_area_percentage=min_image_area_percentage,
                max_image_area_percentage=max_image_area_percentage,
                approximation_percentage=approximation_percentage,
            )

    @classmethod
    def merge(cls, dataset_list: List[DetectionDataset]) -> DetectionDataset:
        """
        Merge a list of `DetectionDataset` objects into a single
            `DetectionDataset` object.

        This method takes a list of `DetectionDataset` objects and combines
        their respective fields (`classes`, `images`,
        `annotations`) into a single `DetectionDataset` object.

        Args:
            dataset_list (List[DetectionDataset]): A list of `DetectionDataset`
                objects to merge.

        Returns:
            (DetectionDataset): A single `DetectionDataset` object containing
            the merged data from the input list.

        Example:
            ```python
            >>> import supervision as sv

            >>> ds_1 = sv.DetectionDataset(...)
            >>> len(ds_1)
            100
            >>> ds_1.classes
            ['dog', 'person']

            >>> ds_2 = sv.DetectionDataset(...)
            >>> len(ds_2)
            200
            >>> ds_2.classes
            ['cat']

            >>> ds_merged = sv.DetectionDataset.merge([ds_1, ds_2])
            >>> len(ds_merged)
            300
            >>> ds_merged.classes
            ['cat', 'dog', 'person']
            ```
        """
        merged_images, merged_annotations = {}, {}
        class_lists = [dataset.classes for dataset in dataset_list]
        merged_classes = merge_class_lists(class_lists=class_lists)

        for dataset in dataset_list:
            class_index_mapping = build_class_index_mapping(
                source_classes=dataset.classes, target_classes=merged_classes
            )
            for image_name, image, detections in dataset:
                if image_name in merged_annotations:
                    raise ValueError(
                        f"Image name {image_name} is not unique across datasets."
                    )

                merged_images[image_name] = image
                merged_annotations[image_name] = map_detections_class_id(
                    source_to_target_mapping=class_index_mapping,
                    detections=detections,
                )

        return cls(
            classes=merged_classes, images=merged_images, annotations=merged_annotations
        )


@dataclass
class ClassificationDataset(BaseDataset):
    """
    Dataclass containing information about a classification dataset.

    Attributes:
        classes (List[str]): List containing dataset class names.
        images (Dict[str, np.ndarray]): Dictionary mapping image name to image.
        annotations (Dict[str, Detections]): Dictionary mapping
            image name to annotations.
    """

    classes: List[str]
    images: Dict[str, np.ndarray]
    annotations: Dict[str, Classifications]

    def __len__(self) -> int:
        return len(self.images)

    def split(
        self, split_ratio=0.8, random_state=None, shuffle: bool = True
    ) -> Tuple[ClassificationDataset, ClassificationDataset]:
        """
        Splits the dataset into two parts (training and testing)
            using the provided split_ratio.

        Args:
            split_ratio (float, optional): The ratio of the training
                set to the entire dataset.
            random_state (int, optional): The seed for the
                random number generator. This is used for reproducibility.
            shuffle (bool, optional): Whether to shuffle the data before splitting.

        Returns:
            Tuple[ClassificationDataset, ClassificationDataset]: A tuple containing
            the training and testing datasets.

        Example:
            ```python
            >>> import supervision as sv

            >>> cd = sv.ClassificationDataset(...)
            >>> train_cd,test_cd = cd.split(split_ratio=0.7,
            ...                             random_state=42,shuffle=True)
            >>> len(train_cd), len(test_cd)
            (700, 300)
            ```
        """
        image_names = list(self.images.keys())
        train_names, test_names = train_test_split(
            data=image_names,
            train_ratio=split_ratio,
            random_state=random_state,
            shuffle=shuffle,
        )

        train_dataset = ClassificationDataset(
            classes=self.classes,
            images={name: self.images[name] for name in train_names},
            annotations={name: self.annotations[name] for name in train_names},
        )
        test_dataset = ClassificationDataset(
            classes=self.classes,
            images={name: self.images[name] for name in test_names},
            annotations={name: self.annotations[name] for name in test_names},
        )
        return train_dataset, test_dataset

    def as_folder_structure(self, root_directory_path: str) -> None:
        """
        Saves the dataset as a multi-class folder structure.

        Args:
            root_directory_path (str): The path to the directory
                where the dataset will be saved.
        """
        os.makedirs(root_directory_path, exist_ok=True)

        for class_name in self.classes:
            os.makedirs(os.path.join(root_directory_path, class_name), exist_ok=True)

        for image_path in self.images:
            classification = self.annotations[image_path]
            image = self.images[image_path]
            image_name = Path(image_path).name
            class_id = (
                classification.class_id[0]
                if classification.confidence is None
                else classification.get_top_k(1)[0][0]
            )
            class_name = self.classes[class_id]
            image_path = os.path.join(root_directory_path, class_name, image_name)
            cv2.imwrite(image_path, image)

    @classmethod
    def from_folder_structure(cls, root_directory_path: str) -> ClassificationDataset:
        """
        Load data from a multiclass folder structure into a ClassificationDataset.

        Args:
            root_directory_path (str): The path to the dataset directory.

        Returns:
            ClassificationDataset: The dataset.

        Example:
            ```python
            >>> import roboflow
            >>> from roboflow import Roboflow
            >>> import supervision as sv

            >>> roboflow.login()

            >>> rf = Roboflow()

            >>> project = rf.workspace(WORKSPACE_ID).project(PROJECT_ID)
            >>> dataset = project.version(PROJECT_VERSION).download("folder")

            >>> cd = sv.ClassificationDataset.from_folder_structure(
            ...     root_directory_path=f"{dataset.location}/train"
            ... )
            ```
        """
        classes = os.listdir(root_directory_path)
        classes = sorted(set(classes))

        images = {}
        annotations = {}

        for class_name in classes:
            class_id = classes.index(class_name)

            for image in os.listdir(os.path.join(root_directory_path, class_name)):
                image_path = str(os.path.join(root_directory_path, class_name, image))
                images[image_path] = cv2.imread(image_path)
                annotations[image_path] = Classifications(
                    class_id=np.array([class_id]),
                )

        return cls(
            classes=classes,
            images=images,
            annotations=annotations,
        )
