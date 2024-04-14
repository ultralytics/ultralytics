# !pip install efemarai[full]
import efemarai as ef
import numpy as np
from torchvision import datasets

from ultralytics import YOLO


def get_class_ids_remapping(source_classes, target_classes):
    """Creates dict mapping source id to target id based on class name."""
    lookup_target_id = {v: k for k, v in target_classes.items()}
    return {
        class_id: lookup_target_id[class_name]
        for class_id, class_name in source_classes.items()
        if class_name in lookup_target_id
    }


def main():
    path = "data/coco"

    dataset = datasets.CocoDetection(
        root=f"{path}/val2017",
        annFile=f"{path}/annotations/instances_val2017.json",
    )
    coco_classes = {cat["id"]: cat["name"] for cat in dataset.coco.loadCats(dataset.coco.getCatIds())}

    # domains = [ef.domains.NoiseVariability, ef.domains.GeometricVariability, ef.domains.ColorVariability]
    model_names = ["n", "s", "m", "l", "x"]
    for model_name in model_names:
        model = YOLO(f"yolov8{model_name}.pt")

        report = ef.test_robustness(
            dataset=dataset,
            model=lambda x: model(source=np.array(x), verbose=False)[0].cpu(),
            domain=ef.domains.NoiseVariability,
            dataset_format=ef.formats.COCO_DATASET,
            output_format=ef.formats.ULTRALYTICS_DETECTION,
            class_ids=list(coco_classes.keys()),
            remap_class_ids=get_class_ids_remapping(model.names, coco_classes),
            # hooks=[ef.hooks.show_sample], # uncomment to see generated samples
        )

        report.plot(f"robustness_report_yolov8{model_name}.pdf")


if __name__ == "__main__":
    main()
