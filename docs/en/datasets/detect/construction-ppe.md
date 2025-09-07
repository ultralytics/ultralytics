---
comments: true
description: Discover Construction-PPE, a specialized dataset for detecting helmets, vests, gloves, boots, and goggles in real-world construction sites. Includes compliant and non-compliant scenarios for AI-powered safety monitoring.
keywords: Construction-PPE, PPE dataset, safety compliance, construction workers, object detection, YOLO11, workplace safety, computer vision
---

# Construction-PPE Dataset

The Construction-PPE dataset is designed to improve safety compliance in construction sites by enabling detection of essential protective gear such as helmets, vests, gloves, boots, and goggles, along with annotations for missing equipment. Curated from real construction environments, it includes both compliant and non-compliant cases, making it a valuable resource for training AI models that monitor workplace safety.

## Dataset Structure

The Construction-PPE dataset is organized into three main subsets:

- **Training Set**: The primary collection of annotated construction images featuring workers with both complete and partial PPE usage.
- **Validation Set**: A designated subset used to fine-tune and assess model performance during PPE detection and compliance monitoring.
- **Test Set**: An independent subset reserved for evaluating the final model's effectiveness in detecting PPE and identifying compliance issues.

Each image is annotated in the [Ultralytics YOLO format](../#ultralytics-yolo-format) ensuring compatibility with state-of-the-art [object detection](../../tasks/detect.md) and [tracking](../../modes/track.md) pipelines.

The dataset provides **11 classes** divided into positive (worn PPE) and negative (missing PPE) categories. This dual-positive/negative structure enables models to detect properly worn gear **and** identify safety violations.

## Applications

Construction-PPE powers a variety of safety-focused computer vision applications:

- **Automated compliance monitoring**: Train AI models to instantly check if workers are wearing required safety gear like helmets, vests, or gloves, reducing risks on site.
- **Workplace safety analytics**: Track PPE usage over time, spot frequent violations, and generate insights to improve safety culture.
- **Smart surveillance systems**: Connect detection models with cameras to send real-time alerts when PPE is missing, preventing accidents before they happen.
- **Robotics and autonomous systems**: Enable drones or robots to perform PPE checks across large sites, supporting faster and safer inspections.
- **Research and education**: Provide a real-world dataset for students and researchers exploring workplace safety and human-object interactions.

## Dataset YAML

The Construction-PPE dataset includes a YAML configuration file that defines the training and validation image paths along with the full list of object classes. You can access the `construction-ppe.yaml` file directly in the Ultralytics repository here: [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/construction-ppe.yaml)

!!! example "ultralytics/cfg/datasets/construction-ppe.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/construction-ppe.yaml"
    ```

## Usage

Train YOLO11n on the Construction-PPE dataset with a single command. Below are quick-start examples:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load pretrained model
        model = YOLO("yolo11n.pt")

        # Train the model on Construction-PPE dataset
        model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo detect train data=construction-ppe.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

## Sample Images and Annotations

The dataset captures construction workers across varied environments, lighting conditions, and postures. Both **compliant** and **non-compliant** cases are included.

![Construction-PPE dataset sample image, showing compliant and non-compliant safety gear detection](https://github.com/ultralytics/docs/releases/download/0/construction-ppe-dataset-sample.avif)

## License and Attribution

Construction-PPE is developed and released under the [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE), supporting open-source research and commercial applications with proper attribution.

If you use this dataset in your research, please cite it:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @dataset{Dalvi_Construction_PPE_Dataset_2025,
            author = {Mrunmayee Dalvi and Niyati Singh and Sahil Bhingarde and Ketaki Chalke},
            title = {Construction-PPE: Personal Protective Equipment Detection Dataset},
            month = {January},
            year = {2025},
            version = {1.0.0},
            license = {AGPL-3.0},
            url = {https://docs.ultralytics.com/datasets/detect/construction-ppe/},
            publisher = {Ultralytics}
        }
        ```

## FAQ

### What makes the Construction-PPE dataset unique?

Unlike generic construction datasets, Construction-PPE explicitly includes **missing equipment classes**. This dual-labeling approach allows models to not only detect PPE but also flag violations in real-time.

### Which object categories are included?

The dataset covers helmets, vests, gloves, boots, goggles, and workers, along with their “missing PPE” counterparts. This ensures comprehensive compliance coverage.

### How can I train a YOLO model using the Construction-PPE dataset?

To train a YOLO11 model using the Construction-PPE dataset, you can use the following code snippets:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="construction-ppe.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo detect train data=construction-ppe.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

### Is this dataset suitable for real-world applications?

Yes. Images are curated from real construction sites under diverse conditions. This makes it highly effective for building deployable workplace safety monitoring systems.

### What are the benefits of using the Construction-PPE dataset in AI projects?

The dataset enables real-time detection of personal protective equipment, helping monitor worker safety on construction sites. With classes for both worn and missing gear, it supports AI systems that can automatically flag safety violations, generate compliance insights, and reduce risks. It also provides a practical resource for developing computer vision solutions in workplace safety, robotics, and academic research.
