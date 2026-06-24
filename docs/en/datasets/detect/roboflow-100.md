---
comments: true
description: Explore the Roboflow 100 dataset featuring 100 diverse datasets designed to test object detection models across various domains, from healthcare to video games.
keywords: Roboflow 100, Ultralytics, object detection, dataset, benchmarking, machine learning, computer vision, diverse datasets, model evaluation
---

# Roboflow 100 Dataset

Roboflow 100, sponsored by [Intel](https://www.intel.com/), is a groundbreaking [object detection](../../tasks/detect.md) benchmark dataset. It includes 100 diverse datasets. This benchmark is specifically designed to test the adaptability of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models, like [Ultralytics YOLO models](../../models/yolo26.md), to various domains, including healthcare, aerial imagery, and video games.

!!! question "Licensing"

    Ultralytics offers two licensing options to accommodate different use cases:

    - **AGPL-3.0 License**: This [OSI-approved](https://opensource.org/license) open-source license is ideal for students and enthusiasts, promoting open collaboration and knowledge sharing. See the [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) file for more details and visit our [AGPL-3.0 License page](https://www.ultralytics.com/legal/agpl-3-0-software-license).
    - **Enterprise License**: For development and production use, this license enables seamless integration of Ultralytics software and AI models into business products and services, including internal tools, automated workflows, and production deployments, bypassing the open-source requirements of AGPL-3.0. To get started, please contact us via [Ultralytics Licensing](https://www.ultralytics.com/license).

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/roboflow-100-overview.avif" alt="Roboflow 100 diverse object detection benchmark">
</p>

## Key Features

- **Diverse Domains**: Includes 100 datasets across seven distinct domains: Aerial, Video games, Microscopic, Underwater, Documents, Electromagnetic, and Real World.
- **Scale**: The benchmark comprises 224,714 images across 805 classes, representing over 11,170 hours of [data labeling](https://www.ultralytics.com/glossary/data-labeling) effort.
- **Standardization**: All images are [preprocessed](https://www.ultralytics.com/glossary/data-preprocessing) and resized to 640x640 pixels for consistent evaluation.
- **Clean Evaluation**: Focuses on eliminating class ambiguity and filters out underrepresented classes to ensure cleaner [model evaluation](../../guides/model-evaluation-insights.md).
- **Annotations**: Includes [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) for objects, suitable for [training](../../modes/train.md) and evaluating object detection models using metrics like [mAP](https://www.ultralytics.com/glossary/mean-average-precision-map).

## Dataset Structure

The Roboflow 100 dataset is organized into seven categories, each containing a unique collection of datasets, images, and classes:

- **Aerial**: 7 datasets, 9,683 images, 24 classes.
- **Video Games**: 7 datasets, 11,579 images, 88 classes.
- **Microscopic**: 11 datasets, 13,378 images, 28 classes.
- **Underwater**: 5 datasets, 18,003 images, 39 classes.
- **Documents**: 8 datasets, 24,813 images, 90 classes.
- **Electromagnetic**: 12 datasets, 36,381 images, 41 classes.
- **Real World**: 50 datasets, 110,615 images, 495 classes.

This structure provides a diverse and extensive testing ground for [object detection](https://www.ultralytics.com/glossary/object-detection) models, reflecting a wide array of real-world application scenarios found in various [Ultralytics Solutions](https://www.ultralytics.com/solutions).

## Benchmarking

Dataset [benchmarking](../../modes/benchmark.md) involves evaluating the performance of [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) models on specific datasets using standardized metrics. Common metrics include [accuracy](https://www.ultralytics.com/glossary/accuracy), mean Average Precision (mAP), and [F1-score](https://www.ultralytics.com/glossary/f1-score). You can learn more about these in our [YOLO Performance Metrics guide](../../guides/yolo-performance-metrics.md).

!!! tip "Benchmarking Results"

    Each dataset is fine-tuned in its own folder under `runs/` (with its own `results.png`), and a cross-dataset bar chart of the per-dataset metric is saved as `multitrain_results.png`. The `model.train()` call also returns a `{dataset: metrics}` dictionary for programmatic access.

!!! example "Benchmarking Example"

    The script below downloads the Roboflow 100 datasets listed in `datasets_links.txt` from Roboflow, then fine-tunes a single base model (e.g., YOLO26n) across the whole collection in one `model.train()` call. Passing a list of datasets fine-tunes the base model on each one in series and automatically visualizes the cross-dataset results. A free [Roboflow API key](https://docs.roboflow.com/api-reference/authentication) is required to download the datasets.

    === "Python"

        ```python
        import re
        from pathlib import Path

        from ultralytics import YOLO
        from ultralytics.utils import ASSETS_URL, YAML
        from ultralytics.utils.checks import check_requirements
        from ultralytics.utils.downloads import safe_download

        # Download the RF100 datasets from Roboflow (requires a free Roboflow API key)
        check_requirements("roboflow")
        from roboflow import Roboflow

        rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
        safe_download(f"{ASSETS_URL}/datasets_links.txt")  # list of RF100 dataset links

        datasets = []
        for line in Path("datasets_links.txt").read_text().splitlines():
            try:
                _, _url, workspace, project, version = re.split("/+", line.strip())
                location = f"rf-100/{project}-{version}"
                rf.workspace(workspace).project(project).version(version).download("yolov8", location=location)
                yaml = Path(location) / "data.yaml"
                cfg = YAML.load(yaml)  # point train/val at the downloaded image folders
                cfg["train"], cfg["val"] = "train/images", "valid/images"
                YAML.save(yaml, cfg)
                datasets.append(str(yaml))
            except Exception:
                continue

        # Fine-tune one base model across all RF100 datasets and visualize the cross-dataset results
        model = YOLO("yolo26n.pt")
        results = model.train(data=datasets, epochs=100, imgsz=640)  # {run: metrics}; saves multitrain_results.png

        # Inspect the per-dataset results
        for run, metrics in results.items():
            if metrics:  # None if that dataset failed to train
                print(f"{run}: mAP50-95 = {metrics['metrics/mAP50-95(B)']:.4f}")
        ```

## Applications

Roboflow 100 is invaluable for various applications related to [computer vision](https://www.ultralytics.com/blog/everything-you-need-to-know-about-computer-vision-in-2025) and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl). Researchers and engineers can leverage this benchmark to:

- Evaluate the performance of object detection models in a multi-domain context.
- Test the adaptability and [robustness](<https://en.wikipedia.org/wiki/Robustness_(computer_science)>) of models to real-world scenarios beyond common [benchmark datasets](https://www.ultralytics.com/glossary/benchmark-dataset) like [COCO](https://cocodataset.org/#home) or [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).
- Benchmark the capabilities of object detection models across diverse datasets, including specialized areas like healthcare, aerial imagery, and video games.
- Compare model performance across different [neural network](https://www.ultralytics.com/glossary/neural-network-nn) architectures and [optimization](https://www.ultralytics.com/glossary/optimization-algorithm) techniques.
- Identify domain-specific challenges that may require specialized [model training tips](../../guides/model-training-tips.md) or [fine-tuning](https://www.ultralytics.com/glossary/fine-tuning) approaches like [transfer learning](https://www.ultralytics.com/glossary/transfer-learning).

For more ideas and inspiration on real-world applications, explore [our guides on practical projects](../../guides/index.md) or check out [Ultralytics Platform](https://platform.ultralytics.com) for streamlined [model training](../../modes/train.md) and [deployment](../../guides/model-deployment-options.md).

## Usage

The Roboflow 100 dataset, including metadata and download links, is available on the official [Roboflow 100 GitHub repository](https://github.com/roboflow/roboflow-100-benchmark). You can access and utilize the dataset directly from there for your benchmarking needs. Once the datasets are downloaded and prepared as shown above, Ultralytics models can be fine-tuned across the entire collection in a single `model.train()` call by passing the list of dataset YAMLs.

## Sample Data and Annotations

Roboflow 100 consists of datasets with diverse images captured from various angles and domains. Below are examples of annotated images included in the RF100 benchmark, showcasing the variety of objects and scenes. Techniques like [data augmentation](https://www.ultralytics.com/glossary/data-augmentation) can further enhance the diversity during training.

<p align="center">
  <img width="640" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/sample-data-annotations.avif" alt="Roboflow 100 sample images with annotations">
</p>

The diversity seen in the Roboflow 100 benchmark represents a significant advancement from traditional benchmarks, which often focus on optimizing a single metric within a limited domain. This comprehensive approach aids in developing more robust and versatile [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models capable of performing well across a multitude of different scenarios.

## Citations and Acknowledgments

If you use the Roboflow 100 dataset in your research or development work, please cite the original paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{rf100benchmark,
            Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
            Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
            Year = {2022},
            Eprint = {arXiv:2211.13523},
            url = {https://arxiv.org/abs/2211.13523}
        }
        ```

We extend our gratitude to the Roboflow team and all contributors for their significant efforts in creating and maintaining the Roboflow 100 dataset as a valuable resource for the computer vision community.

If you are interested in exploring more datasets to enhance your object detection and machine learning projects, feel free to visit [our comprehensive dataset collection](../index.md), which includes a variety of other [detection datasets](../detect/index.md).

## FAQ

### What is the Roboflow 100 dataset, and why is it significant for object detection?

The **Roboflow 100** dataset is a benchmark for [object detection](../../tasks/detect.md) models. It comprises 100 diverse datasets covering domains like healthcare, aerial imagery, and video games. Its significance lies in providing a standardized way to test model adaptability and robustness across a wide range of real-world scenarios, moving beyond traditional, often domain-limited, benchmarks.

### Which domains are covered by the Roboflow 100 dataset?

The **Roboflow 100** dataset spans seven diverse domains, offering unique challenges for [object detection](https://www.ultralytics.com/glossary/object-detection) models:

1.  **Aerial**: 7 datasets (e.g., satellite imagery, drone views).
2.  **Video Games**: 7 datasets (e.g., objects from various game environments).
3.  **Microscopic**: 11 datasets (e.g., cells, particles).
4.  **Underwater**: 5 datasets (e.g., marine life, submerged objects).
5.  **Documents**: 8 datasets (e.g., text regions, form elements).
6.  **Electromagnetic**: 12 datasets (e.g., radar signatures, spectral data visualizations).
7.  **Real World**: 50 datasets (a broad category including everyday objects, scenes, retail, etc.).

This variety makes RF100 an excellent resource for assessing the [generalizability](<https://en.wikipedia.org/wiki/Generalization_(learning)>) of computer vision models.

### What should I include when citing the Roboflow 100 dataset in my research?

When using the Roboflow 100 dataset, please cite the original paper to give credit to the creators. Here is the recommended BibTeX citation:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{rf100benchmark,
            Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
            Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
            Year = {2022},
            Eprint = {arXiv:2211.13523},
            url = {https://arxiv.org/abs/2211.13523}
        }
        ```

For further exploration, consider visiting our [comprehensive dataset collection](../index.md) or browsing other [detection datasets](../detect/index.md) compatible with Ultralytics models.
