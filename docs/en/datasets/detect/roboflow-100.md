---
comments: true
description: Explore the Roboflow 100 dataset featuring 100 diverse datasets designed to test object detection models across various domains, from healthcare to video games.
keywords: Roboflow 100, Ultralytics, object detection, dataset, benchmarking, machine learning, computer vision, diverse datasets, model evaluation
---

# Roboflow 100 Dataset

Roboflow 100, developed by [Roboflow](https://roboflow.com/?ref=ultralytics) and sponsored by Intel, is a groundbreaking [object detection](../../tasks/detect.md) benchmark. It includes 100 diverse datasets sampled from over 90,000 public datasets. This benchmark is designed to test the adaptability of models to various domains, including healthcare, aerial imagery, and video games.

<p align="center">
  <img width="640" src="https://user-images.githubusercontent.com/15908060/202452898-9ca6b8f7-4805-4e8e-949a-6e080d7b94d2.jpg" alt="Roboflow 100 Overview">
</p>

## Key Features

- Includes 100 datasets across seven domains: Aerial, Video games, Microscopic, Underwater, Documents, Electromagnetic, and Real World.
- The benchmark comprises 224,714 images across 805 classes, thanks to over 11,170 hours of labeling efforts.
- All images are resized to 640x640 pixels, with a focus on eliminating class ambiguity and filtering out underrepresented classes.
- Annotations include bounding boxes for objects, making it suitable for [training](../../modes/train.md) and evaluating object detection models.

## Dataset Structure

The Roboflow 100 dataset is organized into seven categories, each with a distinct set of datasets, images, and classes:

- **Aerial**: Consists of 7 datasets with a total of 9,683 images, covering 24 distinct classes.
- **Video Games**: Includes 7 datasets, featuring 11,579 images across 88 classes.
- **Microscopic**: Comprises 11 datasets with 13,378 images, spanning 28 classes.
- **Underwater**: Contains 5 datasets, encompassing 18,003 images in 39 classes.
- **Documents**: Consists of 8 datasets with 24,813 images, divided into 90 classes.
- **Electromagnetic**: Made up of 12 datasets, totaling 36,381 images in 41 classes.
- **Real World**: The largest category with 50 datasets, offering 110,615 images across 495 classes.

This structure enables a diverse and extensive testing ground for object detection models, reflecting real-world application scenarios.

## Benchmarking

Dataset benchmarking evaluates machine learning model performance on specific datasets using standardized metrics like accuracy, mean average precision and F1-score.

!!! Tip "Benchmarking"

    Benchmarking results will be stored in "ultralytics-benchmarks/evaluation.txt"

!!! Example "Benchmarking example"

    === "Python"

        ```python
        import os
        import shutil
        from pathlib import Path

        from ultralytics.utils.benchmarks import RF100Benchmark

        # Initialize RF100Benchmark and set API key
        benchmark = RF100Benchmark()
        benchmark.set_key(api_key="YOUR_ROBOFLOW_API_KEY")

        # Parse dataset and define file paths
        names, cfg_yamls = benchmark.parse_dataset()
        val_log_file = Path("ultralytics-benchmarks") / "validation.txt"
        eval_log_file = Path("ultralytics-benchmarks") / "evaluation.txt"

        # Run benchmarks on each dataset in RF100
        for ind, path in enumerate(cfg_yamls):
            path = Path(path)
            if path.exists():
                # Fix YAML file and run training
                benchmark.fix_yaml(str(path))
                os.system(f"yolo detect train data={path} model=yolov8s.pt epochs=1 batch=16")

                # Run validation and evaluate
                os.system(f"yolo detect val data={path} model=runs/detect/train/weights/best.pt > {val_log_file} 2>&1")
                benchmark.evaluate(str(path), str(val_log_file), str(eval_log_file), ind)

                # Remove the 'runs' directory
                runs_dir = Path.cwd() / "runs"
                shutil.rmtree(runs_dir)
            else:
                print("YAML file path does not exist")
                continue

        print("RF100 Benchmarking completed!")
        ```

## Applications

Roboflow 100 is invaluable for various applications related to computer vision and deep learning. Researchers and engineers can use this benchmark to:

- Evaluate the performance of object detection models in a multi-domain context.
- Test the adaptability of models to real-world scenarios beyond common object recognition.
- Benchmark the capabilities of object detection models across diverse datasets, including those in healthcare, aerial imagery, and video games.

For more ideas and inspiration on real-world applications, be sure to check out [our guides on real-world projects](../../guides/index.md).

## Usage

The Roboflow 100 dataset is available on both [GitHub](https://github.com/roboflow/roboflow-100-benchmark) and [Roboflow Universe](https://universe.roboflow.com/roboflow-100).

You can access it directly from the Roboflow 100 GitHub repository. In addition, on Roboflow Universe, you have the flexibility to download individual datasets by simply clicking the export button within each dataset.

## Sample Data and Annotations

Roboflow 100 consists of datasets with diverse images and videos captured from various angles and domains. Here's a look at examples of annotated images in the RF100 benchmark.

<p align="center">
  <img width="640" src="https://blog.roboflow.com/content/images/2022/11/image-2.png" alt="Sample Data and Annotations">
</p>

The diversity in the Roboflow 100 benchmark that can be seen above is a significant advancement from traditional benchmarks which often focus on optimizing a single metric within a limited domain.

## Citations and Acknowledgments

If you use the Roboflow 100 dataset in your research or development work, please cite the following paper:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{2211.13523,
            Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
            Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
            Eprint = {arXiv:2211.13523},
        }
        ```

Our thanks go to the Roboflow team and all the contributors for their hard work in creating and sustaining the Roboflow 100 dataset.

If you are interested in exploring more datasets to enhance your object detection and machine learning projects, feel free to visit [our comprehensive dataset collection](../index.md).

## FAQ

### What is the Roboflow 100 dataset, and why is it significant for object detection?

The **Roboflow 100** dataset, developed by [Roboflow](https://roboflow.com/?ref=ultralytics) and sponsored by Intel, is a crucial [object detection](../../tasks/detect.md) benchmark. It features 100 diverse datasets from over 90,000 public datasets, covering domains such as healthcare, aerial imagery, and video games. This diversity ensures that models can adapt to various real-world scenarios, enhancing their robustness and performance.

### How can I use the Roboflow 100 dataset for benchmarking my object detection models?

To use the Roboflow 100 dataset for benchmarking, you can implement the RF100Benchmark class from the Ultralytics library. Here's a brief example:

!!! Example "Benchmarking example"

    === "Python"
    
        ```python
        import os
        import shutil
        from pathlib import Path

        from ultralytics.utils.benchmarks import RF100Benchmark

        # Initialize RF100Benchmark and set API key
        benchmark = RF100Benchmark()
        benchmark.set_key(api_key="YOUR_ROBOFLOW_API_KEY")

        # Parse dataset and define file paths
        names, cfg_yamls = benchmark.parse_dataset()
        val_log_file = Path("ultralytics-benchmarks") / "validation.txt"
        eval_log_file = Path("ultralytics-benchmarks") / "evaluation.txt"

        # Run benchmarks on each dataset in RF100
        for ind, path in enumerate(cfg_yamls):
            path = Path(path)
            if path.exists():
                # Fix YAML file and run training
                benchmark.fix_yaml(str(path))
                os.system(f"yolo detect train data={path} model=yolov8s.pt epochs=1 batch=16")

                # Run validation and evaluate
                os.system(f"yolo detect val data={path} model=runs/detect/train/weights/best.pt > {val_log_file} 2>&1")
                benchmark.evaluate(str(path), str(val_log_file), str(eval_log_file), ind)

                # Remove 'runs' directory
                runs_dir = Path.cwd() / "runs"
                shutil.rmtree(runs_dir)
            else:
                print("YAML file path does not exist")
                continue

        print("RF100 Benchmarking completed!")
        ```

### Which domains are covered by the Roboflow 100 dataset?

The **Roboflow 100** dataset spans seven domains, each providing unique challenges and applications for object detection models:

1. **Aerial**: 7 datasets, 9,683 images, 24 classes
2. **Video Games**: 7 datasets, 11,579 images, 88 classes
3. **Microscopic**: 11 datasets, 13,378 images, 28 classes
4. **Underwater**: 5 datasets, 18,003 images, 39 classes
5. **Documents**: 8 datasets, 24,813 images, 90 classes
6. **Electromagnetic**: 12 datasets, 36,381 images, 41 classes
7. **Real World**: 50 datasets, 110,615 images, 495 classes

This setup allows for extensive and varied testing of models across different real-world applications.

### How do I access and download the Roboflow 100 dataset?

The **Roboflow 100** dataset is accessible on [GitHub](https://github.com/roboflow/roboflow-100-benchmark) and [Roboflow Universe](https://universe.roboflow.com/roboflow-100). You can download the entire dataset from GitHub or select individual datasets on Roboflow Universe using the export button.

### What should I include when citing the Roboflow 100 dataset in my research?

When using the Roboflow 100 dataset in your research, ensure to properly cite it. Here is the recommended citation:

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{2211.13523,
            Author = {Floriana Ciaglia and Francesco Saverio Zuppichini and Paul Guerrie and Mark McQuade and Jacob Solawetz},
            Title = {Roboflow 100: A Rich, Multi-Domain Object Detection Benchmark},
            Eprint = {arXiv:2211.13523},
        }
        ```

For more details, you can refer to our [comprehensive dataset collection](../index.md).
