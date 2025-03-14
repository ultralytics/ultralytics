# Sesame Plant Dataset for Computer Vision in Agriculture

## Overview

Computer vision has numerous applications, and agriculture is no exception. Integrating computer vision into agriculture offers many advantages, including:

- **Enhanced precision** in identifying plant diseases, pests, and growth anomalies.
- **Automation** of crop monitoring and management tasks.
- **Increased efficiency** in farming processes by reducing manual labor.
- **Real-time data analysis** for better decision-making in farming practices.

This dataset focuses on **Sesame Plants**, providing labeled images for training and validation. It can be used for tasks like plant identification, health assessment, and yield estimation.

---

## Dataset Description

The dataset was manually captured by my **EJAZTECH.AI** team. It contains high-quality images of sesame plants with their corresponding annotations.

### Dataset Structure

The dataset is organized into two subsets:

1. **Training Set**:

    - **Images**: 262
    - **Annotations**: Corresponding YOLO format labels.

2. **Validation Set**:
    - **Images**: 31
    - **Annotations**: Corresponding YOLO format labels.

### Annotation Format

The annotations are provided in the YOLO `.txt` format. Each annotation file contains:

- Class ID
- Bounding box coordinates in normalized format (x_center, y_center, width, height).

These annotations are fully compatible with YOLO-based frameworks, including Ultralytics.

---

## Key Features of the Dataset

- **Manually Captured**: Ensures high-quality, reliable images.
- **Domain-Specific**: Focused exclusively on sesame plants, a crucial crop in agriculture.
- **Well-Structured**: Organized for seamless integration into training workflows like YOLOv8.

---

## Usage

### Applications

This dataset can be used for:

- Identifying sesame plants in diverse environments.
- assessing plant health for diseases or pest infestations.
- Developing automation systems for sesame farming processes.
- Building AI models for general plant classification and detection tasks.

### Credits

The dataset was created and curated by the my EJAZTECH.AI team. All images were manually captured in agricultural fields to ensure quality and relevance.
