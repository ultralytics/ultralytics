# DINOv2: Feature Extractor for BoT-SORT

DINOv2 is an advanced vision transformer model used as a feature extractor in the BoT-SORT tracker for object re-identification (ReID). It enhances tracking performance by providing robust embeddings for matching objects across frames.

## Features

- **High-Quality Embeddings:** Extracts detailed feature representations for accurate object matching.
- **Flexible Token Options:** Supports CLS token or average of patch tokens for feature extraction.
- **Seamless Integration:** Easily configurable in the `botsort.yaml` file.

## Configuration

To enable DINOv2 as the feature extractor in BoT-SORT, update the `model` parameter in the `botsort.yaml` file:

```yaml
with_reid: True
model: dinov2_vits14  # Options: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14u
return_clstoken: True  # Use CLS token (faster) or average of patch tokens (more detailed)
```

## Supported Models

| Model           | Description                  |
|------------------|------------------------------|
| `dinov2_vits14`  | Small vision transformer     |
| `dinov2_vitb14`  | Base vision transformer      |
| `dinov2_vitl14`  | Large vision transformer     |
| `dinov2_vitg14`  | Giant vision transformer     |

## Usage in BoT-SORT

DINOv2 is automatically invoked when `with_reid: True` is set in the tracker configuration. It processes object crops and generates normalized embeddings for ReID.

```python
from ultralytics.trackers.bot_sort import DINOv2ReID

# Initialize DINOv2 for ReID
reid_model = DINOv2ReID(model="dinov2_vits14", device="cuda", return_clstoken=True)

# Extract embeddings
embeddings = reid_model(img, detections)
```

## Applications

- **Object Re-Identification:** Match objects across frames in multi-object tracking.
- **Enhanced Tracking:** Improve tracking accuracy in crowded or occluded scenes.

## References

- [DINOv2 GitHub Repository](https://github.com/facebookresearch/dinov2?tab=readme-ov-file)
- [DinoTool GitHub Repository](https://github.com/mikkoim/dinotool)

For more details, refer to the [BoT-SORT documentation](../reference/trackers/bot_sort.md).
