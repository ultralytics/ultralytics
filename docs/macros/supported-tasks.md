{% set unsupported = unsupported or [] %}
| Task | Supported |
| :-------------------------------------------- | :-------- |
| [Object Detection](../tasks/detect.md) | {{ "❌" if "detect" in unsupported else "✅" }} |
| [Instance Segmentation](../tasks/segment.md) | {{ "❌" if "segment" in unsupported else "✅" }} |
| [Semantic Segmentation](../tasks/semantic.md) | {{ "❌" if "semantic" in unsupported else "✅" }} |
| [Pose Estimation](../tasks/pose.md) | {{ "❌" if "pose" in unsupported else "✅" }} |
| [OBB Detection](../tasks/obb.md) | {{ "❌" if "obb" in unsupported else "✅" }} |
| [Classification](../tasks/classify.md) | {{ "❌" if "classify" in unsupported else "✅" }} |
| [Depth Estimation](../tasks/depth.md) | {{ "❌" if "depth" in unsupported else "✅" }} |
