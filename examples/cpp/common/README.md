# Shared Helpers for the C++ Examples

Header-only utilities shared across the [Ultralytics YOLO](https://docs.ultralytics.com/) C++ examples. They have no dependencies beyond [OpenCV](https://opencv.org/) and the C++ standard library, so an example can use them by adding this folder to its include path:

```cmake
target_include_directories(your_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
```

| Header             | Provides                                                                                              |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| `yolo_colors.hpp`  | `yolo::Color(class_id)` — the Ultralytics color palette as OpenCV `cv::Scalar` (BGR), indexed by class. |
| `coco_names.hpp`   | `yolo::CocoNames()` — the 80 COCO class names, a fallback for models without baked-in `names` metadata. |

Models exported by Ultralytics embed their class `names` (and `imgsz`, `task`, `stride`) in the model metadata, so prefer reading names from the model directly and use `coco_names.hpp` only as a fallback.
