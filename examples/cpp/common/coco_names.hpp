// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
// Shared COCO class-name list for the Ultralytics YOLO C++ examples.
// Header-only fallback for models that do not carry their class names in
// metadata. Models exported by Ultralytics embed `names` directly, so prefer
// reading them from the model when available and use this only as a fallback.

#pragma once

#include <string>
#include <vector>

namespace yolo {

// The 80 COCO dataset class names, in class-id order.
inline const std::vector<std::string>& CocoNames() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush",
    };
    return names;
}

}  // namespace yolo
