// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// Shared color helper for the Ultralytics YOLO C++ examples.
// Header-only: add this folder (examples/cpp/common) to your include path and
// `#include "yolo_colors.hpp"`.

#pragma once

#include <vector>
#include <cstdlib>
#include <opencv2/core.hpp>

namespace yolo {

// Ultralytics color palette (https://docs.ultralytics.com/), returned as BGR for OpenCV.
// Indexed by class id so each class always gets the same, distinct color.
inline cv::Scalar Color(int class_id) {
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0x38, 0x38, 0xFF), cv::Scalar(0x97, 0x9D, 0xFF), cv::Scalar(0x1F, 0x70, 0xFF),
        cv::Scalar(0x1D, 0xB2, 0xFF), cv::Scalar(0x31, 0xD2, 0xCF), cv::Scalar(0x0A, 0xF9, 0x48),
        cv::Scalar(0x17, 0xCC, 0x92), cv::Scalar(0x86, 0xDB, 0x3D), cv::Scalar(0x34, 0x93, 0x1A),
        cv::Scalar(0xBB, 0xD4, 0x00), cv::Scalar(0xA8, 0x99, 0x2C), cv::Scalar(0xFF, 0xC2, 0x00),
        cv::Scalar(0x93, 0x45, 0x34), cv::Scalar(0xFF, 0x73, 0x64), cv::Scalar(0xEC, 0x18, 0x00),
        cv::Scalar(0xFF, 0x38, 0x84), cv::Scalar(0x85, 0x00, 0x52), cv::Scalar(0xFF, 0x38, 0xCB),
        cv::Scalar(0xC8, 0x95, 0xFF), cv::Scalar(0xC7, 0x37, 0xFF),
    };
    return palette[std::abs(class_id) % static_cast<int>(palette.size())];
}

}  // namespace yolo
