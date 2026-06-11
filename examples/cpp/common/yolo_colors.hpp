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
    // Mirrors ultralytics.utils.plotting.Colors (hex palette), converted to BGR.
    static const std::vector<cv::Scalar> palette = {
        cv::Scalar(0xFF, 0x2A, 0x04), cv::Scalar(0xEB, 0xDB, 0x0B), cv::Scalar(0xF3, 0xF3, 0xF3),
        cv::Scalar(0xB7, 0xDF, 0x00), cv::Scalar(0x68, 0x1F, 0x11), cv::Scalar(0xDD, 0x6F, 0xFF),
        cv::Scalar(0x4F, 0x44, 0xFF), cv::Scalar(0x00, 0xED, 0xCC), cv::Scalar(0x44, 0xF3, 0x00),
        cv::Scalar(0xFF, 0x00, 0xBD), cv::Scalar(0xFF, 0xB4, 0x00), cv::Scalar(0xBA, 0x00, 0xDD),
        cv::Scalar(0xFF, 0xFF, 0x00), cv::Scalar(0x00, 0xC0, 0x26), cv::Scalar(0xB3, 0xFF, 0x01),
        cv::Scalar(0xFF, 0x24, 0x7D), cv::Scalar(0x68, 0x00, 0x7B), cv::Scalar(0x6C, 0x1B, 0xFF),
        cv::Scalar(0x2F, 0x6D, 0xFC), cv::Scalar(0x0B, 0xFF, 0xA2),
    };
    return palette[std::abs(class_id) % static_cast<int>(palette.size())];
}

}  // namespace yolo
