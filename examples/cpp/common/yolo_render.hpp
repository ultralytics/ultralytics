// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
// Shared result rendering for the Ultralytics YOLO C++ examples. Draws the
// detections (or the semantic map) onto an image exactly like the Python
// Annotator and prints them to the console, so every example's `main` shares
// one identical render path. Header-only: add examples/cpp/common to the path.

#pragma once

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "yolo_cli.hpp"
#include "yolo_draw.hpp"
#include "yolo_types.hpp"

namespace yolo {

// Annotate `canvas` for `task` (drawing `results`, or `semantic` for semantic
// segmentation) and print each detection to stdout.
inline void RenderAndPrint(cv::Mat& canvas, Task task, const std::vector<Result>& results,
                           const std::vector<std::string>& names, const cv::Mat& semantic) {
    switch (task) {
        case Task::Semantic:
            DrawSemantic(canvas, semantic);
            std::cout << "semantic map rendered (" << names.size() << " classes)" << std::endl;
            break;
        case Task::Segment:
        case Task::Detect:
        case Task::Pose: {
            for (const Result& r : results) {
                const std::string name = NameOf(names, r.class_id);
                if (!r.mask.empty()) DrawMask(canvas, r.mask, r.class_id);
                if (!r.keypoints.empty()) DrawPose(canvas, r.keypoints, r.keypoint_scores);
                DrawBox(canvas, r.box, Label(name, r.confidence), r.class_id);
                std::cout << name << " " << std::fixed << std::setprecision(2) << r.confidence << " box=["
                          << r.box.x << ", " << r.box.y << ", " << r.box.width << ", " << r.box.height << "]"
                          << std::endl;
            }
            break;
        }
        case Task::Obb: {
            for (const Result& r : results) {
                const std::string name = NameOf(names, r.class_id);
                cv::RotatedRect rr(cv::Point2f(r.box.x + r.box.width * 0.5f, r.box.y + r.box.height * 0.5f),
                                   cv::Size2f(static_cast<float>(r.box.width), static_cast<float>(r.box.height)),
                                   r.angle * 180.0f / static_cast<float>(CV_PI));
                DrawObb(canvas, rr, Label(name, r.confidence), r.class_id);
                std::cout << name << " " << std::fixed << std::setprecision(2) << r.confidence << " angle="
                          << r.angle << std::endl;
            }
            break;
        }
        case Task::Classify: {
            int y = 28;
            for (const Result& r : results) {
                std::ostringstream label;
                label << NameOf(names, r.class_id) << " " << std::fixed << std::setprecision(2) << r.confidence;
                cv::putText(canvas, label.str(), {12, y}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 0, 0}, 3, cv::LINE_AA);
                cv::putText(canvas, label.str(), {12, y}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {255, 255, 255}, 1, cv::LINE_AA);
                std::cout << label.str() << std::endl;
                y += 28;
            }
            break;
        }
        default:
            std::cerr << "[yolo] task '" << TaskName(task) << "' is not supported." << std::endl;
    }
}

}  // namespace yolo
