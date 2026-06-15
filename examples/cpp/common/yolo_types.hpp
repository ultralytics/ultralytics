// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// Shared types for the Ultralytics YOLO C++ examples: the task enum and the
// unified per-detection result. Header-only.

#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace yolo {

// YOLO task type. For ONNX it comes from the model metadata; for backends without
// a task field (e.g. OpenVINO IR) it is inferred from the output shapes.
enum class Task { Detect, Segment, Pose, Obb, Classify, Semantic, Unknown };

inline Task TaskFromString(const std::string& s) {
    if (s == "detect") return Task::Detect;
    if (s == "segment") return Task::Segment;
    if (s == "pose") return Task::Pose;
    if (s == "obb") return Task::Obb;
    if (s == "classify") return Task::Classify;
    if (s == "semantic") return Task::Semantic;
    return Task::Unknown;
}

inline std::string TaskName(Task task) {
    switch (task) {
        case Task::Detect: return "detect";
        case Task::Segment: return "segment";
        case Task::Pose: return "pose";
        case Task::Obb: return "obb";
        case Task::Classify: return "classify";
        case Task::Semantic: return "semantic";
        default: return "unknown";
    }
}

// A single prediction. Fields are populated per task: box (all detection tasks),
// keypoints (pose), mask (segment), angle (obb). Classify uses class_id/confidence.
struct Result {
    int class_id = 0;
    float confidence = 0.0f;
    cv::Rect box;
    float angle = 0.0f;                  // OBB rotation in radians
    std::vector<cv::Point2f> keypoints;  // Pose keypoints, original-image coordinates
    std::vector<float> keypoint_scores;  // Per-keypoint confidence
    cv::Mat mask;                        // Segment: 8-bit binary mask at original image size
};

}  // namespace yolo
