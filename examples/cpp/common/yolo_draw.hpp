// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// Shared drawing helper for the Ultralytics YOLO C++ examples. Mirrors the
// Python Annotator.box_label() so every example annotates detections identically:
// a box plus a filled label with contrasting text. Header-only: add
// examples/cpp/common to your include path and `#include "yolo_draw.hpp"`.

#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <opencv2/imgproc.hpp>

#include "yolo_colors.hpp"

namespace yolo {

// Text color for a given box color, mirroring Annotator.get_txt_color():
// a fixed set of light palette colors use dark text; everything else uses white.
inline cv::Scalar TextColor(const cv::Scalar& bg) {
    // "dark_colors" in plotting.py: backgrounds that read better with dark text (BGR).
    static const int dark_bgr[][3] = {
        {235, 219, 11}, {243, 243, 243}, {183, 223, 0}, {221, 111, 255}, {0, 237, 204},
        {68, 243, 0}, {255, 255, 0}, {179, 255, 1}, {11, 255, 162},
    };
    const int b = static_cast<int>(bg[0]), g = static_cast<int>(bg[1]), r = static_cast<int>(bg[2]);
    for (const auto& c : dark_bgr) {
        if (c[0] == b && c[1] == g && c[2] == r) return cv::Scalar(104, 31, 17);
    }
    return cv::Scalar(255, 255, 255);
}

// Line width derived from image size, matching Annotator's default `lw`.
inline int LineWidth(const cv::Mat& image) {
    return std::max(static_cast<int>(std::round((image.rows + image.cols + image.channels()) / 2.0 * 0.003)), 2);
}

// Draw a detection box for `class_id` (color from the Ultralytics palette) and an
// optional filled label, exactly like Annotator.box_label() in the Python package.
inline void DrawBox(cv::Mat& image, const cv::Rect& box, const std::string& label, int class_id) {
    const cv::Scalar color = Color(class_id);
    const int lw = LineWidth(image);
    const double sf = lw / 3.0;       // font scale
    const int tf = std::max(lw - 1, 1);  // font thickness

    cv::Point p1(box.x, box.y);
    cv::rectangle(image, p1, cv::Point(box.x + box.width, box.y + box.height), color, lw, cv::LINE_AA);

    if (label.empty()) return;

    int baseline = 0;
    const cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, sf, tf, &baseline);
    const int w = ts.width;
    const int h = ts.height + 3;  // pad text height, like the Python code
    const bool outside = p1.y >= h;
    if (p1.x > image.cols - w) p1.x = image.cols - w;  // keep the label on-screen

    const cv::Point p2(p1.x + w, outside ? p1.y - h : p1.y + h);
    cv::rectangle(image, p1, p2, color, cv::FILLED, cv::LINE_AA);
    cv::putText(image, label, cv::Point(p1.x, outside ? p1.y - 2 : p1.y + h - 1),
                cv::FONT_HERSHEY_SIMPLEX, sf, TextColor(color), tf, cv::LINE_AA);
}

// Format a "<name> <conf>" label the same way the Python package does (2 decimals).
inline std::string Label(const std::string& name, float confidence) {
    int pct = static_cast<int>(std::round(confidence * 100.0f));
    return name + " " + std::to_string(pct / 100) + "." + (pct % 100 < 10 ? "0" : "") + std::to_string(pct % 100);
}

// Blend an instance mask (8-bit, image size) over the image in the class color.
inline void DrawMask(cv::Mat& image, const cv::Mat& mask, int class_id, float alpha = 0.5f) {
    if (mask.empty()) return;
    const cv::Scalar color = Color(class_id);
    cv::Mat colored(image.size(), CV_8UC3, color);
    cv::Mat blended;
    cv::addWeighted(image, 1.0f - alpha, colored, alpha, 0.0, blended);
    blended.copyTo(image, mask);
}

// Draw 17 COCO keypoints and the skeleton for one pose detection, using the same
// pose palette as the Python Annotator (limb and keypoint colors in BGR).
inline void DrawPose(cv::Mat& image, const std::vector<cv::Point2f>& kpts,
                     const std::vector<float>& scores, float kpt_threshold = 0.5f) {
    static const int kSkeleton[19][2] = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8},
        {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6},
    };
    // pose_palette colors used by Ultralytics, in BGR.
    const cv::Scalar kBlue(255, 153, 51), kMagenta(255, 51, 255), kOrange(0, 128, 255), kGreen(0, 255, 0);
    static const cv::Scalar kLimbColor[19] = {kBlue, kBlue, kBlue, kBlue, kMagenta, kMagenta, kMagenta,
                                              kOrange, kOrange, kOrange, kOrange, kOrange, kGreen, kGreen,
                                              kGreen, kGreen, kGreen, kGreen, kGreen};
    static const cv::Scalar kKptColor[17] = {kGreen, kGreen, kGreen, kGreen, kGreen, kOrange, kOrange, kOrange,
                                             kOrange, kOrange, kOrange, kBlue, kBlue, kBlue, kBlue, kBlue, kBlue};

    for (int i = 0; i < 19; ++i) {
        const int a = kSkeleton[i][0], b = kSkeleton[i][1];
        if (a < static_cast<int>(kpts.size()) && b < static_cast<int>(kpts.size()) &&
            scores[a] > kpt_threshold && scores[b] > kpt_threshold) {
            cv::line(image, kpts[a], kpts[b], kLimbColor[i], 2, cv::LINE_AA);
        }
    }
    for (size_t i = 0; i < kpts.size(); ++i) {
        if (scores[i] > kpt_threshold)
            cv::circle(image, kpts[i], 4, i < 17 ? kKptColor[i] : kGreen, -1, cv::LINE_AA);
    }
}

// Draw a rotated (oriented) bounding box plus a filled label.
inline void DrawObb(cv::Mat& image, const cv::RotatedRect& rect, const std::string& label, int class_id) {
    const cv::Scalar color = Color(class_id);
    const int lw = LineWidth(image);
    cv::Point2f pts[4];
    rect.points(pts);
    for (int i = 0; i < 4; ++i) cv::line(image, pts[i], pts[(i + 1) % 4], color, lw, cv::LINE_AA);

    if (label.empty()) return;
    // Anchor the label at the top-most corner.
    cv::Point2f top = pts[0];
    for (int i = 1; i < 4; ++i) if (pts[i].y < top.y) top = pts[i];
    DrawBox(image, cv::Rect(static_cast<int>(top.x), static_cast<int>(top.y), 0, 0), label, class_id);
}

// Color a per-pixel class map (CV_8U or CV_32S, image size) and blend it over the image.
inline void DrawSemantic(cv::Mat& image, const cv::Mat& class_map, float alpha = 0.45f) {
    if (class_map.empty()) return;
    cv::Mat colored(image.size(), CV_8UC3);
    for (int y = 0; y < class_map.rows; ++y) {
        for (int x = 0; x < class_map.cols; ++x) {
            const int c = class_map.type() == CV_32S ? class_map.at<int>(y, x) : class_map.at<uchar>(y, x);
            const cv::Scalar s = Color(c);
            colored.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(s[0]), static_cast<uchar>(s[1]), static_cast<uchar>(s[2]));
        }
    }
    cv::addWeighted(image, 1.0f - alpha, colored, alpha, 0.0, image);
}

}  // namespace yolo
