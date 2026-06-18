// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
// Backend-agnostic pre/post-processing for the Ultralytics YOLO C++ examples.
// These free functions operate on raw float tensors + shapes, so ONNX Runtime,
// OpenVINO, or any other runtime can share the exact same logic. Header-only.

#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo_types.hpp"

namespace yolo {

// ---- Pre-processing ---------------------------------------------------------

// Letterbox (detection tasks) or center-crop (classification) a BGR image into a
// square `imgsz`, return it as RGB and set `scale` (used to map boxes back).
inline cv::Mat Preprocess(const cv::Mat& image, int imgsz, bool classify, float& scale) {
    if (classify) {
        scale = 1.0f;
        const float r = imgsz / static_cast<float>(std::min(image.cols, image.rows));
        const int rw = static_cast<int>(std::round(image.cols * r));
        const int rh = static_cast<int>(std::round(image.rows * r));
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(rw, rh), 0, 0, cv::INTER_AREA);
        cv::Mat crop = resized(cv::Rect((rw - imgsz) / 2, (rh - imgsz) / 2, imgsz, imgsz)).clone();
        cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
        return crop;
    }
    scale = std::min(imgsz / static_cast<float>(image.cols), imgsz / static_cast<float>(image.rows));
    const int new_w = static_cast<int>(std::round(image.cols * scale));
    const int new_h = static_cast<int>(std::round(image.rows * scale));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));
    cv::Mat out = cv::Mat::zeros(imgsz, imgsz, CV_8UC3);  // top-left letterbox, pad bottom/right
    resized.copyTo(out(cv::Rect(0, 0, new_w, new_h)));
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    return out;
}

// HWC uint8 (RGB) -> CHW float32 normalized to [0, 1].
inline std::vector<float> ToBlob(const cv::Mat& rgb, int imgsz) {
    const int area = imgsz * imgsz;
    std::vector<float> blob(3 * area);
    for (int h = 0; h < imgsz; ++h) {
        for (int w = 0; w < imgsz; ++w) {
            const cv::Vec3b& px = rgb.at<cv::Vec3b>(h, w);
            for (int c = 0; c < 3; ++c) blob[c * area + h * imgsz + w] = px[c] / 255.0f;
        }
    }
    return blob;
}

// ---- Task inference (for backends without a task metadata field) ------------

// Infer the task from output shapes + class-label count. `num_labels` comes from
// the model metadata (e.g. OpenVINO rt_info "labels").
inline Task InferTask(const std::vector<std::vector<int64_t>>& shapes, int num_labels) {
    bool has_4d = false, has_non_4d = false;
    std::vector<int64_t> main;
    for (const auto& s : shapes) {
        if (s.size() == 4) has_4d = true;
        else { has_non_4d = true; if (main.empty()) main = s; }
    }
    if (has_4d && !has_non_4d) return Task::Semantic;  // single [1, nc, H, W]
    if (has_4d && has_non_4d) return Task::Segment;    // dets + proto

    if (main.size() == 2) return Task::Classify;       // [1, N]
    if (main.size() == 3) {
        const int dim1 = static_cast<int>(main[1]);
        const int dim2 = static_cast<int>(main[2]);
        if (dim1 > dim2) {  // end2end [1, 300, attrs]
            if (dim2 == 6) return Task::Detect;
            if (dim2 == 7) return Task::Obb;
            return Task::Pose;  // 6 + nkpt*3
        }
        const int c = dim1;  // grid [1, C, 8400]
        if (c == 4 + num_labels) return Task::Detect;
        if (c == 5 + num_labels) return Task::Obb;
        if (num_labels == 1 && c >= 10 && (c - 5) % 3 == 0) return Task::Pose;
        return Task::Detect;
    }
    return Task::Unknown;
}

// ---- Post-processing --------------------------------------------------------

inline std::vector<Result> PostprocessDetect(const float* data, const std::vector<int64_t>& shape,
                                             float scale, float conf_thr, float iou_thr) {
    std::vector<Result> results;
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);
    const bool end2end = dim1 > dim2;

    if (end2end) {  // [1, 300, 6] = x1,y1,x2,y2,conf,cls (already NMS'd)
        for (int i = 0; i < dim1; ++i) {
            const float* row = data + i * dim2;
            const float conf = row[4];
            if (conf < conf_thr) continue;
            Result r;
            r.class_id = static_cast<int>(row[5]);
            r.confidence = conf;
            r.box = cv::Rect(static_cast<int>(row[0] * inv), static_cast<int>(row[1] * inv),
                             static_cast<int>((row[2] - row[0]) * inv), static_cast<int>((row[3] - row[1]) * inv));
            results.push_back(r);
        }
        return results;
    }

    // Grid [1, 4+nc, 8400]: index the channel-major buffer directly (avoids a full
    // transpose copy and a per-anchor cv::minMaxLoc) and argmax over the class scores.
    const int nc = dim1 - 4;
    const int a = dim2;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    for (int i = 0; i < a; ++i) {
        int best = 0;
        float best_score = data[4 * a + i];
        for (int c = 1; c < nc; ++c) {
            const float s = data[(4 + c) * a + i];
            if (s > best_score) { best_score = s; best = c; }
        }
        if (best_score < conf_thr) continue;
        const float cx = data[i], cy = data[a + i], w = data[2 * a + i], h = data[3 * a + i];
        boxes.emplace_back(static_cast<int>((cx - 0.5f * w) * inv), static_cast<int>((cy - 0.5f * h) * inv),
                           static_cast<int>(w * inv), static_cast<int>(h * inv));
        confidences.push_back(best_score);
        class_ids.push_back(best);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thr, iou_thr, keep);
    for (int idx : keep) {
        Result r;
        r.class_id = class_ids[idx];
        r.confidence = confidences[idx];
        r.box = boxes[idx];
        results.push_back(r);
    }
    return results;
}

inline std::vector<Result> PostprocessPose(const float* data, const std::vector<int64_t>& shape,
                                           float scale, float conf_thr, float iou_thr) {
    std::vector<Result> results;
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);

    if (dim1 > dim2) {  // end2end [1, 300, 6 + nkpt*3]
        const int nkpt = (dim2 - 6) / 3;
        for (int i = 0; i < dim1; ++i) {
            const float* row = data + i * dim2;
            const float conf = row[4];
            if (conf < conf_thr) continue;
            Result r;
            r.class_id = static_cast<int>(row[5]);
            r.confidence = conf;
            r.box = cv::Rect(static_cast<int>(row[0] * inv), static_cast<int>(row[1] * inv),
                             static_cast<int>((row[2] - row[0]) * inv), static_cast<int>((row[3] - row[1]) * inv));
            for (int k = 0; k < nkpt; ++k) {
                const float* kp = row + 6 + k * 3;
                r.keypoints.emplace_back(kp[0] * inv, kp[1] * inv);
                r.keypoint_scores.push_back(kp[2]);
            }
            results.push_back(r);
        }
        return results;
    }

    // Grid [1, 4+1+nkpt*3, 8400]: direct channel-major indexing (no transpose copy).
    const int nkpt = (dim1 - 5) / 3;
    const int a = dim2;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point2f>> kpts;
    std::vector<std::vector<float>> kscores;
    for (int i = 0; i < a; ++i) {
        const float conf = data[4 * a + i];
        if (conf < conf_thr) continue;
        const float cx = data[i], cy = data[a + i], w = data[2 * a + i], h = data[3 * a + i];
        boxes.emplace_back(static_cast<int>((cx - 0.5f * w) * inv), static_cast<int>((cy - 0.5f * h) * inv),
                           static_cast<int>(w * inv), static_cast<int>(h * inv));
        confidences.push_back(conf);
        std::vector<cv::Point2f> kp;
        std::vector<float> ks;
        for (int k = 0; k < nkpt; ++k) {
            kp.emplace_back(data[(5 + k * 3) * a + i] * inv, data[(5 + k * 3 + 1) * a + i] * inv);
            ks.push_back(data[(5 + k * 3 + 2) * a + i]);
        }
        kpts.push_back(std::move(kp));
        kscores.push_back(std::move(ks));
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thr, iou_thr, keep);
    for (int idx : keep) {
        Result r;
        r.class_id = 0;
        r.confidence = confidences[idx];
        r.box = boxes[idx];
        r.keypoints = kpts[idx];
        r.keypoint_scores = kscores[idx];
        results.push_back(r);
    }
    return results;
}

inline std::vector<Result> PostprocessObb(const float* data, const std::vector<int64_t>& shape,
                                          float scale, float conf_thr, float iou_thr) {
    std::vector<Result> results;
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);

    if (dim1 > dim2) {  // end2end [1, 300, 7] = cx,cy,w,h,conf,cls,angle
        for (int i = 0; i < dim1; ++i) {
            const float* row = data + i * dim2;
            const float conf = row[4];
            if (conf < conf_thr) continue;
            const float cx = row[0] * inv, cy = row[1] * inv, w = row[2] * inv, h = row[3] * inv;
            Result r;
            r.class_id = static_cast<int>(row[5]);
            r.confidence = conf;
            r.angle = row[6];
            r.box = cv::Rect(static_cast<int>(cx - 0.5f * w), static_cast<int>(cy - 0.5f * h),
                             static_cast<int>(w), static_cast<int>(h));
            results.push_back(r);
        }
        return results;
    }

    // Grid [1, 4+nc+1, 8400]: direct channel-major indexing (no transpose copy).
    const int nc = dim1 - 5;
    const int a = dim2;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<Result> candidates;
    for (int i = 0; i < a; ++i) {
        int best = 0;
        float best_score = data[4 * a + i];
        for (int c = 1; c < nc; ++c) {
            const float s = data[(4 + c) * a + i];
            if (s > best_score) { best_score = s; best = c; }
        }
        if (best_score < conf_thr) continue;
        const float cx = data[i] * inv, cy = data[a + i] * inv, w = data[2 * a + i] * inv, h = data[3 * a + i] * inv;
        boxes.emplace_back(static_cast<int>(cx - 0.5f * w), static_cast<int>(cy - 0.5f * h),
                           static_cast<int>(w), static_cast<int>(h));
        confidences.push_back(best_score);
        Result r;
        r.class_id = best;
        r.confidence = best_score;
        r.angle = data[(4 + nc) * a + i];
        r.box = boxes.back();
        candidates.push_back(r);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thr, iou_thr, keep);  // axis-aligned proxy NMS
    for (int idx : keep) results.push_back(candidates[idx]);
    return results;
}

inline std::vector<Result> PostprocessSegment(const float* det, const std::vector<int64_t>& det_shape,
                                              const float* protos, const std::vector<int64_t>& proto_shape,
                                              float scale, float conf_thr, float iou_thr, int imgsz,
                                              const cv::Size& orig) {
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(det_shape[1]);
    const int dim2 = static_cast<int>(det_shape[2]);
    const bool end2end = dim1 > dim2;
    const int pc = static_cast<int>(proto_shape[1]);
    const int mh = static_cast<int>(proto_shape[2]);
    const int mw = static_cast<int>(proto_shape[3]);
    cv::Mat proto_mat(pc, mh * mw, CV_32F, const_cast<float*>(protos));

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<cv::Mat> coeffs;

    if (end2end) {  // [1, 300, 6 + pc]
        for (int i = 0; i < dim1; ++i) {
            const float* row = det + i * dim2;
            const float conf = row[4];
            if (conf < conf_thr) continue;
            boxes.emplace_back(static_cast<int>(row[0] * inv), static_cast<int>(row[1] * inv),
                               static_cast<int>((row[2] - row[0]) * inv), static_cast<int>((row[3] - row[1]) * inv));
            confidences.push_back(conf);
            class_ids.push_back(static_cast<int>(row[5]));
            coeffs.push_back(cv::Mat(1, pc, CV_32F, const_cast<float*>(row + 6)).clone());
        }
    } else {  // [1, 4 + nc + pc, 8400]: direct channel-major indexing (no transpose copy).
        const int nc = dim1 - 4 - pc;
        const int a = dim2;
        for (int i = 0; i < a; ++i) {
            int best = 0;
            float best_score = det[4 * a + i];
            for (int c = 1; c < nc; ++c) {
                const float s = det[(4 + c) * a + i];
                if (s > best_score) { best_score = s; best = c; }
            }
            if (best_score < conf_thr) continue;
            const float cx = det[i], cy = det[a + i], w = det[2 * a + i], h = det[3 * a + i];
            boxes.emplace_back(static_cast<int>((cx - 0.5f * w) * inv), static_cast<int>((cy - 0.5f * h) * inv),
                               static_cast<int>(w * inv), static_cast<int>(h * inv));
            confidences.push_back(best_score);
            class_ids.push_back(best);
            cv::Mat coeff(1, pc, CV_32F);
            for (int j = 0; j < pc; ++j) coeff.at<float>(0, j) = det[(4 + nc + j) * a + i];
            coeffs.push_back(coeff);
        }
    }

    std::vector<int> keep;
    if (end2end) {
        keep.resize(boxes.size());
        for (size_t i = 0; i < keep.size(); ++i) keep[i] = static_cast<int>(i);
    } else {
        cv::dnn::NMSBoxes(boxes, confidences, conf_thr, iou_thr, keep);
    }

    const int new_w = static_cast<int>(std::round(orig.width * scale));
    const int new_h = static_cast<int>(std::round(orig.height * scale));

    std::vector<Result> results;
    for (int idx : keep) {
        cv::Mat mask_flat = coeffs[idx] * proto_mat;  // 1 x (mh*mw)
        cv::Mat mask = mask_flat.reshape(1, mh);      // mh x mw
        cv::Mat neg, sig;
        cv::exp(-mask, neg);
        sig = 1.0 / (1.0 + neg);
        cv::Mat full;
        cv::resize(sig, full, cv::Size(imgsz, imgsz), 0, 0, cv::INTER_LINEAR);
        cv::Mat valid = full(cv::Rect(0, 0, std::min(new_w, imgsz), std::min(new_h, imgsz)));
        cv::Mat mask_orig;
        cv::resize(valid, mask_orig, orig, 0, 0, cv::INTER_LINEAR);
        cv::Mat binary = mask_orig > 0.5f;
        cv::Mat cropped = cv::Mat::zeros(orig, CV_8U);
        cv::Rect b = boxes[idx] & cv::Rect(0, 0, orig.width, orig.height);
        if (b.area() > 0) binary(b).copyTo(cropped(b));
        Result r;
        r.class_id = class_ids[idx];
        r.confidence = confidences[idx];
        r.box = boxes[idx];
        r.mask = cropped;
        results.push_back(r);
    }
    return results;
}

inline void PostprocessSemantic(const float* data, const std::vector<int64_t>& shape, float scale, int imgsz,
                                const cv::Size& orig, cv::Mat& out) {
    const int nc = static_cast<int>(shape[1]);
    const int h = static_cast<int>(shape[2]);
    const int w = static_cast<int>(shape[3]);
    const long plane = static_cast<long>(h) * w;
    cv::Mat class_map(h, w, CV_8U);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int best = 0;
            float best_val = data[y * w + x];
            for (int c = 1; c < nc; ++c) {
                const float v = data[c * plane + y * w + x];
                if (v > best_val) { best_val = v; best = c; }
            }
            class_map.at<uchar>(y, x) = static_cast<uchar>(best);
        }
    }
    const int new_w = static_cast<int>(std::round(orig.width * scale));
    const int new_h = static_cast<int>(std::round(orig.height * scale));
    cv::Mat valid = class_map(cv::Rect(0, 0, std::min(new_w, w), std::min(new_h, h)));
    cv::resize(valid, out, orig, 0, 0, cv::INTER_NEAREST);
    (void)imgsz;
}

inline std::vector<Result> PostprocessClassify(const float* data, const std::vector<int64_t>& shape, int topk = 5) {
    const int n = static_cast<int>(shape[1]);
    std::vector<Result> results;
    results.reserve(n);
    for (int i = 0; i < n; ++i) {
        Result r;
        r.class_id = i;
        r.confidence = data[i];
        results.push_back(r);
    }
    std::sort(results.begin(), results.end(),
              [](const Result& a, const Result& b) { return a.confidence > b.confidence; });
    if (static_cast<int>(results.size()) > topk) results.resize(topk);
    return results;
}

}  // namespace yolo
