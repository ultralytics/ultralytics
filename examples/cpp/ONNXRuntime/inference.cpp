// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <regex>

#include "coco_names.hpp"

namespace yolo {

Task TaskFromString(const std::string& s) {
    if (s == "detect") return Task::Detect;
    if (s == "segment") return Task::Segment;
    if (s == "pose") return Task::Pose;
    if (s == "obb") return Task::Obb;
    if (s == "classify") return Task::Classify;
    if (s == "semantic") return Task::Semantic;
    return Task::Unknown;
}

std::string TaskName(Task task) {
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

Predictor::Predictor(const Config& config) : config_(config) {
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(1);
#ifdef USE_CUDA
    if (config_.cuda) {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
    }
#endif

    session_ = new Ort::Session(env_, config_.model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < session_->GetInputCount(); ++i) {
        input_name_storage_.emplace_back(session_->GetInputNameAllocated(i, allocator).get());
    }
    for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
        output_name_storage_.emplace_back(session_->GetOutputNameAllocated(i, allocator).get());
    }
    for (const auto& s : input_name_storage_) input_names_.push_back(s.c_str());
    for (const auto& s : output_name_storage_) output_names_.push_back(s.c_str());

    load_metadata(allocator);
}

Predictor::~Predictor() { delete session_; }

void Predictor::load_metadata(Ort::AllocatorWithDefaultOptions& allocator) {
    Ort::ModelMetadata metadata = session_->GetModelMetadata();

    // Task (Ultralytics bakes "task" into every export).
    auto task_meta = metadata.LookupCustomMetadataMapAllocated("task", allocator);
    task_ = task_meta ? TaskFromString(task_meta.get()) : Task::Detect;

    // Input size, e.g. "[640, 640]".
    auto imgsz_meta = metadata.LookupCustomMetadataMapAllocated("imgsz", allocator);
    if (imgsz_meta) {
        std::string s = imgsz_meta.get();
        std::smatch m;
        if (std::regex_search(s, m, std::regex("\\d+"))) imgsz_ = std::stoi(m.str());
    }

    // Class names, e.g. "{0: 'person', 1: 'bicycle', ...}".
    auto names_meta = metadata.LookupCustomMetadataMapAllocated("names", allocator);
    names_.clear();
    if (names_meta) {
        std::string s = names_meta.get();
        std::regex item("(\\d+)\\s*:\\s*['\"]([^'\"]*)['\"]");
        for (std::sregex_iterator it(s.begin(), s.end(), item), end; it != end; ++it) {
            int idx = std::stoi((*it)[1].str());
            if (static_cast<int>(names_.size()) <= idx) names_.resize(idx + 1);
            names_[idx] = (*it)[2].str();
        }
    }
    if (names_.empty()) names_ = CocoNames();
}

cv::Mat Predictor::preprocess(const cv::Mat& image, float& scale) {
    // Classification uses a center crop (resize shortest side, crop the middle), not a letterbox.
    if (task_ == Task::Classify) {
        scale = 1.0f;
        const float r = imgsz_ / static_cast<float>(std::min(image.cols, image.rows));
        int rw = static_cast<int>(std::round(image.cols * r));
        int rh = static_cast<int>(std::round(image.rows * r));
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(rw, rh), 0, 0, cv::INTER_AREA);  // antialiased, matches torchvision
        cv::Mat crop = resized(cv::Rect((rw - imgsz_) / 2, (rh - imgsz_) / 2, imgsz_, imgsz_)).clone();
        cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
        return crop;
    }

    scale = std::min(imgsz_ / static_cast<float>(image.cols), imgsz_ / static_cast<float>(image.rows));
    int new_w = static_cast<int>(std::round(image.cols * scale));
    int new_h = static_cast<int>(std::round(image.rows * scale));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    // Letterbox into a square: top-left aligned, pad bottom/right (matches scale-only un-mapping).
    cv::Mat out = cv::Mat::zeros(imgsz_, imgsz_, CV_8UC3);
    resized.copyTo(out(cv::Rect(0, 0, new_w, new_h)));
    cv::cvtColor(out, out, cv::COLOR_BGR2RGB);
    return out;
}

std::vector<Result> Predictor::predict(const cv::Mat& image, cv::Mat& semantic) {
    float scale = 1.0f;
    cv::Mat input = preprocess(image, scale);

    // HWC uint8 -> CHW float32 normalized to [0, 1].
    const int area = imgsz_ * imgsz_;
    std::vector<float> blob(3 * area);
    for (int h = 0; h < imgsz_; ++h) {
        for (int w = 0; w < imgsz_; ++w) {
            const cv::Vec3b& px = input.at<cv::Vec3b>(h, w);
            for (int c = 0; c < 3; ++c) blob[c * area + h * imgsz_ + w] = px[c] / 255.0f;
        }
    }

    std::array<int64_t, 4> input_shape{1, 3, imgsz_, imgsz_};
    Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory, blob.data(), blob.size(),
                                                              input_shape.data(), input_shape.size());

    auto outputs = session_->Run(run_options_, input_names_.data(), &input_tensor, 1,
                                 output_names_.data(), output_names_.size());

    // Identify the primary output (3D for detection-style, 2D for classify) and the
    // 4D proto/semantic tensor when present, regardless of output ordering.
    int main_idx = 0, aux_idx = -1;
    for (size_t i = 0; i < outputs.size(); ++i) {
        const size_t rank = outputs[i].GetTensorTypeAndShapeInfo().GetShape().size();
        if (rank == 4) aux_idx = static_cast<int>(i);
        else main_idx = static_cast<int>(i);
    }
    std::vector<int64_t> shape = outputs[main_idx].GetTensorTypeAndShapeInfo().GetShape();
    const float* data = outputs[main_idx].GetTensorData<float>();

    switch (task_) {
        case Task::Detect:
            return postprocess_detect(data, shape, scale);
        case Task::Classify:
            return postprocess_classify(data, shape);
        case Task::Pose:
            return postprocess_pose(data, shape, scale);
        case Task::Obb:
            return postprocess_obb(data, shape, scale);
        case Task::Segment: {
            if (aux_idx < 0) return {};
            std::vector<int64_t> pshape = outputs[aux_idx].GetTensorTypeAndShapeInfo().GetShape();
            const float* pdata = outputs[aux_idx].GetTensorData<float>();
            return postprocess_segment(data, shape, pdata, pshape, scale, image.size());
        }
        case Task::Semantic:
            postprocess_semantic(data, shape, scale, image.size(), semantic);
            return {};
        default:
            std::cerr << "[yolo] task '" << TaskName(task_) << "' is not supported." << std::endl;
            return {};
    }
}

std::vector<Result> Predictor::postprocess_detect(const float* data, const std::vector<int64_t>& shape, float scale) {
    std::vector<Result> results;
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);

    // End2end (YOLO26): [1, 300, 6] -> rows already decoded & NMS'd. Grid (v8/11): [1, 4+nc, 8400].
    const bool end2end = dim1 > dim2;

    if (end2end) {
        const int num = dim1, step = dim2;  // 300 x 6
        for (int i = 0; i < num; ++i) {
            const float* row = data + i * step;
            const float conf = row[4];
            if (conf < config_.conf) continue;  // rows are sorted by confidence
            const int x1 = static_cast<int>(row[0] * inv);
            const int y1 = static_cast<int>(row[1] * inv);
            const int x2 = static_cast<int>(row[2] * inv);
            const int y2 = static_cast<int>(row[3] * inv);
            Result r;
            r.class_id = static_cast<int>(row[5]);
            r.confidence = conf;
            r.box = cv::Rect(x1, y1, x2 - x1, y2 - y1);
            results.push_back(r);
        }
        return results;
    }

    // Grid: transpose [C, A] -> [A, C], argmax over class scores, then NMS.
    const int channels = dim1, anchors = dim2;
    const int nc = channels - 4;
    cv::Mat raw(channels, anchors, CV_32F, const_cast<float*>(data));
    cv::Mat t = raw.t();

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    for (int i = 0; i < anchors; ++i) {
        const float* row = t.ptr<float>(i);
        cv::Mat scores(1, nc, CV_32F, const_cast<float*>(row + 4));
        cv::Point cls;
        double max_score;
        cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &cls);
        if (max_score < config_.conf) continue;
        const float cx = row[0], cy = row[1], w = row[2], h = row[3];
        boxes.emplace_back(static_cast<int>((cx - 0.5f * w) * inv), static_cast<int>((cy - 0.5f * h) * inv),
                           static_cast<int>(w * inv), static_cast<int>(h * inv));
        confidences.push_back(static_cast<float>(max_score));
        class_ids.push_back(cls.x);
    }

    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, config_.conf, config_.iou, keep);
    for (int idx : keep) {
        Result r;
        r.class_id = class_ids[idx];
        r.confidence = confidences[idx];
        r.box = boxes[idx];
        results.push_back(r);
    }
    return results;
}

std::vector<Result> Predictor::postprocess_classify(const float* data, const std::vector<int64_t>& shape) {
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
    if (results.size() > 5) results.resize(5);  // top-5
    return results;
}

std::vector<Result> Predictor::postprocess_pose(const float* data, const std::vector<int64_t>& shape, float scale) {
    std::vector<Result> results;
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);
    const bool end2end = dim1 > dim2;

    if (end2end) {
        const int num = dim1, step = dim2;  // [1, 300, 6 + nkpt*3]
        const int nkpt = (step - 6) / 3;
        for (int i = 0; i < num; ++i) {
            const float* row = data + i * step;
            const float conf = row[4];
            if (conf < config_.conf) continue;
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

    // Grid: [1, 4 + 1 + nkpt*3, 8400]
    const int channels = dim1, anchors = dim2;
    const int nkpt = (channels - 5) / 3;
    cv::Mat raw(channels, anchors, CV_32F, const_cast<float*>(data));
    cv::Mat t = raw.t();
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point2f>> kpts;
    std::vector<std::vector<float>> kpt_scores;
    for (int i = 0; i < anchors; ++i) {
        const float* row = t.ptr<float>(i);
        const float conf = row[4];
        if (conf < config_.conf) continue;
        const float cx = row[0], cy = row[1], w = row[2], h = row[3];
        boxes.emplace_back(static_cast<int>((cx - 0.5f * w) * inv), static_cast<int>((cy - 0.5f * h) * inv),
                           static_cast<int>(w * inv), static_cast<int>(h * inv));
        confidences.push_back(conf);
        std::vector<cv::Point2f> kp;
        std::vector<float> ks;
        for (int k = 0; k < nkpt; ++k) {
            const float* p = row + 5 + k * 3;
            kp.emplace_back(p[0] * inv, p[1] * inv);
            ks.push_back(p[2]);
        }
        kpts.push_back(std::move(kp));
        kpt_scores.push_back(std::move(ks));
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, config_.conf, config_.iou, keep);
    for (int idx : keep) {
        Result r;
        r.class_id = 0;
        r.confidence = confidences[idx];
        r.box = boxes[idx];
        r.keypoints = kpts[idx];
        r.keypoint_scores = kpt_scores[idx];
        results.push_back(r);
    }
    return results;
}

std::vector<Result> Predictor::postprocess_obb(const float* data, const std::vector<int64_t>& shape, float scale) {
    std::vector<Result> results;
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(shape[1]);
    const int dim2 = static_cast<int>(shape[2]);
    const bool end2end = dim1 > dim2;

    if (end2end) {
        const int num = dim1, step = dim2;  // [1, 300, 7] = cx,cy,w,h,conf,cls,angle
        for (int i = 0; i < num; ++i) {
            const float* row = data + i * step;
            const float conf = row[4];
            if (conf < config_.conf) continue;
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

    // Grid: [1, 4 + nc + 1, 8400] = cx,cy,w,h, class scores..., angle
    const int channels = dim1, anchors = dim2;
    const int nc = channels - 5;
    cv::Mat raw(channels, anchors, CV_32F, const_cast<float*>(data));
    cv::Mat t = raw.t();
    std::vector<cv::Rect> boxes;  // axis-aligned proxy for NMS
    std::vector<float> confidences;
    std::vector<Result> candidates;
    for (int i = 0; i < anchors; ++i) {
        const float* row = t.ptr<float>(i);
        cv::Mat scores(1, nc, CV_32F, const_cast<float*>(row + 4));
        cv::Point cls;
        double max_score;
        cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &cls);
        if (max_score < config_.conf) continue;
        const float cx = row[0] * inv, cy = row[1] * inv, w = row[2] * inv, h = row[3] * inv;
        const float angle = row[4 + nc];
        boxes.emplace_back(static_cast<int>(cx - 0.5f * w), static_cast<int>(cy - 0.5f * h),
                           static_cast<int>(w), static_cast<int>(h));
        confidences.push_back(static_cast<float>(max_score));
        Result r;
        r.class_id = cls.x;
        r.confidence = static_cast<float>(max_score);
        r.angle = angle;
        r.box = boxes.back();
        candidates.push_back(r);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, confidences, config_.conf, config_.iou, keep);
    for (int idx : keep) results.push_back(candidates[idx]);
    return results;
}

std::vector<Result> Predictor::postprocess_segment(const float* det, const std::vector<int64_t>& det_shape,
                                                   const float* protos, const std::vector<int64_t>& proto_shape,
                                                   float scale, const cv::Size& orig) {
    const float inv = 1.0f / scale;
    const int dim1 = static_cast<int>(det_shape[1]);
    const int dim2 = static_cast<int>(det_shape[2]);
    const bool end2end = dim1 > dim2;
    const int pc = static_cast<int>(proto_shape[1]);   // 32 mask coefficients
    const int mh = static_cast<int>(proto_shape[2]);   // 160
    const int mw = static_cast<int>(proto_shape[3]);   // 160
    cv::Mat proto_mat(pc, mh * mw, CV_32F, const_cast<float*>(protos));

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<cv::Mat> coeffs;

    if (end2end) {
        const int num = dim1, step = dim2;  // [1, 300, 6 + 32]
        for (int i = 0; i < num; ++i) {
            const float* row = det + i * step;
            const float conf = row[4];
            if (conf < config_.conf) continue;
            boxes.emplace_back(static_cast<int>(row[0] * inv), static_cast<int>(row[1] * inv),
                               static_cast<int>((row[2] - row[0]) * inv), static_cast<int>((row[3] - row[1]) * inv));
            confidences.push_back(conf);
            class_ids.push_back(static_cast<int>(row[5]));
            coeffs.push_back(cv::Mat(1, pc, CV_32F, const_cast<float*>(row + 6)).clone());
        }
    } else {
        const int channels = dim1, anchors = dim2;  // [1, 4 + nc + 32, 8400]
        const int nc = channels - 4 - pc;
        cv::Mat raw(channels, anchors, CV_32F, const_cast<float*>(det));
        cv::Mat t = raw.t();
        for (int i = 0; i < anchors; ++i) {
            const float* row = t.ptr<float>(i);
            cv::Mat scores(1, nc, CV_32F, const_cast<float*>(row + 4));
            cv::Point cls;
            double max_score;
            cv::minMaxLoc(scores, nullptr, &max_score, nullptr, &cls);
            if (max_score < config_.conf) continue;
            const float cx = row[0], cy = row[1], w = row[2], h = row[3];
            boxes.emplace_back(static_cast<int>((cx - 0.5f * w) * inv), static_cast<int>((cy - 0.5f * h) * inv),
                               static_cast<int>(w * inv), static_cast<int>(h * inv));
            confidences.push_back(static_cast<float>(max_score));
            class_ids.push_back(cls.x);
            coeffs.push_back(cv::Mat(1, pc, CV_32F, const_cast<float*>(row + 4 + nc)).clone());
        }
    }

    std::vector<int> keep;
    if (end2end) {
        keep.resize(boxes.size());
        for (size_t i = 0; i < keep.size(); ++i) keep[i] = static_cast<int>(i);
    } else {
        cv::dnn::NMSBoxes(boxes, confidences, config_.conf, config_.iou, keep);
    }

    const int new_w = static_cast<int>(std::round(orig.width * scale));
    const int new_h = static_cast<int>(std::round(orig.height * scale));

    std::vector<Result> results;
    for (int idx : keep) {
        // mask = sigmoid(coeffs @ protos), reshaped to the proto grid.
        cv::Mat mask_flat = coeffs[idx] * proto_mat;  // 1 x (mh*mw)
        cv::Mat mask = mask_flat.reshape(1, mh);      // mh x mw
        cv::Mat neg, sig;
        cv::exp(-mask, neg);
        sig = 1.0 / (1.0 + neg);

        // Proto grid corresponds to the letterboxed input; crop the valid region, scale to original.
        cv::Mat full;
        cv::resize(sig, full, cv::Size(imgsz_, imgsz_), 0, 0, cv::INTER_LINEAR);
        cv::Mat valid = full(cv::Rect(0, 0, std::min(new_w, imgsz_), std::min(new_h, imgsz_)));
        cv::Mat mask_orig;
        cv::resize(valid, mask_orig, orig, 0, 0, cv::INTER_LINEAR);

        cv::Mat binary = mask_orig > 0.5f;  // 8-bit 0/255
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

void Predictor::postprocess_semantic(const float* data, const std::vector<int64_t>& shape, float scale,
                                     const cv::Size& orig, cv::Mat& out) {
    const int nc = static_cast<int>(shape[1]);
    const int h = static_cast<int>(shape[2]);
    const int w = static_cast<int>(shape[3]);
    const long plane = static_cast<long>(h) * w;

    // Per-pixel argmax over the class channels.
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
}

}  // namespace yolo
