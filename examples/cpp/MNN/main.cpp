// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "inference.h"
#include "yolo_draw.hpp"
#include "yolo_show.hpp"

namespace {

std::string ArgValue(int argc, char** argv, const std::string& key, const std::string& fallback) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) return argv[i + 1];
    }
    return fallback;
}

std::string NameOf(const std::vector<std::string>& names, int id) {
    return id >= 0 && id < static_cast<int>(names.size()) ? names[id] : std::to_string(id);
}

}  // namespace

int main(int argc, char** argv) {
    yolo::Config config;
    config.model_path = ArgValue(argc, argv, "--model", "yolo26n.mnn");
    config.conf = std::stof(ArgValue(argc, argv, "--conf", "0.25"));
    config.iou = std::stof(ArgValue(argc, argv, "--iou", "0.45"));
    config.threads = std::stoi(ArgValue(argc, argv, "--threads", "4"));
    const std::string source = ArgValue(argc, argv, "--source", "bus.jpg");
    const std::string output = ArgValue(argc, argv, "--out", "result.jpg");
    const bool show = yolo::ShowRequested(argc, argv);

    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        std::cerr << "ERROR: could not read image '" << source << "'" << std::endl;
        return 1;
    }

    yolo::Predictor predictor(config);
    const std::vector<std::string>& names = predictor.names();
    std::cout << "Model: " << config.model_path << " | task: " << yolo::TaskName(predictor.task())
              << " | classes: " << names.size() << std::endl;

    cv::Mat semantic;
    std::vector<yolo::Result> results = predictor.predict(image, semantic);

    cv::Mat canvas = image.clone();
    switch (predictor.task()) {
        case yolo::Task::Semantic:
            yolo::DrawSemantic(canvas, semantic);
            std::cout << "semantic map rendered (" << names.size() << " classes)" << std::endl;
            break;
        case yolo::Task::Segment:
        case yolo::Task::Detect:
        case yolo::Task::Pose: {
            for (const yolo::Result& r : results) {
                const std::string name = NameOf(names, r.class_id);
                if (!r.mask.empty()) yolo::DrawMask(canvas, r.mask, r.class_id);
                if (!r.keypoints.empty()) yolo::DrawPose(canvas, r.keypoints, r.keypoint_scores);
                yolo::DrawBox(canvas, r.box, yolo::Label(name, r.confidence), r.class_id);
                std::cout << name << " " << std::fixed << std::setprecision(2) << r.confidence
                          << " box=[" << r.box.x << ", " << r.box.y << ", " << r.box.width << ", "
                          << r.box.height << "]" << std::endl;
            }
            break;
        }
        case yolo::Task::Obb: {
            for (const yolo::Result& r : results) {
                const std::string name = NameOf(names, r.class_id);
                cv::RotatedRect rr(cv::Point2f(r.box.x + r.box.width * 0.5f, r.box.y + r.box.height * 0.5f),
                                   cv::Size2f(static_cast<float>(r.box.width), static_cast<float>(r.box.height)),
                                   r.angle * 180.0f / static_cast<float>(CV_PI));
                yolo::DrawObb(canvas, rr, yolo::Label(name, r.confidence), r.class_id);
                std::cout << name << " " << std::fixed << std::setprecision(2) << r.confidence
                          << " angle=" << r.angle << std::endl;
            }
            break;
        }
        case yolo::Task::Classify: {
            int y = 28;
            for (const yolo::Result& r : results) {
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
            std::cerr << "[yolo] task '" << yolo::TaskName(predictor.task()) << "' is not supported." << std::endl;
    }

    cv::imwrite(output, canvas);
    std::cout << "Result image written to " << output << std::endl;
    yolo::Show("YOLO", canvas, show);
    return 0;
}
