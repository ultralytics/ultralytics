// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "inference.hpp"
#include "yolo_cli.hpp"
#include "yolo_render.hpp"
#include "yolo_show.hpp"

int main(int argc, char** argv) {
    yolo::Config config;
    config.url = yolo::ArgValue(argc, argv, "--url", "localhost:8001");
    config.model = yolo::ArgValue(argc, argv, "--model", "yolo26n");
    config.model_version = yolo::ArgValue(argc, argv, "--version", "");
    config.conf = std::stof(yolo::ArgValue(argc, argv, "--conf", "0.25"));
    config.iou = std::stof(yolo::ArgValue(argc, argv, "--iou", "0.45"));
    config.imgsz = std::stoi(yolo::ArgValue(argc, argv, "--imgsz", "640"));
    config.task = yolo::TaskFromString(yolo::ArgValue(argc, argv, "--task", "unknown"));
    const std::string source = yolo::ArgValue(argc, argv, "--source", "bus.jpg");
    const std::string output = yolo::ArgValue(argc, argv, "--out", "result.jpg");
    const bool show = yolo::ShowRequested(argc, argv);

    cv::Mat image = cv::imread(source);
    if (image.empty()) {
        std::cerr << "ERROR: could not read image '" << source << "'" << std::endl;
        return 1;
    }

    try {
        yolo::Predictor predictor(config);
        const std::vector<std::string>& names = predictor.names();

        cv::Mat semantic;
        std::vector<yolo::Result> results = predictor.predict(image, semantic);
        std::cout << "Model: " << config.model << " @ " << config.url
                  << " | task: " << yolo::TaskName(predictor.task()) << " | classes: " << names.size() << std::endl;

        cv::Mat canvas = image.clone();
        yolo::RenderAndPrint(canvas, predictor.task(), results, names, semantic);

        cv::imwrite(output, canvas);
        std::cout << "Result image written to " << output << std::endl;
        yolo::Show("YOLO", canvas, show);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
