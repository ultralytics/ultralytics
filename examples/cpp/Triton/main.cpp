// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.hpp"
#include "yolo_draw.hpp"
#include "yolo_show.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <chrono>

#define MODEL_INPUT_IMAGE_WIDTH 640
#define MODEL_INPUT_IMAGE_HEIGHT 640
#define NETWORK_THRESHOLD 0.50
#define IMAGE_CHANNEL 3

double get_time_since_epoch_millis()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto duration = now.time_since_epoch();
    return duration_cast<microseconds>(duration).count() / 1000.0;
}

int main(int argc, char *argv[])
{
    std::string triton_address= "localhost:8001";
    std::string model_name= "yolo11";
    std::string model_version= "1";
	std::string image_path = "test.jpg";
	std::string output_path = "output.jpg";
	std::vector<std::string> object_class_list = {"class1", "class2"};

    std::vector<uint16_t> triton_request_data;
    triton_request_data.resize(IMAGE_CHANNEL*MODEL_INPUT_IMAGE_WIDTH*MODEL_INPUT_IMAGE_HEIGHT);
    std::vector<struct detection_struct> detections;

    std::shared_ptr<TritonCommunication> triton_communication = std::make_shared<TritonCommunication>(triton_address, model_name, model_version, IMAGE_CHANNEL, MODEL_INPUT_IMAGE_WIDTH, MODEL_INPUT_IMAGE_HEIGHT,object_class_list.size());

    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Image couldn't read: " << image_path << std::endl;
        return -1;
    }

    int image_width = frame.cols;
    int image_height = frame.rows;

    double preprocess_time = get_time_since_epoch_millis();
	Image::preprocess(&frame, triton_request_data, MODEL_INPUT_IMAGE_WIDTH, MODEL_INPUT_IMAGE_HEIGHT);
    std::cout << "Preprocess time : " << (get_time_since_epoch_millis() - preprocess_time)<< " millisecond."<< std::endl;

    double infer_time = get_time_since_epoch_millis();
    triton_communication->infer(triton_request_data.data());
    std::cout << "Triton Server execute time : " << (get_time_since_epoch_millis() - infer_time) << " millisecond." << std::endl;

    getDetectionsFromTritonRawData(triton_communication->output_raw_data, detections, object_class_list, NETWORK_THRESHOLD, image_width, image_height);

    const bool show = yolo::ShowRequested(argc, argv);  // pass --show to display the result
    for (int i = 0; i < detections.size(); i++)
    {
        yolo::DrawBox(frame, detections[i].bbox,
                      yolo::Label(detections[i].name, static_cast<float>(detections[i].confidence_score)),
                      detections[i].class_id);
    }

    cv::imwrite(output_path, frame);
    std::cout << "Result image saved!"<< std::endl;
    yolo::Show("Result", frame, show);

    return 0;
}
