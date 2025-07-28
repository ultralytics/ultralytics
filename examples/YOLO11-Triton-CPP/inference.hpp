// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <opencv2/opencv.hpp>
#include "grpc_client.h"

class Image 
{
public:
    Image() = default;
    static void preprocess(cv::Mat* image, std::vector<uint16_t>& triton_data, int input_w, int input_h);
};

struct struct_yolo_output
{
    std::vector<int> num_dets, det_classes;
    std::vector<float> det_boxes, det_scores;
};

struct BoundingBox {
    float x, y, w, h;
    float score;
    int class_id;
};

struct detection_struct
{
    cv::Rect bbox;
    int class_id;
    std::string name;
    double confidence_score;
};

// C-compatible declarations
#ifdef __cplusplus
extern "C" {
#endif

int getDetectionsFromTritonRawData(
    std::vector<float>& detection_results,
    std::vector<struct detection_struct>& tespitler,
    std::vector<std::string>& object_class_list,
    float confidence_threshold,
    int image_width,
    int image_height
);

std::vector<BoundingBox> NMS(const std::vector<BoundingBox>& boxes, float iou_threshold);
float IoU(const BoundingBox& box1, const BoundingBox& box2);

#ifdef __cplusplus
}
#endif

namespace tc = triton::client;

class TritonCommunication
{
private:
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    std::string triton_url;
    std::vector<int64_t> shape;
    tc::InferOptions options;
    size_t input_byte_size;
    size_t output_byte_size;
    std::shared_ptr<tc::InferResult> results_ptr;

public:
    std::vector<float> output_raw_data;

    TritonCommunication(std::string triton_address,
                        std::string model_name,
                        std::string model_version,
                        int image_channel,
                        int image_width,
                        int image_height,
                        int class_count);

    void infer(uint16_t* triton_data);
};
