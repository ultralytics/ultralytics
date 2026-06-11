// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include "yolo_draw.hpp"
#include "coco_names.hpp"

namespace yolo {

struct Detection {
	short class_id;
	float confidence;
	cv::Rect box;
};

class Inference {
 public:
	Inference() {}
	// Constructor to initialize the model with default input shape
	Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold);
	// Constructor to initialize the model with specified input shape
	Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold);

	void RunInference(cv::Mat &frame);

 private:
	void InitializeModel(const std::string &model_path);
	void Preprocessing(const cv::Mat &frame);
	void PostProcessing(cv::Mat &frame);
	cv::Rect GetBoundingBox(const cv::Rect &src) const;
	void DrawDetectedObject(cv::Mat &frame, const Detection &detections) const;

	cv::Point2f scale_factor_;			// Scaling factor for the input frame
	cv::Size2f model_input_shape_;	// Input shape of the model
	cv::Size model_output_shape_;		// Output shape of the model

	ov::InferRequest inference_request_;  // OpenVINO inference request
	ov::CompiledModel compiled_model_;    // OpenVINO compiled model

	float model_confidence_threshold_;  // Confidence threshold for detections
	float model_NMS_threshold_;         // Non-Maximum Suppression threshold

	// COCO class names from common/coco_names.hpp. Replace with your own list if
	// you train on a custom dataset.
	std::vector<std::string> classes_ = CocoNames();
};

} // namespace yolo

#endif // YOLO_INFERENCE_H_
