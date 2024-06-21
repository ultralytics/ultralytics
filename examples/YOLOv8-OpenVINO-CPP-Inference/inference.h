#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace yolo {

// Structure to store the detection result
struct Detection {
	short class_id;        // Class ID of the detected object
	float confidence;      // Confidence score of the detection
	cv::Rect box;          // Bounding box around the detected object
};

class Inference {
 public:
	// Default constructor
	Inference() {}

// Constructor to initialize the model with default input shape
	Inference(const std::string &model_path, const float &model_confidence_threshold, const float &model_NMS_threshold);

// Constructor to initialize the model with specified input shape
	Inference(const std::string &model_path, const cv::Size model_input_shape, const float &model_confidence_threshold, const float &model_NMS_threshold);

	// Method to run inference on an input frame
	std::vector<Detection> RunInference(const cv::Mat &frame);

 private:
	// Method to initialize the model
	void InitialModel(const std::string &model_path);

	// Method to preprocess the input frame
	void Preprocessing(const cv::Mat &frame);

	// Method to postprocess the inference results
	void PostProcessing();

	// Method to get the bounding box in the correct scale
	cv::Rect GetBoundingBox(const cv::Rect &src) const;

	cv::Point2f factor_;             // Scaling factor for the input frame
	cv::Size2f model_input_shape_;   // Input shape of the model
	cv::Size model_output_shape_;    // Output shape of the model

	ov::InferRequest inference_request_;  // OpenVINO inference request
	ov::CompiledModel compiled_model_;    // OpenVINO compiled model

	std::vector<Detection> detections_;  // Vector to store the detection results

	float model_confidence_threshold_;  // Confidence threshold for detections
	float model_NMS_threshold_;         // Non-Maximum Suppression threshold
};

} // namespace yolo

#endif // YOLO_INFERENCE_H_
