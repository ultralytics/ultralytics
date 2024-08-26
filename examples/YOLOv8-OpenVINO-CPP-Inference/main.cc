#include "inference.h"

#include <iostream>
#include <opencv2/highgui.hpp>

int main(int argc, char **argv) {
	// Check if the correct number of arguments is provided
	if (argc != 3) {
		std::cerr << "usage: " << argv[0] << " <model_path> <image_path>" << std::endl;
		return 1;
	}
	
	// Get the model and image paths from the command-line arguments
	const std::string model_path = argv[1];
	const std::string image_path = argv[2];
	
	// Read the input image
	cv::Mat image = cv::imread(image_path);
	
	// Check if the image was successfully loaded
	if (image.empty()) {
		std::cerr << "ERROR: image is empty" << std::endl;
		return 1;
	}
	
	// Define the confidence and NMS thresholds
	const float confidence_threshold = 0.5;
	const float NMS_threshold = 0.5;
	
	// Initialize the YOLO inference with the specified model and parameters
	yolo::Inference inference(model_path, cv::Size(640, 640), confidence_threshold, NMS_threshold);

	// Run inference on the input image
	inference.RunInference(image);
	
	// Display the image with the detections
	cv::imshow("image", image);
	cv::waitKey(0);

	return 0;
}
