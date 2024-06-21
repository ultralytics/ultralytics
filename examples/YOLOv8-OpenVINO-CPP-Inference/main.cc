#include "inference.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <random>

// List of class names corresponding to the YOLO model
std::vector<std::string> classes{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
	"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
	"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
	"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
	"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
	"scissors", "teddy bear", "hair drier", "toothbrush"
};

// Function to draw detected objects on the frame
void DrawDetectedObject(cv::Mat &frame, const std::vector<yolo::Detection> &detections);

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
	
	// Start measuring the inference time
	const auto start = std::chrono::high_resolution_clock::now();
	
	// Run inference on the input image
	std::vector<yolo::Detection> detections = inference.RunInference(image);
	
	// End measuring the inference time
	const auto end = std::chrono::high_resolution_clock::now();
	std::cout << "Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
	
	// Draw the detected objects on the image
	DrawDetectedObject(image, detections);
	
	// Display the image with the detections
	cv::imshow("image", image);
	
	// Define the escape key to exit the display window
	const char escape_key = 27;
	
	// Wait for the escape key to be pressed to close the window
	while (cv::waitKey(0) != escape_key);
	
	// Destroy all OpenCV windows
	cv::destroyAllWindows();
	
	return 0;
}

// Function to draw detected objects on the frame
void DrawDetectedObject(cv::Mat &frame, const std::vector<yolo::Detection> &detections) {
	for (int i = 0; i < detections.size(); ++i) {
		yolo::Detection detection = detections[i];
		cv::Rect box = detection.box;
		float confidence = detection.confidence;
		int class_id = detection.class_id;
		
		// Generate a random color for the bounding box
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(120, 255);
		cv::Scalar color = cv::Scalar(dis(gen), dis(gen), dis(gen));
		
		// Draw the bounding box around the detected object
		cv::rectangle(frame, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), color, 3);
		
		// Prepare the class label and confidence text
		std::string classString = classes[class_id] + std::to_string(confidence).substr(0, 4);
		
		// Get the size of the text box
		cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 0.75, 2, 0);
		cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);
		
		// Draw the text box
		cv::rectangle(frame, textBox, color, cv::FILLED);
		
		// Put the class label and confidence text above the bounding box
		cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 0.75, cv::Scalar(0, 0, 0), 2, 0);
	}
}
