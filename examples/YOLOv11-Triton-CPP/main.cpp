#include "inference.hpp"

#include <signal.h>
#include <iostream>
#include <ostream>
#include <vector>
#include <cstring>


#include <cstdint>
#include <chrono>


#define GORUNTU_GENISLIK 640
#define GORUNTU_YUKSEKLIK 640
#define GORUNTU_KANAL 3

#define NETWORK_THRESHOLD 0.50

double get_time_since_epoch_millis()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    auto duration = now.time_since_epoch();
    return duration_cast<microseconds>(duration).count() / 1000.0;
}

int main(int argc, char *argv[])
{
    std::string url= "localhost:8001";
    std::string model_name= "yolov11";
    std::string model_version= "1";
	std::string image_path = "/home/robolaunch/workspaces/opensource/ipms_contribute/test.jpg";
	std::string output_path = "/home/robolaunch/workspaces/opensource/ipms_contribute";
	std::vector<std::string> object_class_list = {"person", "forklift"};

    double ag_tespit_suresi_milisaniye;
    std::vector<uint16_t> triton_request_data;
    triton_request_data.resize(3*GORUNTU_GENISLIK*GORUNTU_YUKSEKLIK);

    std::vector<struct detection_struct> detections;

    std::shared_ptr<TritonCommunication> triton_communication = std::make_shared<TritonCommunication>(url, model_name, model_version, GORUNTU_KANAL, GORUNTU_GENISLIK, GORUNTU_YUKSEKLIK,object_class_list.size());

    cv::Mat frame = cv::imread(image_path);
	if (frame.empty()) {
		std::cerr << "Image couldn't read: " << image_path << std::endl;
		return -1;
	}
    int image_width = frame.cols;
    int image_height = frame.rows;

    double preprocess_time = get_time_since_epoch_millis();
	Image::preprocess(&frame, triton_request_data, GORUNTU_GENISLIK, GORUNTU_YUKSEKLIK);
    std::cout << "Preprocess time : " << (get_time_since_epoch_millis() - preprocess_time)<< " millisecond."<< std::endl;

	double infer_time = get_time_since_epoch_millis(); 
    triton_communication->infer(triton_request_data.data());
	std::cout << "Triton Server execute time : " << (get_time_since_epoch_millis() - infer_time)<< " millisecond."<<std::endl;

	getDetectionsFromTritonRawData(triton_communication->output_raw_data,detections,object_class_list,NETWORK_THRESHOLD,image_width,image_height);
	    
	for (int i = 0 ; i < detections.size(); i++)
	{
		std::ostringstream oss;
    	oss << detections[i].name << " "              
        << std::fixed << std::setprecision(2) 
        << detections[i].confidence_score;            

		cv::rectangle(frame,detections[i].bbox,cv::Scalar(255,0,0),2);
		cv::putText(frame,oss.str(),cv::Point((detections[i].bbox.x),(detections[i].bbox.y-5)),cv::FONT_HERSHEY_DUPLEX, ((frame.cols/640.0f) * 0.35),cv::Scalar(0, 0, 0), (int)(frame.cols/640.0f)+1);
		cv::putText(frame,oss.str(),cv::Point((detections[i].bbox.x),(detections[i].bbox.y-5)),cv::FONT_HERSHEY_DUPLEX, ((frame.cols/640.0f) * 0.35),cv::Scalar(0xFF, 0xFF, 0xFF), (int)(frame.cols/640.0f));

		cv::imwrite((output_path+"/result_image.jpg"),frame);
		std::cout<< "Result image saved!"<<std::endl;
	}

    return 0;
}


