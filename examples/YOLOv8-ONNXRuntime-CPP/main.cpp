#include <iostream>
#include <stdio.h>
#include "inference.h"
#include <filesystem>



void file_iterator(DCSP_CORE*& p)
{
	std::filesystem::path img_path = R"(E:\project\Project_C++\DCPS_ONNX\TEST_ORIGIN)";
	int k = 0;
	for (auto& i : std::filesystem::directory_iterator(img_path))
	{
		if (i.path().extension() == ".jpg")
		{
			std::string img_path = i.path().string();
			//std::cout << img_path << std::endl;
			cv::Mat img = cv::imread(img_path);
			std::vector<DCSP_RESULT> res;
			char* ret = p->RunSession(img, res);
			for (int i = 0; i < res.size(); i++)
			{
				cv::rectangle(img, res.at(i).box, cv::Scalar(125, 123, 0), 3);
			}

			k++;
			cv::imshow("TEST_ORIGIN", img);
			cv::waitKey(0);
			cv::destroyAllWindows();
			//cv::imwrite("E:\\output\\" + std::to_string(k) + ".png", img);
		}
	}
}



int main()
{
	DCSP_CORE* p1 = new DCSP_CORE;
	std::string model_path = "yolov8n.onnx";
	DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {640, 640}, 80, 0.1, 0.5, false };
	char* ret = p1->CreateSession(params);
	file_iterator(p1);
}
