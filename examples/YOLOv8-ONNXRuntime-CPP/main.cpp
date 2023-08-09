#include <iostream>
#include "inference.h"
#include <filesystem>

void file_iterator(DCSP_CORE*& p)
{
	std::filesystem::path current_path = std::filesystem::current_path();
	std::filesystem::path imgs_path = current_path/"images";
	for (auto& i : std::filesystem::directory_iterator(imgs_path))
	{
		if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
		{
			std::string img_path = i.path().string();
			cv::Mat img = cv::imread(img_path);
			std::vector<DCSP_RESULT> res;
			p->RunSession(img, res);

			for (auto & re : res)
			{
				cv::rectangle(img, re.box, cv::Scalar(0, 0 , 255), 3);
                std::string label = p->classes[re.classId];
                cv::putText(
                        img,
                        label,
                        cv::Point(re.box.x, re.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.75,
                        cv::Scalar(255, 255, 0),
                        2
                );
			}
            cv::imshow("Result", img);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}
}


int main()
{
	DCSP_CORE* p1 = new DCSP_CORE;
	std::string model_path = "yolov8n.onnx";
	// GPU inference
	DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {640, 640},  0.1, 0.5, true };
	// CPU inference
    // DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {640, 640},  0.1, 0.5, false };
    p1->CreateSession(params);
	file_iterator(p1);
}
