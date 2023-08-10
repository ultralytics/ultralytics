#include <iostream>
#include "inference.h"
#include <filesystem>
#include <fstream>

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

int read_coco_yaml(DCSP_CORE*& p)
{
    // Open the YAML file
    std::ifstream file("coco.yaml");
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++) {
        if (lines[i].find("names:") != std::string::npos) {
            start = i + 1;
        } else if (start > 0 && lines[i].find(':') == std::string::npos) {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++) {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}


int main()
{
	DCSP_CORE* p1 = new DCSP_CORE;
	std::string model_path = "yolov8n.onnx";
    read_coco_yaml(p1);
	// GPU inference
	DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {640, 640},  0.1, 0.5, true };
	// CPU inference
    // DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {640, 640},  0.1, 0.5, false };
    p1->CreateSession(params);
	file_iterator(p1);
}
