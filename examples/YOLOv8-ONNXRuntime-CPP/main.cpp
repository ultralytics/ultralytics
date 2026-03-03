// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <iostream>
#include <iomanip>
#include "inference.h"
#include <filesystem>
#include <fstream>
#include <random>

void Detector(YOLO_V8*& p) {
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path / "images/detect/";
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".jpeg")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );


            }
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }
}


void Classifier(YOLO_V8*& p)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::filesystem::path imgs_path = current_path;// / "images"
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);
    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png")
        {
            std::string img_path = i.path().string();
            //std::cout << img_path << std::endl;
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            char* ret = p->RunSession(img, res);

            float positionY = 50;
            for (int i = 0; i < res.size(); i++)
            {
                int r = dis(gen);
                int g = dis(gen);
                int b = dis(gen);
                cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                positionY += 50;
            }

            cv::imshow("TEST_CLS", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
            //cv::imwrite("E:\\output\\" + std::to_string(k) + ".png", img);
        }

    }
}

void PoseEstimator(YOLO_V8*& p)
{
    std::filesystem::path current_path = std::filesystem::current_path();
    std::cout << "current_path: " << current_path << std::endl;
    std::filesystem::path imgs_path = current_path / "images/pose/";

    for (auto& i : std::filesystem::directory_iterator(imgs_path))
    {
        if (i.path().extension() == ".jpg" || i.path().extension() == ".png" || i.path().extension() == ".bmp")
        {
            std::string img_path = i.path().string();
            cv::Mat img = cv::imread(img_path);
            std::vector<DL_RESULT> res;
            p->RunSession(img, res);
            if (res.empty())
            {
                std::cout << "No pose detected in image: " << img_path << std::endl;
                //continue;
            }
            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color_box(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
                cv::Scalar color_point(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color_box, 2);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label_box = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                for (int i = 0; i < re.keyPoints.size(); i++)
                {
                    cv::circle(img, re.keyPoints[i], 5, color_point, -1);
                    std::string label_point = p->classes[i + 1];
                    cv::putText(img, label_point, re.keyPoints[i], cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 2);
                }
                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label_box.length() * 15, re.box.y),
                    color_box,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label_box,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );
            }
            //std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();


        }
    }
}

int ReadCocoYaml(YOLO_V8*& p, const std::string& yamlPath = "coco.yaml") {
    // Open the YAML file
    std::ifstream file(yamlPath);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    std::string kpt_num = "none";
    int kpts = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("kpt_shape:") != std::string::npos)
        {
            std::stringstream kpt_shape(lines[i]);
            std::getline(kpt_shape, kpt_num, '[');
            std::getline(kpt_shape, kpt_num, ',');
            if (!kpt_num.empty()) {
                try {
                    kpts = std::stoi(kpt_num);
                    std::cout << "kpt_num as integer: " << kpts << std::endl;
                }
                catch (const std::exception& e) {
                    std::cerr << "Error converting kpt_num to integer: " << e.what() << std::endl;
                }
            }
        }
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }


    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }
    if (kpts != 0) {
        for (int i = 1; i <= kpts; i++) {
            names.push_back(std::to_string(i));
		}
    }

    p->classes = names;
	p->kpts_num = kpts;
    return 0;
}


void DetectTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector, "./yaml/coco.yaml");
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "./models/yolov8n.onnx";
    params.imgSize = { 640, 640 };
#ifdef USE_CUDA
    params.cudaEnable = true;

    // GPU FP32 inference
    params.modelType = YOLO_DETECT_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // CPU inference
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif
    yoloDetector->CreateSession(params);
    Detector(yoloDetector);
}


void ClsTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    std::string model_path = "cls.onnx";
    ReadCocoYaml(yoloDetector, "./yaml/cls.yaml");
    DL_INIT_PARAM params{ model_path, YOLO_CLS, {224, 224} };
    yoloDetector->CreateSession(params);
    Classifier(yoloDetector);
}

void PoseTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector, "./yaml/coco8-pose.yaml");
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.25;
    params.pointScoresThreshold = 0.5;
    params.iouThreshold = 0.7;
    params.modelPath = "./models/yolov8n-pose.onnx";
    params.imgSize = { 640, 640 };
#ifdef USE_CUDA
    params.cudaEnable = true;

    // GPU FP32 inference
    params.modelType = YOLO_POSE_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // CPU inference
    params.modelType = YOLO_POSE_V8;
    params.cudaEnable = false;

#endif
    yoloDetector->CreateSession(params);
    PoseEstimator(yoloDetector);

    delete yoloDetector;
}

int main()
{
    //DetectTest();
    //ClsTest();
	PoseTest();
}
