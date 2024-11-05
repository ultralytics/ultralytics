#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
#include <random>
struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0};
    cv::Scalar color{};
    cv::Rect box{};
};

std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                                 "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
                                 "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                                 "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
                                 "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                                 "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                 "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

std::unordered_map<std::string, cv::Mat> preprocess(const cv::Mat &img, const cv::Size &img_size)
{
    cv::Mat imgc = img.clone();
    if (imgc.size() != img_size)
        cv::resize(imgc, imgc, img_size);
    cv::Mat blob = cv::dnn::blobFromImage(imgc, 1 / 255.f, cv::Size(), cv::Scalar(), true, false);
    std::unordered_map<std::string, cv::Mat> input_blob;
    input_blob["images"] = blob;
    return input_blob;
}
cv::Mat postprocess(const cv::Mat &output_blob, const cv::Mat &img, const cv::Size2f &scale)
{
    cv::Mat imgc = img.clone();

    // reshape
    cv::Mat output_blobc = output_blob.clone().reshape(1, 84);
    output_blobc.convertTo(output_blobc, CV_32F);
    cv::transpose(output_blobc, output_blobc);

    // data
    std::vector<cv::Rect> boxes;
    std::vector<float> scores_classs;
    std::vector<int> indices;

    // NMS
    float confidenceThreshold = 0.5;
    float nmsThreshold = 0.5;

    // convert data
    for (int i = 0; i < output_blobc.rows; i++)
    {
        // class
        float *classes_scores = (float *)output_blobc.row(i).data + 4;
        cv::Mat scores(cv::Size(80, 1), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;
        // maximum and the location
        minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
        if (maxClassScore > confidenceThreshold)
        {
            scores_classs.push_back(maxClassScore);
            indices.push_back(class_id.x);
            float x = output_blobc.at<float>(i, 0);
            float y = output_blobc.at<float>(i, 1);
            float w = output_blobc.at<float>(i, 2);
            float h = output_blobc.at<float>(i, 3);
            int left = int((x - 0.5 * w) * scale.width);
            int top = int((y - 0.5 * h) * scale.height);

            int width = int(w * scale.width);
            int height = int(h * scale.height);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, scores_classs, confidenceThreshold, nmsThreshold, nms_result);
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = indices[idx];
        result.confidence = scores_classs[idx];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(100, 255);
        result.color = cv::Scalar(dis(gen),
                                  dis(gen),
                                  dis(gen));
        result.className = classes[indices[idx]];
        result.box = boxes[idx];
        // box
        cv::rectangle(imgc, boxes[idx], result.color);
        // text
        std::string classString = result.className + ' ' + std::to_string(result.confidence).substr(0, 4);
        cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
        cv::Rect textBox(result.box.x, result.box.y - 40, textSize.width + 10, textSize.height + 20);
        cv::rectangle(imgc, textBox, result.color, cv::FILLED);
        // class
        cv::putText(imgc, classString, cv::Point(result.box.x + 5, result.box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
    }
    return imgc;
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage : exec engine_path img_path inputsize_row inputsize_col" << std::endl;
        return -1;
    }
    // model
    TRTInfer model(argv[1]);
    float row = std::atof(argv[3]);
    float col = std::atof(argv[4]);
    cv::Size imgsize(col, row);
    // image
    cv::Mat image = cv::imread(argv[2]);
    cv::imshow("picure",image);
    cv::waitKey();
    float scalew = static_cast<float>(image.size().width) / col;
    float scaleh = static_cast<float>(image.size().height) / row;
    cv::Size2f scale_factor(scalew, scaleh);

    // preprocess
    auto input_blob = preprocess(image, imgsize);

    // inference
    auto output_blob = model(input_blob);

    // post process
    cv::Mat result = postprocess(output_blob["output0"], image, scale_factor);
    cv::imwrite("./result.jpg", result);
    // show result
    cv::imshow("output", result);
    cv::waitKey();

    return 1;
}