/*
#############
###ZouJiu
###20240421
###1069679911@qq.com
#https://zoujiu.blog.csdn.net/
#https://zhihu.com/people/zoujiu1
#https://github.com/ZouJiu1
#############
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <thread>
#include "thread_safe_Queue.hpp"
#include <ctime>

using namespace std;
using namespace cv;
using namespace cv::dnn;

struct info {
    double fps;
    int width;
    int height;
};

struct Result {
    vector<Mat> outs;
    Mat img;
};

struct letterbox {
    float paddingValue = 116 - 2;
    bool swapRB = false;
    int inpWidth = 640;
    int inpHeight = 640;
    Scalar scale = 1 / 255.0;
    Scalar mean = 0.0;
    float conf_threshold = 0.25;
    float iou_threshold = 0.7;
};

vector<string> classes;
bool GPU = false;
int batch_size = 2;
float conf_threshold = 0.25;
float iou_threshold = 0.7;
string projectBasePath = "C:\\Users\\10696\\Desktop\\CV\\ultralytics\\examples\\YOLOv8-CPP-Inference"; // Set your ultralytics base path
string Onnx_path = projectBasePath + "\\yolov8n_2.onnx";
string classes_path = projectBasePath + "\\classes.txt";
string video_path = projectBasePath + "\\MOT16-06-raw.mp4";
string output_path = video_path.substr(0, video_path.size() - 4) + "_output.mp4";
// string output_path = video_path.substr(0, video_path.size() - 4) + "_output.avi";
string ModelType = "yolov8";
int output_length = 1;
int Image_in_queue_maxsize = 390;
int Result_in_queue_maxsize = 390;
ThreadQueue<cv::Mat> image_que = ThreadQueue<cv::Mat>(Image_in_queue_maxsize);
ThreadQueue<vector<Result>> result_que = ThreadQueue<vector<Result>>(Image_in_queue_maxsize);
ThreadQueue<info> informa_que = ThreadQueue<info>((int)1);

void yoloPostProcessing(
    std::vector<Mat>&& outs,
    std::vector<int>& keep_classIds,
    std::vector<float>& keep_confidences,
    std::vector<Rect2d>& keep_boxes,
    float conf_threshold = 0.25,
    float iou_threshold = 0.7,
    const std::string& modeltype = "yolov8")
{ // https://github.com/opencv/opencv/blob/4.x/modules/dnn/test/test_onnx_importer.cpp#L2666-L2750

    // Retrieve
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect2d> boxes;

    if (modeltype == "yolov8") {
        cv::transposeND(outs[0], { 0, 2, 1 }, outs[0]);
    }

    // each row is [cx, cy, w, h, conf_obj, conf_class1, ..., conf_class80]
    for (auto preds : outs) {

        preds = preds.reshape(1, preds.size[1]); // [1, 8400, 85] -> [8400, 85]

        for (int i = 0; i < preds.rows; ++i)
        {
            // filter out non objects
            float obj_conf = (modeltype != "yolov8") ? preds.at<float>(i, 4) : 1.0f;
            if (obj_conf < conf_threshold)
                continue;

            Mat scores = preds.row(i).colRange((modeltype != "yolov8") ? 5 : 4, preds.cols);
            double conf;
            Point maxLoc;
            minMaxLoc(scores, 0, &conf, 0, &maxLoc);

            conf = (modeltype != "yolov8") ? conf * obj_conf : conf;
            if (conf < conf_threshold)
                continue;

            // get bbox coords
            float* det = preds.ptr<float>(i);
            double cx = det[0];
            double cy = det[1];
            double w = det[2];
            double h = det[3];

            // [x1, y1, x2, y2]
            boxes.push_back(Rect2d(cx - 0.5 * w, cy - 0.5 * h,
                cx + 0.5 * w, cy + 0.5 * h));
            classIds.push_back(maxLoc.x);
            confidences.push_back(conf);
        }
    }

    // NMS
    std::vector<int> keep_idx;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, keep_idx);

    for (auto i : keep_idx)
    {
        keep_classIds.push_back(classIds[i]);
        keep_confidences.push_back(confidences[i]);
        keep_boxes.push_back(boxes[i]);
    }
}

void drawPrediction(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{ // https://github.com/opencv/opencv/blob/4.x/samples/dnn/yolo_detector.cpp#L83-L101
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

    string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    top = max(top, labelSize.height);
    rectangle(frame, Point(left, top - labelSize.height),
        Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void predict_image(Net&& net, const int batch_size) {
    cout << "in predict_image function" << endl;
    ImagePaddingMode paddingMode = static_cast<ImagePaddingMode>(DNN_PMODE_LETTERBOX);
    letterbox ltr;
    Size size(ltr.inpWidth, ltr.inpHeight);
    Image2BlobParams imgParams(
        ltr.scale,
        size,
        ltr.mean,
        ltr.swapRB,
        CV_32F,
        DNN_LAYOUT_NCHW,
        paddingMode,
        ltr.paddingValue);
    Mat inp, img;
    vector<Mat> outs, batch;
    vector<MatShape> inLayerShapes;
    vector<MatShape> outLayerShapes;
    net.getLayerShapes(MatShape(), 0, inLayerShapes, outLayerShapes);
    MatShape kk = inLayerShapes[0];
    if (inLayerShapes[0][0] != batch_size) {
        cerr << "";
    }
    int cnt = 0;
    vector<Range> ranges(3, Range::all());
    vector<Result> ret;
    int img_width = 0;
    while (true) {
        ret.clear();
        if (batch_size > 1) {
            batch.clear();
        }
        for (int i = 0; i < batch_size; i++) {
            image_que.get(img);
            img_width = img.size[1];
            cout << "predict..." << cnt++ << " " << img_width << endl;
            if (img_width < 9) {
                image_que.task_done();
                if (batch_size > 1) {
                    for (int j = 0; j < batch.size(); j++) {
                        image_que.task_done();
                    }
                }
                result_que.put({});
                return;
            }
            if (batch_size > 1) {
                batch.push_back(img);
            }
        }
        if (batch_size > 1) {
            inp = blobFromImagesWithParams(batch, imgParams);
        }
        else {
            inp = blobFromImageWithParams(img, imgParams);
        }
        net.setInput(inp);
        net.forward(outs, net.getUnconnectedOutLayersNames());
        if (batch_size > 1) {
            for (int ic = 0; ic < batch.size(); ic++) {
                ranges[0].start = ic;
                ranges[0].end = ic + 1;
                ret.push_back({ {outs[0](ranges)}, std::move(batch[ic]) });
                image_que.task_done();
            }
        }
        else {
            ret.push_back({ std::move(outs), std::move(img) });
            image_que.task_done();
        }
        result_que.put(std::move(ret));

    }
}

void read_video_image(string video_path) {
    Mat img;
    bool success;
    VideoCapture cap = VideoCapture(video_path);
    MatShape tensor(3, 1);
    img.create(tensor, 0);
    if (!cap.isOpened()) {
        cap.release();
        image_que.put(std::move(img));
        informa_que.put({ -1, -1, -1 });
        return;
    }
    double fps = cap.get(CAP_PROP_FPS);
    double frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    double frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    info tp = { fps, (int)(frame_width), (int)(frame_height) };
    informa_que.put(std::move(tp));
    success = cap.read(img);
    int cnt = 0;
    //string imr = "C:\\Users\\10696\\Desktop\\CV\\ultralytics\\examples\\YOLOv8-CPP-Inference\\tmp";
    while (success) {
        cout << "read..." << cnt++ << " " << (int)success << endl;
        //imwrite(imr + "\\" + to_string(cnt) + "_.jpg", img);
        image_que.put(std::move(img.clone()));
        success = cap.read(img);
        //if (cnt == 31) break;
    }
    cap.release();
    img.create(tensor, 0);
    image_que.put(std::move(img));
}

void write_to_video(string output_path) {
    cout << "in write_to_video function" << endl;
    int frame_width, frame_height;
    double fps;
    info inf;
    informa_que.get(inf);
    fps = inf.fps;
    frame_width = inf.width;
    frame_height = inf.height;
    if (fps < 0 && frame_width < 0 && frame_height < 0) {
        return;
    }
    cout << fps << " " << frame_width << " " << frame_height << endl;
    const int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
    // const int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');  // for avi
    // cv2.VideoWriter_fourcc('I', '4', '2', '0') // for avi
    // cv2.VideoWriter_fourcc('P', 'I', 'M', 'I') // for avi
    VideoWriter video_writer = VideoWriter(output_path, fourcc, fps, Size(frame_width, frame_height));
    if (!video_writer.isOpened()) {
        cout << "video output writer is wrong......" << endl;
        exit(-1);
    }
    cout << "in video_writer" << endl;
    informa_que.task_done();
    informa_que.join();
    Mat inp;
    vector<int> keep_classIds;
    vector<float> keep_confidences;
    vector<Rect2d> keep_boxes;
    vector<Rect> boxes;
    // rescale boxes back to original image
    Image2BlobParams paramNet;
    letterbox ltr;
    Size size(ltr.inpWidth, ltr.inpHeight);
    ImagePaddingMode paddingMode = static_cast<ImagePaddingMode>(DNN_PMODE_LETTERBOX);
    paramNet.scalefactor = ltr.scale;
    paramNet.size = size;
    paramNet.mean = ltr.mean;
    paramNet.swapRB = ltr.swapRB;
    paramNet.paddingmode = paddingMode;
    vector<Result> result;
    int cnt = 0;
    //string imr = "C:\\Users\\10696\\Desktop\\CV\\ultralytics\\examples\\YOLOv8-CPP-Inference\\tmp";
    while (true) {
        result.clear();
        result_que.get(result);
        cout << "result_que.unfinished_tasks: " << result.size() << endl;
        if (result.size() == 0) {
            result_que.task_done();
            video_writer.release();
            break;
        }
        for (int i = 0; i < result.size(); i++) {
            // Retrieve
            keep_boxes.clear();
            keep_confidences.clear();
            boxes.clear();
            keep_classIds.clear();
            yoloPostProcessing(std::move(result[i].outs), keep_classIds, keep_confidences, keep_boxes, conf_threshold, iou_threshold, ModelType);
            for (auto& box : keep_boxes)
            {
                boxes.push_back(Rect(cvFloor(box.x), cvFloor(box.y), cvFloor(box.width - box.x), cvFloor(box.height - box.y)));
            }

            paramNet.blobRectsToImageRects(boxes, boxes, result[i].img.size());

            for (size_t idx = 0; idx < boxes.size(); ++idx)
            {
                Rect box = boxes[idx];
                drawPrediction(keep_classIds[idx], keep_confidences[idx], box.x, box.y,
                    box.width + box.x, box.height + box.y, result[i].img);
            }
            cout << "write frame......." << cnt++ << endl;
            video_writer.write(result[i].img);
            //imwrite(imr +"\\" + to_string(cnt) + ".jpg", result[i].img);
        }
        result_que.task_done();
    }
    cout << "post break" << endl;
}
/*
ThreadQueue<int> trr = ThreadQueue<int>(100);
void putput() {
    for (int i = 0; i < 10000000; i++) {
        trr.put(i);
        int a = 0;
    }
}

void getget() {
    int a;
    for (int i = 0; i < 10; i++) {
        a = trr.get();
        int k = 0;
    }
}
*/

void YoLoForward()
{
    // https://github.com/opencv/opencv/blob/4.x/modules/dnn/src/dnn_read.cpp#L13-L80
    /*
    thread tt(getget);
    thread tt1(putput);
    tt.join();
    tt1.join();
    exit(0);
    */

    if (classes.empty())
    {
        ifstream ifs(classes_path.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + classes_path + " not found");
        string line;
        while (getline(ifs, line))
        {
            classes.push_back(line);
        }
        ifs.close();
    }

    Net net = readNet(Onnx_path, "", "onnx");

    if (GPU) {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA);
    }
    else {
        net.setPreferableBackend(DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    clock_t prevTimestamp = 0;

    thread t0(read_video_image, video_path);
    thread t1(predict_image, std::move(net), batch_size);
    thread t2(write_to_video, output_path);

    t0.join();
    t1.join();
    t2.join();

    image_que.join();
    result_que.join();
    cout << "\n\nAll finished......" << endl;
    time_t time2(time_t * timer);

    double ret = (clock() - prevTimestamp) / (double)CLOCKS_PER_SEC;
    cout << "used time: " << ret << endl;
    return;
}

int main(int argc, char** argv) {
    YoLoForward();
    return EXIT_SUCCESS;
}