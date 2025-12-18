// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <algorithm>
#include <regex>
#include <sstream>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cv/cv.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

class Inference {
public:

    // Load model: Create runtime, set cache if needed, and load the model file.
    bool loadModel(const std::string &modelPath,
                   int forwardType = MNN_FORWARD_CPU,
                   int precision = 0,
                   int thread = 4) {
        MNN::ScheduleConfig sConfig;
        sConfig.type = static_cast<MNNForwardType>(forwardType);
        sConfig.numThread = thread;
        BackendConfig bConfig;
        bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
        sConfig.backendConfig = &bConfig;

        std::shared_ptr<Executor::RuntimeManager> rtmgr(
            Executor::RuntimeManager::createRuntimeManager(sConfig)
        );
        if (rtmgr == nullptr) {
            MNN_ERROR("Empty RuntimeManager\n");
            return false;
        }
        rtmgr->setCache(".cachefile");
        net = std::shared_ptr<Module>(Module::load(std::vector<std::string>{},
                              std::vector<std::string>{}, modelPath.c_str(), rtmgr));
        if (net == nullptr) {
            return false;
        }
        runtimeManager = rtmgr;
        const Module::Info* info = net->getInfo();
        if (info == nullptr) {
            MNN_ERROR("Empty Module Info\n");
            return false;
        }
        // Parse bizCode to extract class names.
        if (info->bizCode.empty()) {
            MNN_ERROR("Empty bizCode\n");
            classNames.clear();
            return false;
        }
        // Get imgsz from bizCode.
        auto imgsz_start = info->bizCode.find("\"imgsz\": [");
        if (imgsz_start == std::string::npos) {
            MNN_PRINT("No imgsz found in bizCode, setting classNames empty.\n");
        } else {
            auto imgsz_end = info->bizCode.find("]", imgsz_start);
            if (imgsz_end == std::string::npos) {
                MNN_PRINT("No closing bracket for imgsz in bizCode, setting classNames empty.\n");
            } else {
                std::string imgszText = info->bizCode.substr(imgsz_start + 10, imgsz_end - imgsz_start - 10);
                std::vector<std::string> imgszVec;
                std::stringstream ss(imgszText);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    imgszVec.push_back(item);
                }
            }
        }
        // Get names from bizCode.
        auto names_start = info->bizCode.find("\"names\": {");
        if (names_start == std::string::npos) {
            MNN_PRINT("No names found in bizCode, setting classNames empty.\n");
            classNames.clear();
        } else {
            auto names_end = info->bizCode.find("}", names_start);
            if (names_end == std::string::npos) {
                MNN_PRINT("No closing brace for names in bizCode, setting classNames empty.\n");
                classNames.clear();
            } else {
                std::string namesDict = info->bizCode.substr(names_start + 10, names_end - names_start - 10);
                parseClassNamesFromBizCode(namesDict);
            }
        }
        return true;
    }

    void parseImgszFromBizCode(const std::string& bizText) {
        std::regex rgx("\"imgsz\":\\s*\\[(\\d+),\\s*(\\d+)\\]");
        std::smatch match;
        if (std::regex_search(bizText, match, rgx)) {
            int ih = std::stoi(match[1].str());
            int iw = std::stoi(match[2].str());
            MNN_PRINT("Input size: %d x %d\n", iw, ih);
        } else {
            MNN_PRINT("No imgsz found in bizCode.\n");
        }
    }

    void parseClassNamesFromBizCode(const std::string& bizText) {
        std::regex rgx("\"(\\d+)\"\\s*:\\s*\"([^\"]+)\"");
        std::smatch match;
        std::string s = bizText;
        classNames.clear();
        while (std::regex_search(s, match, rgx)) {
            int index = std::stoi(match[1].str());
            std::string name = match[2].str();
            if (classNames.size() <= static_cast<size_t>(index)) {
                classNames.resize(index + 1);
            }
            classNames[index] = name;
            s = match.suffix().str();
        }
    }

    VARP preprocess(VARP &originalImage, float &scale) {
        auto dims = originalImage->getInfo()->dim;
        int ih = dims[0];
        int iw = dims[1];
        int targetWidth  = std::stoi(imgszVec[0]);
        int targetHeight = std::stoi(imgszVec[1]);
        int len = ih > iw ? ih : iw;
        scale = static_cast<float>(len) / std::max(targetWidth, targetHeight);
        std::vector<int> padvals { 0, len - ih, 0, len - iw, 0, 0 };
        auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
        auto image = _Pad(originalImage, pads, CONSTANT);
        image = resize(image, Size(targetWidth, targetHeight), 0, 0, INTER_LINEAR, -1, {0., 0., 0.}, {1./255., 1./255., 1./255.});
        auto input = _Unsqueeze(image, {0});
        input = _Convert(input, NC4HW4);
        return input;
    }

    void runInference(VARP input) {
        std::vector<VARP> outputs = net->onForward({input});
        mOutput = outputs[0];
    }

    void postprocess(float scale, VARP originalImage, float iouThreshold = 0.45, float scoreThreshold = 0.25) {
        auto output = _Convert(mOutput, NCHW);
        output = _Squeeze(output);
        // Expected output shape: [84, 8400]
        auto cx = _Gather(output, _Scalar<int>(0));
        auto cy = _Gather(output, _Scalar<int>(1));
        auto w  = _Gather(output, _Scalar<int>(2));
        auto h  = _Gather(output, _Scalar<int>(3));

        std::vector<int> startvals { 4, 0 };
    auto start = _Const(static_cast<void*>(startvals.data()), {2}, NCHW, halide_type_of<int>());
    std::vector<int> sizevals { -1, -1 };
    auto size = _Const(static_cast<void*>(sizevals.data()), {2}, NCHW, halide_type_of<int>());
    auto probs = _Slice(output, start, size);
    // [cx, cy, w, h] -> [x1, y1, x2, y2]
    auto x1 = cx - w * _Const(0.5);
    auto y1 = cy - h * _Const(0.5);
    auto x2 = cx + w * _Const(0.5);
    auto y2 = cy + h * _Const(0.5);
    auto boxes = _Stack({x1, y1, x2, y2}, 1);
    auto scores = _ReduceMax(probs, {0});
    auto ids = _ArgMax(probs, 0);
    auto result_ids = _Nms(boxes, scores, 100, 0.45, 0.25);
    auto result_ptr = result_ids->readMap<int>();
    auto box_ptr = boxes->readMap<float>();
    auto ids_ptr = ids->readMap<int>();
    auto score_ptr = scores->readMap<float>();
    for (int i = 0; i < 100; i++) {
        auto idx = result_ptr[i];
        if (idx < 0) break;
        auto x1 = box_ptr[idx * 4 + 0] * scale;
        auto y1 = box_ptr[idx * 4 + 1] * scale;
        auto x2 = box_ptr[idx * 4 + 2] * scale;
        auto y2 = box_ptr[idx * 4 + 3] * scale;
        auto class_idx = ids_ptr[idx];
        auto score = score_ptr[idx];
            printf("Detection: box = {%.2f, %.2f, %.2f, %.2f}, class = %s, score = %.2f\n",
                x1, y1, x2, y2, classNames[class_idx].c_str(), score);
            rectangle(originalImage, { x1, y1 }, { x2, y2 }, { 0, 255, 0 }, 2);
        }
        if (imwrite("mnn_yolov8_cpp.jpg", originalImage)) {
            MNN_PRINT("Result image written to `mnn_yolov8_cpp.jpg`.\n");
        }
    }

    // Update runtime cache.
    void updateCache() {
        if (runtimeManager)
            runtimeManager->updateCache();
    }

    private:
    std::shared_ptr<Module> net;
    VARP mOutput;
    std::shared_ptr<Executor::RuntimeManager> runtimeManager;
    std::vector<std::string> classNames;
    std::vector<std::string> imgszVec { "640", "640" };
};

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./main yolov8n.mnn bus.jpg [forwardType] [precision] [thread]\n");
        return 0;
    }
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc >= 4) {
        forwardType = atoi(argv[3]);
    }
    if (argc >= 5) {
        precision = atoi(argv[4]);
    }
    if (argc >= 6) {
        thread = atoi(argv[5]);
    }

    Inference infer;
    if (!infer.loadModel(argv[1], forwardType, precision, thread))
        return 1;

    const clock_t t0 = clock();
    float scale = 1.0f;
    VARP originalImage = imread(argv[2]);
    VARP input = infer.preprocess(originalImage, scale);
    double preprocess_time = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;

    const clock_t t1 = clock();
    infer.runInference(input);
    double inference_time = 1000.0 * (clock() - t1) / CLOCKS_PER_SEC;

    const clock_t t2 = clock();
    infer.postprocess(scale, originalImage);
    double postprocess_time = 1000.0 * (clock() - t2) / CLOCKS_PER_SEC;

    printf("Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess\n",
           preprocess_time, inference_time, postprocess_time);

    infer.updateCache();
    return 0;
}
