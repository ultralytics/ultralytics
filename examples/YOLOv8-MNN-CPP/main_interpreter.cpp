// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <regex>
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
    Inference() : interpreter(nullptr), session(nullptr), inputTensor(nullptr) {
        inputDims = {1, 3, 640, 640};
    }

    ~Inference() {
        if(interpreter) {
            delete interpreter;
            interpreter = nullptr;
        }
    }

    // Load model, create session, and resize the input tensor.
    bool loadModel(const std::string &modelPath,
                   int forwardType = MNN_FORWARD_CPU,
                   int precision = 1,
                   int thread = 4) {
        MNN::ScheduleConfig sConfig;
        sConfig.type = static_cast<MNNForwardType>(forwardType);
        sConfig.numThread = thread;
        BackendConfig bConfig;
        bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
        sConfig.backendConfig = &bConfig;

        interpreter = MNN::Interpreter::createFromFile(modelPath.c_str());
        if (!interpreter) {
            MNN_PRINT("Error: Failed to create interpreter from model file.\n");
            return false;
        }
        session = interpreter->createSession(sConfig);
        if(!session) {
            MNN_PRINT("Error: Failed to create session.\n");
            return false;
        }
        inputTensor = interpreter->getSessionInput(session, "images");
        interpreter->resizeTensor(inputTensor, inputDims);
        interpreter->resizeSession(session);
        std::string bizCode = interpreter->bizCode();

        // Get names from bizCode.
        auto names_start = bizCode.find("\"names\": {");
        if (names_start == std::string::npos) {
            MNN_PRINT("No names found in bizCode, setting classNames empty.\n");
            classNames.clear();
        } else {
            auto names_end = bizCode.find("}", names_start);
            if (names_end == std::string::npos) {
                MNN_PRINT("No closing brace for names in bizCode, setting classNames empty.\n");
                classNames.clear();
            } else {
                std::string namesDict = bizCode.substr(names_start + 10, names_end - names_start - 10);
                parseClassNamesFromBizCode(namesDict);
            }
        }
        return true;
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

    VARP preprocess(VARP &originalImage, int targetSize, float &scale) {
        const auto dims = originalImage->getInfo()->dim;
        const int ih = dims[0], iw = dims[1];
        const int len = (ih >= iw ? ih : iw);
        scale = static_cast<float>(len) / targetSize;

        // Use fixed-size array for padding values.
        int padvals[6] = { 0, len - ih, 0, len - iw, 0, 0 };
        auto pads = _Const(static_cast<void*>(padvals), {3, 2}, NCHW, halide_type_of<int>());
        auto padded  = _Pad(originalImage, pads, CONSTANT);

        auto resized = MNN::CV::resize(padded, MNN::CV::Size(targetSize, targetSize),
                                        0, 0, MNN::CV::INTER_LINEAR, -1,
                                        {0.f, 0.f, 0.f},
                                        {1.f/255, 1.f/255, 1.f/255});

        // Chain unsqueeze and conversion
        auto input = _Unsqueeze(resized, {0});
        input = _Convert(input, NCHW);
        return input;
    }

    // Run inference by copying preprocessed data into input tensor.
    void runInference(VARP input) {
        auto tmp_input = MNN::Tensor::create(inputDims, halide_type_of<float>(),
                                              const_cast<void*>(input->readMap<void>()),
                                              MNN::Tensor::CAFFE);
        inputTensor->copyFromHostTensor(tmp_input);
        interpreter->runSession(session);
    }

    // Postprocess the output, perform NMS, and draw bounding boxes on originalImage.
    void postprocess(float scale, VARP originalImage, float modelScoreThreshold = 0.25, float modelNMSThreshold = 0.45) {
        auto outputTensor = interpreter->getSessionOutput(session, "output0");

        // ---------------- Post Processing ----------------
        auto outputs = outputTensor->host<float>();
        auto outputVar = _Const(outputs, outputTensor->shape(), NCHW, halide_type_of<float>());
        auto output = _Squeeze(_Convert(outputVar, NCHW));

        // Expected output shape: [84, 8400] where first 4 rows are [cx, cy, w, h].
        auto cx = _Gather(output, _Scalar<int>(0));
        auto cy = _Gather(output, _Scalar<int>(1));
        auto w  = _Gather(output, _Scalar<int>(2));
        auto h  = _Gather(output, _Scalar<int>(3));

        // Slice probability values (starting at row 4).
        const int startArr[2] = { 4, 0 };
        const int sizeArr[2]  = { -1, -1 };
        auto start = _Const(static_cast<void*>(const_cast<int*>(startArr)), {2}, NCHW, halide_type_of<int>());
        auto size  = _Const(static_cast<void*>(const_cast<int*>(sizeArr)), {2}, NCHW, halide_type_of<int>());
        auto probs = _Slice(output, start, size);

        // Convert [cx, cy, w, h] to [y1, x1, y2, x2] using half-width/height.
        auto half = _Const(0.5);
        auto x1 = cx - w * half;
        auto y1 = cy - h * half;
        auto x2 = cx + w * half;
        auto y2 = cy + h * half;
        auto boxes = _Stack({x1, y1, x2, y2}, 1);

        auto scores = _ReduceMax(probs, {0});
        auto ids    = _ArgMax(probs, 0);
        auto result_ids = _Nms(boxes, scores, 100, modelScoreThreshold, modelNMSThreshold);

        auto result_ptr = result_ids->readMap<int>();
        auto box_ptr    = boxes->readMap<float>();
        auto ids_ptr    = ids->readMap<int>();
        auto score_ptr  = scores->readMap<float>();

        const int numResults = result_ids->getInfo()->size;
        for (int i = 0; i < numResults; i++) {
            int idx = result_ptr[i];
            if (idx < 0) break;
            float x1 = box_ptr[idx * 4 + 0] * scale;
            float y1 = box_ptr[idx * 4 + 1] * scale;
            float x2 = box_ptr[idx * 4 + 2] * scale;
            float y2 = box_ptr[idx * 4 + 3] * scale;
            int class_idx = ids_ptr[idx];
            float score = score_ptr[idx];
            printf("Detection: box = {%.2f, %.2f, %.2f, %.2f}, class = %s, score = %.2f\n",
                x1, y1, x2, y2, classNames[class_idx].c_str(), score);
            MNN::CV::rectangle(originalImage, { x1, y1 }, { x2, y2 }, { 0, 255, 0 }, 2);
            // Note: MNN::CV does not offer a putText function.
            // For text annotations, consider converting the image to cv::Mat and using OpenCV.
        }
        if (MNN::CV::imwrite("mnn_yolov8_cpp.jpg", originalImage)) {
            MNN_PRINT("Result image written to `mnn_yolov8_cpp.jpg`.\n");
        }
    }

private:
    MNN::Interpreter* interpreter;
    MNN::Session* session;
    MNN::Tensor* inputTensor;
    std::vector<int> inputDims;
    std::vector<std::string> classNames;
};

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./main yolov8n.mnn input.jpg [backend] [precision] [thread]\n");
        return 0;
    }
    int backend = MNN_FORWARD_CPU;
    int precision   = 1;
    int thread      = 4;
    if (argc >= 4) {
        backend = atoi(argv[3]);
    }
    if (argc >= 5) {
        precision = atoi(argv[4]);
    }
    if (argc >= 6) {
        thread = atoi(argv[5]);
    }

    Inference infer;
    if (!infer.loadModel(argv[1], backend, precision, thread))
        return 1;

    const clock_t begin_time = clock();
    float scale = 1.0f;
    VARP originalImage = imread(argv[2]);
    VARP input = infer.preprocess(originalImage, 640, scale);
    auto preprocess_time = 1000.0 * (clock() - begin_time) / CLOCKS_PER_SEC;

    const clock_t begin_time2 = clock();
    infer.runInference(input);
    auto inference_time = 1000.0 * (clock() - begin_time2) / CLOCKS_PER_SEC;
    const clock_t begin_time3 = clock();
    infer.postprocess(scale, originalImage);

    auto postprocess_time = 1000.0 * (clock() - begin_time3) / CLOCKS_PER_SEC;
    printf("Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess\n",
        preprocess_time, inference_time, postprocess_time);
    return 0;
}
