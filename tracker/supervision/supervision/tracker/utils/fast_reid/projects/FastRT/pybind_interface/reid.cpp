#include <iostream>
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "fastrt/utils.h"
#include "fastrt/baseline.h"
#include "fastrt/factory.h"
using namespace fastrt;
using namespace nvinfer1;

namespace py = pybind11;


/* Ex1. sbs_R50-ibn */
static const std::string WEIGHTS_PATH = "../sbs_R50-ibn.wts"; 
static const std::string ENGINE_PATH = "./sbs_R50-ibn.engine";

static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 2048;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r50; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = true; 
static const bool WITH_NL = true;
static const int EMBEDDING_DIM = 0; 

FastreidConfig reidCfg { 
        BACKBONE,
        HEAD,
        HEAD_POOLING,
        LAST_STRIDE,
        WITH_IBNA,
        WITH_NL,
        EMBEDDING_DIM};

class ReID
{

private:
    int device;  // GPU id
    fastrt::Baseline baseline;

public:
    ReID(int a);
    int build(const std::string &engine_file);
    // std::list<float> infer_test(const std::string &image_file);
    std::list<float> infer(py::array_t<uint8_t>&);
    std::list<std::list<float>> batch_infer(std::list<py::array_t<uint8_t>>&);
    ~ReID();
};

ReID::ReID(int device): baseline(trt::ModelConfig { 
        WEIGHTS_PATH,
        MAX_BATCH_SIZE,
        INPUT_H,
        INPUT_W,
        OUTPUT_SIZE,
        device})
{
    std::cout << "Init on device " << device << std::endl;
}

int ReID::build(const std::string &engine_file)
{
    if(!baseline.deserializeEngine(engine_file)) {
        std::cout << "DeserializeEngine Failed." << std::endl;
        return -1;
    }
    return 0;
}

ReID::~ReID()
{

    std::cout << "Destroy engine succeed" << std::endl;
}

std::list<float> ReID::infer(py::array_t<uint8_t>& img)
{
    auto rows = img.shape(0);
    auto cols = img.shape(1);
    auto type = CV_8UC3;

    cv::Mat img2(rows, cols, type, (unsigned char*)img.data());
    cv::Mat re(INPUT_H, INPUT_W, CV_8UC3);
    // std::cout << (int)img2.data[0] << std::endl;
    cv::resize(img2, re, re.size(), 0, 0, cv::INTER_CUBIC); /* cv::INTER_LINEAR */
    std::vector<cv::Mat> input;
    input.emplace_back(re);

    if(!baseline.inference(input)) {
        std::cout << "Inference Failed." << std::endl;
    }
    std::list<float> feature;

    float* feat_embedding = baseline.getOutput();
    TRTASSERT(feat_embedding);
    for (int dim = 0; dim < baseline.getOutputSize(); ++dim) {
        feature.push_back(feat_embedding[dim]);
    }

    return feature;
}


std::list<std::list<float>> ReID::batch_infer(std::list<py::array_t<uint8_t>>& imgs)
{
    // auto t1 = Time::now();
    std::vector<cv::Mat> input;
    int count = 0;
    while(!imgs.empty()){
        py::array_t<uint8_t>& img = imgs.front();
        imgs.pop_front();
        // parse to cvmat
        auto rows = img.shape(0);
        auto cols = img.shape(1);
        auto type = CV_8UC3;

        cv::Mat img2(rows, cols, type, (unsigned char*)img.data());
        cv::Mat re(INPUT_H, INPUT_W, CV_8UC3);
        // std::cout << (int)img2.data[0] << std::endl;
        cv::resize(img2, re, re.size(), 0, 0, cv::INTER_CUBIC); /* cv::INTER_LINEAR */
        input.emplace_back(re);

        count += 1;
    }
    // auto t2 = Time::now();
    
    if(!baseline.inference(input)) {
        std::cout << "Inference Failed." << std::endl;
    }
    std::list<std::list<float>> result;

    float* feat_embedding = baseline.getOutput();
    TRTASSERT(feat_embedding);

    // auto t3 = Time::now();
    for (int index = 0; index < count; index++)
    {
        std::list<float> feature;
        for (int dim = 0; dim < baseline.getOutputSize(); ++dim) {
            feature.push_back(feat_embedding[index * baseline.getOutputSize() + dim]);
        }
        result.push_back(feature);
    }
    // std::cout << "[Preprocessing]: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "ms" 
    // << "[Infer]: " << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count() << "ms" 
    // << "[Cast]: " << std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() - t3).count() << "ms" 
    // << std::endl; 
    return result;
}


PYBIND11_MODULE(ReID, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
    )pbdoc";
    py::class_<ReID>(m, "ReID")
        .def(py::init<int>())
        .def("build", &ReID::build)
        .def("infer", &ReID::infer, py::return_value_policy::automatic)
        .def("batch_infer", &ReID::batch_infer, py::return_value_policy::automatic)
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
