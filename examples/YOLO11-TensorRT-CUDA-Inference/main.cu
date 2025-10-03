#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "TRTHandle.h"
#include <string>


float generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols;
    int origin_h = image.rows;

    int target_h = target_size[0];
    int target_w = target_size[1];

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h);
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w);
    float resize_scale = std::min(ratio_h, ratio_w);
    return resize_scale;
}


float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114., 114., 114));
    return resize_scale;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " -input <path_to_input_image> -model <path_to_yolo11_engine>" << std::endl;
        return 1;
    }
    std::string image_path;
    std::string model_path;
    int width = 640;
    int height = 640;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-input") {
            image_path = argv[++i];
        } else if (std::string(argv[i]) == "-model") {
            model_path = argv[++i];
        }
    }

    cv::Mat image = cv::imread(image_path);
    cv::Mat input_image;
    
    letterbox(image, input_image, {640, 640});
    if (input_image.empty()) {
        std::cerr << "Error: Something went wrong during letterboxing!" << std::endl;
        return 1;
    }

    // Fetch the model from the specified path
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        auto msg = "Error, unable to open engine file: " + std::string(model_path);
        throw std::runtime_error(msg);
    }
    
    std::streamsize size = file.tellg();
    std::vector<char> modelData(size);
    file.seekg(0, file.beg);
    file.read(reinterpret_cast<char*>(modelData.data()), size);
    file.close();

    if (modelData.empty()) {
        std::cerr << "Failed to read model from file: " << model_path << std::endl;
        std::cerr << "The Engine file path must be given. If it is not present, it can be generated using the build program present in current directory." << std::endl;
        return 1;
    }

    // Sets the device to GPU 0 by default
    checkCudaErrorCode(cudaSetDevice(0));

    // Deserialize the engine
    IRuntime* runtime = createInferRuntime(logger);

    std::cout << "Deserializing model from " << model_path << ", size " << size << " bytes" << std::endl;
    ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());

    IExecutionContext *context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context from engine" << std::endl;
        return 1;
    }

    // CUDA stream for inference
    cudaStream_t stream;
    checkCudaErrorCode(cudaStreamCreate(&stream));

    // Initialize I/O buffer vectors
    Buffer buffers(engine, stream);
    std::cout << "Buffer initialized" << std::endl;

    std::vector<void*> buffersVec;
    buffers.getBuffers(buffersVec);

    std::vector<std::string> IOTensorNames;
    buffers.getIOTensorNames(IOTensorNames);

    auto outputLength = buffers.getOutputLength();
    auto inputLength = buffers.getInputLength();

    std::cout << "Input length: " << inputLength << ", Output length: " << outputLength << std::endl;

    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const auto name = engine->getIOTensorName(i);
        const auto mode = engine->getTensorIOMode(name);
        const auto shape = engine->getTensorShape(name);
        const auto dtype = engine->getTensorDataType(name);
        std::cout << "Tensor " << i << ": " << name
                << ", Mode: " << static_cast<int32_t>(mode)
                << ", Shape rank: " << shape.nbDims;
        for (int j = 0; j < shape.nbDims; j++) {
            std::cout << ", Dim[" << j << "]: " << shape.d[j];
        }
        std::cout << ", Data type: " << static_cast<int32_t>(dtype)
                << std::endl;
    }

    if (! context->setTensorAddress(IOTensorNames[0].c_str(), buffersVec[0])) {
        std::cerr << "Failed to set input tensor address: " << IOTensorNames[0] << std::endl;
        return 1;
    }

    if (! context->setTensorAddress(IOTensorNames[1].c_str(), buffersVec[1])) {
        std::cerr << "Failed to set output tensor address: " << IOTensorNames[1] << std::endl;
        return 1;
    }

    // Preprocess the input image
    // TensorRT model expects input in NCHW format
    cv::Mat input_tensor = cv::dnn::blobFromImage(input_image, 1.0 / 255.0, cv::Size(width, height), cv::Scalar(0, 0, 0), true, false);

    // Copy the input tensor data to the GPU
    checkCudaErrorCode(cudaMemcpyAsync(buffersVec[0], input_tensor.data, input_tensor.total() * input_tensor.elemSize(), cudaMemcpyHostToDevice, stream));
    std::cout << "Input tensor copied to GPU" << std::endl;

    // Perform inference
    if (! context->enqueueV3(stream)) {
        std::cerr << "Inference failed" << std::endl;
    }

    checkCudaErrorCode(cudaStreamSynchronize(stream));

    // Fetch the output from the GPU
    std::vector<float> output_data(outputLength);
    checkCudaErrorCode(cudaMemcpyAsync(output_data.data(), buffersVec[1], outputLength * sizeof(float), cudaMemcpyDeviceToHost, stream));
    std::cout << "Output tensor copied to CPU" << std::endl;
    checkCudaErrorCode(cudaStreamSynchronize(stream));

    // draw bounding boxes on the input image
    for (size_t i = 0; i < output_data.size(); i += 6) {
        auto confidence = output_data[i + 4];
        if (confidence > 0.5) { // Threshold for confidence
            // The coordinates are supposed to be normalized for the original image size
            int x1 = static_cast<int>(output_data[i]);
            int y1 = static_cast<int>(output_data[i + 1]);
            int x2 = static_cast<int>(output_data[i + 2]);
            int y2 = static_cast<int>(output_data[i + 3]);
            std::cout << "Detection: [" << x1 << ", " << y1 << ", " << x2 << ", " << y2 << "] with confidence " << confidence << std::endl;
            cv::rectangle(input_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
            std::string label = "Confidence: " + std::to_string(confidence);
            cv::putText(input_image, label, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("Output", input_image);
    cv::waitKey(0);

    // Cleanup
    checkCudaErrorCode(cudaStreamDestroy(stream));

    return 0;
}