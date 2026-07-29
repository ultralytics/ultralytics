#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "coco_names.hpp"
#include "latest_queue.hpp"
#include "tensorrt_detector.hpp"
#include "yolo_draw.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Options {
    std::string engine_path;
    std::string labels_path;
    int camera = 0;
    int camera_width = 0;
    int camera_height = 0;
    float confidence = 0.25f;
    float iou = 0.45f;
    std::size_t input_queue_size = 2;
    std::size_t output_queue_size = 2;
    bool help = false;
};

struct FramePacket {
    Clock::time_point captured_at;
    cv::Mat image;
};

struct DisplayPacket {
    FramePacket frame;
    std::vector<yolo::Result> detections;
    yolo::InferenceTiming timing;
};

void PrintUsage() {
    std::cout
        << "YOLOv8 TensorRT three-thread asynchronous camera pipeline\n\n"
        << "Usage:\n"
        << "  yolo_tensorrt_async --engine <model.engine> [options]\n\n"
        << "Options:\n"
        << "  --camera <index>       Camera index (default: 0)\n"
        << "  --width <pixels>       Requested capture width\n"
        << "  --height <pixels>      Requested capture height\n"
        << "  --conf <0..1>          Confidence threshold (default: 0.25)\n"
        << "  --iou <0..1>           NMS IoU threshold (default: 0.45)\n"
        << "  --labels <file>        One class name per line (default: COCO)\n"
        << "  --input-queue <count>  Capture queue capacity (default: 2)\n"
        << "  --output-queue <count> Display queue capacity (default: 2)\n"
        << "  --help                 Show this help\n\n"
        << "Press Q or Esc in the display window to stop.\n";
}

std::string RequireValue(int& index, int argc, char** argv) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[index]);
    return argv[++index];
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--help" || argument == "-h") {
            options.help = true;
        } else if (argument == "--engine") {
            options.engine_path = RequireValue(i, argc, argv);
        } else if (argument == "--camera") {
            options.camera = std::stoi(RequireValue(i, argc, argv));
        } else if (argument == "--width") {
            options.camera_width = std::stoi(RequireValue(i, argc, argv));
        } else if (argument == "--height") {
            options.camera_height = std::stoi(RequireValue(i, argc, argv));
        } else if (argument == "--conf") {
            options.confidence = std::stof(RequireValue(i, argc, argv));
        } else if (argument == "--iou") {
            options.iou = std::stof(RequireValue(i, argc, argv));
        } else if (argument == "--labels") {
            options.labels_path = RequireValue(i, argc, argv);
        } else if (argument == "--input-queue") {
            options.input_queue_size = std::stoul(RequireValue(i, argc, argv));
        } else if (argument == "--output-queue") {
            options.output_queue_size = std::stoul(RequireValue(i, argc, argv));
        } else {
            throw std::invalid_argument("unknown option: " + argument);
        }
    }

    if (!options.help && options.engine_path.empty()) throw std::invalid_argument("--engine is required");
    if (options.camera < 0) throw std::invalid_argument("--camera must be zero or greater");
    if (options.camera_width < 0 || options.camera_height < 0) {
        throw std::invalid_argument("--width and --height must be zero or greater");
    }
    if (options.confidence < 0.0f || options.confidence > 1.0f) {
        throw std::invalid_argument("--conf must be between 0 and 1");
    }
    if (options.iou < 0.0f || options.iou > 1.0f) {
        throw std::invalid_argument("--iou must be between 0 and 1");
    }
    if (options.input_queue_size == 0 || options.output_queue_size == 0) {
        throw std::invalid_argument("queue capacities must be greater than zero");
    }
    return options;
}

std::vector<std::string> LoadLabels(const std::string& path, int class_count) {
    if (path.empty()) {
        if (class_count == static_cast<int>(yolo::CocoNames().size())) return yolo::CocoNames();
        std::vector<std::string> labels;
        labels.reserve(static_cast<std::size_t>(class_count));
        for (int i = 0; i < class_count; ++i) labels.push_back("class_" + std::to_string(i));
        return labels;
    }

    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open labels file: " + path);

    std::vector<std::string> labels;
    std::string label;
    while (std::getline(file, label)) {
        if (!label.empty() && label.back() == '\r') label.pop_back();
        if (!label.empty()) labels.push_back(label);
    }
    if (labels.empty()) throw std::runtime_error("labels file is empty: " + path);
    return labels;
}

void DrawOverlay(
    cv::Mat& image,
    const DisplayPacket& packet,
    const std::vector<std::string>& labels,
    double display_fps,
    std::size_t input_drops,
    std::size_t output_drops
) {
    for (const yolo::Result& detection : packet.detections) {
        const std::string name =
            detection.class_id >= 0 && detection.class_id < static_cast<int>(labels.size())
                ? labels[static_cast<std::size_t>(detection.class_id)]
                : "class_" + std::to_string(detection.class_id);
        yolo::DrawBox(
            image, detection.box, yolo::Label(name, detection.confidence), detection.class_id
        );
    }

    const float end_to_end_ms =
        std::chrono::duration<float, std::milli>(Clock::now() - packet.frame.captured_at).count();
    std::ostringstream status;
    status << std::fixed << std::setprecision(1)
           << "FPS " << display_fps
           << " | pre " << packet.timing.preprocess_ms
           << " ms | GPU " << packet.timing.gpu_ms
           << " ms | post " << packet.timing.postprocess_ms
           << " ms | E2E " << end_to_end_ms
           << " ms | drop " << input_drops << "/" << output_drops;
    const std::string text = status.str();
    cv::putText(
        image, text, cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 3, cv::LINE_AA
    );
    cv::putText(
        image, text, cv::Point(16, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 1, cv::LINE_AA
    );
}

int Run(const Options& options) {
    yolo::TensorRTDetector detector(options.engine_path, options.confidence, options.iou);
    const std::vector<std::string> labels = LoadLabels(options.labels_path, detector.classCount());
    std::cout << "Engine ready: " << detector.inputWidth() << 'x' << detector.inputHeight()
              << ", classes=" << detector.classCount() << '\n';
    if (labels.size() != static_cast<std::size_t>(detector.classCount())) {
        std::cerr << "Warning: loaded " << labels.size() << " labels for " << detector.classCount()
                  << " model classes; missing names will use class_<id>.\n";
    }

    yolo::LatestQueue<FramePacket> input_queue(options.input_queue_size);
    yolo::LatestQueue<DisplayPacket> output_queue(options.output_queue_size);
    std::atomic_bool stop{false};
    std::atomic_uint64_t captured{0};
    std::atomic_uint64_t inferred{0};
    std::atomic_uint64_t displayed{0};
    std::mutex error_mutex;
    std::string worker_error;

    const auto stop_pipeline = [&] {
        stop.store(true, std::memory_order_relaxed);
        input_queue.close();
        output_queue.close();
    };
    const auto fail = [&](const char* worker, const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (worker_error.empty()) worker_error = std::string(worker) + ": " + message;
        }
        stop_pipeline();
    };

    std::thread capture_thread([&] {
        try {
            cv::VideoCapture camera(options.camera, cv::CAP_ANY);
            if (!camera.isOpened()) {
                throw std::runtime_error("cannot open camera " + std::to_string(options.camera));
            }
            if (options.camera_width > 0) camera.set(cv::CAP_PROP_FRAME_WIDTH, options.camera_width);
            if (options.camera_height > 0) camera.set(cv::CAP_PROP_FRAME_HEIGHT, options.camera_height);
            camera.set(cv::CAP_PROP_BUFFERSIZE, 1);

            while (!stop.load(std::memory_order_relaxed)) {
                cv::Mat frame;
                if (!camera.read(frame) || frame.empty()) break;
                FramePacket packet{Clock::now(), std::move(frame)};
                if (!input_queue.push(std::move(packet))) break;
                captured.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& error) {
            fail("capture thread", error.what());
        } catch (...) {
            fail("capture thread", "unknown error");
        }
        input_queue.close();
    });

    std::thread inference_thread([&] {
        try {
            while (auto frame = input_queue.popLatest()) {
                if (stop.load(std::memory_order_relaxed)) break;
                yolo::InferenceTiming timing;
                std::vector<yolo::Result> detections = detector.infer(frame->image, timing);
                DisplayPacket packet{std::move(*frame), std::move(detections), timing};
                if (!output_queue.push(std::move(packet))) break;
                inferred.fetch_add(1, std::memory_order_relaxed);
            }
        } catch (const std::exception& error) {
            fail("inference thread", error.what());
        } catch (...) {
            fail("inference thread", "unknown error");
        }
        output_queue.close();
    });

    std::thread display_thread([&] {
        try {
            constexpr const char* window_name = "YOLOv8 TensorRT Async";
            cv::namedWindow(window_name, cv::WINDOW_NORMAL);
            auto previous_frame_at = Clock::time_point{};
            double display_fps = 0.0;

            while (auto packet = output_queue.popLatest()) {
                const auto now = Clock::now();
                if (previous_frame_at != Clock::time_point{}) {
                    const double seconds = std::chrono::duration<double>(now - previous_frame_at).count();
                    const double instant_fps = seconds > 0.0 ? 1.0 / seconds : 0.0;
                    display_fps = display_fps == 0.0 ? instant_fps : display_fps * 0.9 + instant_fps * 0.1;
                }
                previous_frame_at = now;

                DrawOverlay(
                    packet->frame.image,
                    *packet,
                    labels,
                    display_fps,
                    input_queue.dropped(),
                    output_queue.dropped()
                );
                cv::imshow(window_name, packet->frame.image);
                displayed.fetch_add(1, std::memory_order_relaxed);

                const int key = cv::waitKey(1);
                if (key == 27 || key == 'q' || key == 'Q' ||
                    cv::getWindowProperty(window_name, cv::WND_PROP_VISIBLE) < 1.0) {
                    stop_pipeline();
                    break;
                }
            }
            cv::destroyWindow(window_name);
        } catch (const std::exception& error) {
            fail("display thread", error.what());
        } catch (...) {
            fail("display thread", "unknown error");
        }
    });

    capture_thread.join();
    inference_thread.join();
    display_thread.join();

    {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!worker_error.empty()) throw std::runtime_error(worker_error);
    }
    std::cout << "Stopped. captured=" << captured.load() << ", inferred=" << inferred.load()
              << ", displayed=" << displayed.load() << ", input_drops=" << input_queue.dropped()
              << ", output_drops=" << output_queue.dropped() << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        if (options.help) {
            PrintUsage();
            return 0;
        }
        return Run(options);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n\n";
        PrintUsage();
        return 1;
    }
}
