// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace yolo {

// Logging -------------------------------------------------------------------

enum class LogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

void logf(LogLevel level, const char* fmt, ...);
extern int g_log_level;

#define YOLO_LOG_DEBUG(...) ::yolo::logf(::yolo::LogLevel::DEBUG, __VA_ARGS__)
#define YOLO_LOG_INFO(...)  ::yolo::logf(::yolo::LogLevel::INFO, __VA_ARGS__)
#define YOLO_LOG_WARN(...)  ::yolo::logf(::yolo::LogLevel::WARN, __VA_ARGS__)
#define YOLO_LOG_ERROR(...) ::yolo::logf(::yolo::LogLevel::ERROR, __VA_ARGS__)

// Detection result -----------------------------------------------------------

struct Detection {
    float x1, y1, x2, y2;  // pixels in the original input image
    float score;           // max class probability after sigmoid
    int class_id;
};

struct LetterboxInfo {
    float scale;   // resized / original
    int pad_w;     // left padding in pixels (right may differ by 1px)
    int pad_h;
    int new_w;     // resized dims before padding
    int new_h;
    int imgsz_w;   // padded canvas dims
    int imgsz_h;
};

// Model metadata from GGUF ----------------------------------------------------

struct ModelMeta {
    std::string name;
    std::string task;
    std::string dtype;
    int nc = 80;
    int nl = 3;
    int imgsz = 640;
    int reg_max = 16;
    bool end2end = false;
    int max_det = 300;
    std::vector<float> strides;
    std::vector<std::string> class_names;
};

// Timing helper ---------------------------------------------------------------

struct Clock {
    double t0;
    Clock();
    double ms_since() const;
};

}  // namespace yolo
