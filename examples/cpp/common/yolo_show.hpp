// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// Shared display helper for the Ultralytics YOLO C++ examples. Every example
// always writes/prints its result; pass --show on the command line to also open
// a window. Header-only: add examples/cpp/common to your include path.

#pragma once

#include <string>
#include <opencv2/highgui.hpp>

namespace yolo {

// True if "--show" (or "-show") was passed on the command line.
inline bool ShowRequested(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--show" || a == "-show") return true;
    }
    return false;
}

// Display the annotated image in a window and wait for a key, but only if `show`
// is set. Safe to call on headless machines when `show` is false.
inline void Show(const std::string& title, const cv::Mat& image, bool show) {
    if (!show) return;
    cv::imshow(title, image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

}  // namespace yolo
