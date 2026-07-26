// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
// Tiny command-line helpers shared by the Ultralytics YOLO C++ example `main`s.
// Header-only: add examples/cpp/common to your include path.

#pragma once

#include <string>
#include <vector>

namespace yolo {

// Return the value following `key` on the command line, or `fallback` if absent.
inline std::string ArgValue(int argc, char** argv, const std::string& key, const std::string& fallback) {
    for (int i = 1; i < argc - 1; ++i) {
        if (key == argv[i]) return argv[i + 1];
    }
    return fallback;
}

// True if the boolean flag `key` is present on the command line.
inline bool HasFlag(int argc, char** argv, const std::string& key) {
    for (int i = 1; i < argc; ++i) {
        if (key == argv[i]) return true;
    }
    return false;
}

// Class name for `id`, or the numeric id as a string when out of range (e.g. a
// 1000-class model with only the 80 COCO names available).
inline std::string NameOf(const std::vector<std::string>& names, int id) {
    return id >= 0 && id < static_cast<int>(names.size()) ? names[id] : std::to_string(id);
}

}  // namespace yolo
