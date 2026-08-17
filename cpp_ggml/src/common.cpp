// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "common.hpp"

#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace yolo {

int g_log_level = 1;  // INFO

void logf(LogLevel level, const char* fmt, ...) {
    if (static_cast<int>(level) < g_log_level) return;
    static const char* tags[] = {"[debug]", "[info] ", "[warn] ", "[error]"};
    std::fprintf(stderr, "%s ", tags[static_cast<int>(level)]);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
}

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return double(ts.tv_sec) * 1e3 + double(ts.tv_nsec) / 1e6;
}

Clock::Clock() : t0(now_ms()) {}
double Clock::ms_since() const { return now_ms() - t0; }

}  // namespace yolo
