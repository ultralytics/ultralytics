// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "image_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace yolo {

bool load_image(const std::string& path, Image& img) {
    int n = 0;
    uint8_t* data = stbi_load(path.c_str(), &img.w, &img.h, &n, 3);
    if (!data) {
        YOLO_LOG_ERROR("failed to load image %s: %s", path.c_str(), stbi_failure_reason());
        return false;
    }
    img.c = 3;
    img.rgb.assign(data, data + (size_t)img.w * img.h * 3);
    stbi_image_free(data);
    return true;
}

// Bilinear resize matching cv2.INTER_LINEAR: src coord = (dst + 0.5) * s - 0.5.
static void resize_bilinear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    const float fx = (float)sw / dw;
    const float fy = (float)sh / dh;
    for (int y = 0; y < dh; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int y0 = (int)std::floor(sy);
        const int y1 = y0 + 1;
        const float wy = sy - y0;
        const int yc0 = std::clamp(y0, 0, sh - 1);
        const int yc1 = std::clamp(y1, 0, sh - 1);
        for (int x = 0; x < dw; x++) {
            const float sx = (x + 0.5f) * fx - 0.5f;
            const int x0 = (int)std::floor(sx);
            const int x1 = x0 + 1;
            const float wx = sx - x0;
            const int xc0 = std::clamp(x0, 0, sw - 1);
            const int xc1 = std::clamp(x1, 0, sw - 1);
            for (int c = 0; c < 3; c++) {
                const float v00 = src[(yc0 * sw + xc0) * 3 + c];
                const float v01 = src[(yc0 * sw + xc1) * 3 + c];
                const float v10 = src[(yc1 * sw + xc0) * 3 + c];
                const float v11 = src[(yc1 * sw + xc1) * 3 + c];
                const float v0 = v00 + (v01 - v00) * wx;
                const float v1 = v10 + (v11 - v10) * wx;
                dst[(y * dw + x) * 3 + c] = (uint8_t)(v0 + (v1 - v0) * wy + 0.5f);
            }
        }
    }
}

static std::vector<float> resize_bilinear_float(const float* src, int sw, int sh, int dw, int dh) {
    std::vector<float> dst((size_t)dw * dh);
    const float fx = (float)sw / dw;
    const float fy = (float)sh / dh;
    for (int y = 0; y < dh; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int y0 = (int)std::floor(sy), y1 = y0 + 1;
        const float wy = sy - y0;
        const int yc0 = std::clamp(y0, 0, sh - 1), yc1 = std::clamp(y1, 0, sh - 1);
        for (int x = 0; x < dw; x++) {
            const float sx = (x + 0.5f) * fx - 0.5f;
            const int x0 = (int)std::floor(sx), x1 = x0 + 1;
            const float wx = sx - x0;
            const int xc0 = std::clamp(x0, 0, sw - 1), xc1 = std::clamp(x1, 0, sw - 1);
            const float v0 = src[(size_t)yc0 * sw + xc0] * (1.0f - wx) + src[(size_t)yc0 * sw + xc1] * wx;
            const float v1 = src[(size_t)yc1 * sw + xc0] * (1.0f - wx) + src[(size_t)yc1 * sw + xc1] * wx;
            dst[(size_t)y * dw + x] = v0 * (1.0f - wy) + v1 * wy;
        }
    }
    return dst;
}

std::vector<float> letterbox_image(const Image& img, int imgsz, LetterboxInfo& info) {
    const float r = std::min((float)imgsz / img.w, (float)imgsz / img.h);
    // nearbyint = round-half-to-even, matching Python round().
    const int new_w = (int)std::nearbyint(img.w * r);
    const int new_h = (int)std::nearbyint(img.h * r);

    // Ultralytics LetterBox(auto=True, center=True): mod stride first, then split padding.
    int dw = (imgsz - new_w) % 32, dh = (imgsz - new_h) % 32;
    const float hw = dw / 2.0f, hh = dh / 2.0f;
    const int left = (int)std::nearbyint(hw - 0.1f), right = (int)std::nearbyint(hw + 0.1f);
    const int top = (int)std::nearbyint(hh - 0.1f), bottom = (int)std::nearbyint(hh + 0.1f);
    const int canvas_w = new_w + left + right;
    const int canvas_h = new_h + top + bottom;

    info = LetterboxInfo{r, left, top, new_w, new_h, canvas_w, canvas_h};

    std::vector<uint8_t> resized((size_t)new_w * new_h * 3);
    resize_bilinear(img.rgb.data(), img.w, img.h, resized.data(), new_w, new_h);

    std::vector<uint8_t> canvas((size_t)canvas_w * canvas_h * 3, 114);  // ultralytics pad color
    for (int y = 0; y < new_h; y++) {
        memcpy(canvas.data() + ((size_t)(y + top) * canvas_w + left) * 3,
               resized.data() + (size_t)y * new_w * 3, (size_t)new_w * 3);
    }

    // HWC RGB -> CHW float /255
    std::vector<float> out((size_t)3 * canvas_h * canvas_w);
    const size_t plane = (size_t)canvas_w * canvas_h;
    for (size_t i = 0; i < plane; i++) {
        out[i] = canvas[i * 3 + 0] / 255.0f;
        out[plane + i] = canvas[i * 3 + 1] / 255.0f;
        out[2 * plane + i] = canvas[i * 3 + 2] / 255.0f;
    }
    return out;
}

void unscale_boxes(std::vector<Detection>& dets, const LetterboxInfo& info) {
    for (auto& d : dets) {
        d.x1 = (d.x1 - info.pad_w) / info.scale;
        d.y1 = (d.y1 - info.pad_h) / info.scale;
        d.x2 = (d.x2 - info.pad_w) / info.scale;
        d.y2 = (d.y2 - info.pad_h) / info.scale;
    }
}

static uint8_t clamp8(int v) { return (uint8_t)std::clamp(v, 0, 255); }

bool draw_detections(const std::string& out_path, Image& img,
                     const std::vector<Detection>& dets, const std::vector<std::string>& names) {
    const int t = 2;
    for (const auto& d : dets) {
        const int x1 = std::clamp((int)d.x1, 0, img.w - 1), y1 = std::clamp((int)d.y1, 0, img.h - 1);
        const int x2 = std::clamp((int)d.x2, 0, img.w - 1), y2 = std::clamp((int)d.y2, 0, img.h - 1);
        const uint8_t col[3] = {clamp8(d.class_id * 53 + 30), clamp8(220 - d.class_id * 37),
                                clamp8(d.class_id * 91 + 60)};
        for (int y = y1; y <= y2; y++)
            for (int x = x1; x <= x2; x++)
                if (x < x1 + t || x >= x2 - t + 1 || y < y1 + t || y >= y2 - t + 1)
                    for (int c = 0; c < 3; c++) img.rgb[(size_t)(y * img.w + x) * 3 + c] = col[c];
        char label[128];
        const char* cname = d.class_id < (int)names.size() ? names[d.class_id].c_str() : "?";
        snprintf(label, sizeof(label), "%s %.2f", cname, d.score);
        const int ly = std::max(y1 - 10, 8);
        for (const char* p = label; *p; p++) {
            const int ox = x1 + (int)(p - label) * 6;
            // 3x5 pixel font
            static const uint8_t glyphs[64][5] = {
                {0x0E, 0x11, 0x11, 0x11, 0x0E}, {0x04, 0x0C, 0x04, 0x04, 0x0E}, {0x0E, 0x01, 0x0E, 0x10, 0x0F},
                {0x0E, 0x01, 0x06, 0x01, 0x0E}, {0x12, 0x12, 0x1F, 0x02, 0x02}, {0x1F, 0x10, 0x1E, 0x01, 0x1E},
                {0x0E, 0x10, 0x1E, 0x11, 0x0E}, {0x1F, 0x01, 0x02, 0x04, 0x08}, {0x0E, 0x11, 0x0E, 0x11, 0x0E},
                {0x0E, 0x11, 0x0F, 0x01, 0x0E}, {0x1E, 0x11, 0x1E, 0x11, 0x1E}, {0x1E, 0x10, 0x1E, 0x10, 0x1E},
                {0x0E, 0x10, 0x10, 0x10, 0x0E}, {0x1E, 0x11, 0x11, 0x11, 0x1E}, {0x1F, 0x10, 0x1F, 0x10, 0x1F},
                {0x1F, 0x10, 0x1F, 0x10, 0x10}, {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x1F, 0x00, 0x00},
                {0x00, 0x1F, 0x00, 0x1F, 0x00}, {0x0A, 0x04, 0x1F, 0x04, 0x0A}, {0x11, 0x08, 0x04, 0x02, 0x11},
                {0x18, 0x14, 0x12, 0x11, 0x00}, {0x0A, 0x15, 0x17, 0x05, 0x00}, {0x0E, 0x11, 0x1F, 0x10, 0x0E},
                {0x0F, 0x12, 0x0F, 0x12, 0x0F}, {0x00, 0x00, 0x00, 0x00, 0x00}, {0x0C, 0x12, 0x12, 0x12, 0x0C},
                {0x0F, 0x10, 0x10, 0x10, 0x0F}, {0x0F, 0x12, 0x0F, 0x10, 0x0F}, {0x0F, 0x12, 0x0F, 0x12, 0x0F},
                {0x0E, 0x10, 0x1E, 0x11, 0x0F}, {0x1F, 0x01, 0x01, 0x01, 0x01}, {0x0E, 0x11, 0x11, 0x11, 0x0E},
                {0x1F, 0x11, 0x11, 0x11, 0x0F}, {0x1F, 0x11, 0x11, 0x11, 0x1F}, {0x1E, 0x11, 0x1E, 0x10, 0x10},
                {0x1F, 0x01, 0x1F, 0x01, 0x1F}, {0x11, 0x11, 0x1F, 0x11, 0x11}, {0x01, 0x01, 0x1F, 0x01, 0x01},
                {0x11, 0x0A, 0x04, 0x0A, 0x11}, {0x11, 0x11, 0x0F, 0x01, 0x01}, {0x1F, 0x02, 0x04, 0x08, 0x1F},
                {0x0E, 0x08, 0x08, 0x08, 0x0E}, {0x11, 0x0A, 0x04, 0x02, 0x01}, {0x1F, 0x02, 0x02, 0x02, 0x1E},
                {0x00, 0x1F, 0x11, 0x11, 0x11}, {0x00, 0x1E, 0x11, 0x11, 0x1E}, {0x00, 0x0F, 0x10, 0x10, 0x0F},
                {0x00, 0x1E, 0x11, 0x11, 0x1E}, {0x00, 0x1F, 0x10, 0x1F, 0x01}, {0x00, 0x00, 0x00, 0x00, 0x00},
                {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00},
                {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00},
                {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00},
                {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x00, 0x00},
            };
            const int gi = (*p >= ' ' && *p < '`') ? *p - ' ' : 0;
            for (int gy = 0; gy < 5; gy++)
                for (int gx = 0; gx < 5; gx++)
                    if (glyphs[gi][gy] & (1 << (4 - gx))) {
                        const int px = ox + gx, py = ly + gy;
                        if (px < img.w && py < img.h)
                            for (int c = 0; c < 3; c++)
                                img.rgb[(size_t)(py * img.w + px) * 3 + c] = 255 - col[c] / 2;
                    }
        }
    }
    return stbi_write_png(out_path.c_str(), img.w, img.h, 3, img.rgb.data(), img.w * 3) != 0;
}

std::vector<float> restore_depth(const std::vector<float>& depth, int depth_w, int depth_h,
                                 const LetterboxInfo& info, int image_w, int image_h) {
    if ((int)depth.size() != depth_w * depth_h || depth_w <= 0 || depth_h <= 0 || image_w <= 0 || image_h <= 0) {
        return {};
    }
    std::vector<float> canvas = resize_bilinear_float(depth.data(), depth_w, depth_h, info.imgsz_w, info.imgsz_h);
    std::vector<float> crop((size_t)info.new_w * info.new_h);
    for (int y = 0; y < info.new_h; y++) {
        memcpy(crop.data() + (size_t)y * info.new_w,
               canvas.data() + (size_t)(y + info.pad_h) * info.imgsz_w + info.pad_w,
               (size_t)info.new_w * sizeof(float));
    }
    return resize_bilinear_float(crop.data(), info.new_w, info.new_h, image_w, image_h);
}

bool write_depth_png(const std::string& out_path, const std::vector<float>& depth, int width, int height,
                     float max_depth) {
    if ((int)depth.size() != width * height || width <= 0 || height <= 0) return false;
    std::vector<float> valid;
    valid.reserve(depth.size());
    for (float v : depth)
        if (std::isfinite(v) && v > 0.0f) valid.push_back(v);
    if (valid.empty()) return false;
    const float min_depth = *std::min_element(valid.begin(), valid.end());
    if (!(max_depth > min_depth)) {
        const size_t p95 = std::min(valid.size() - 1, valid.size() * 95 / 100);
        std::nth_element(valid.begin(), valid.begin() + p95, valid.end());
        max_depth = valid[p95];
    }
    const float scale = 1.0f / std::max(max_depth - min_depth, 1e-6f);
    std::vector<uint8_t> rgb(depth.size() * 3);
    for (size_t i = 0; i < depth.size(); i++) {
        const float t = std::clamp((depth[i] - min_depth) * scale, 0.0f, 1.0f);
        const float r = std::clamp(1.5f - std::fabs(4.0f * t - 3.0f), 0.0f, 1.0f);
        const float g = std::clamp(1.5f - std::fabs(4.0f * t - 2.0f), 0.0f, 1.0f);
        const float b = std::clamp(1.5f - std::fabs(4.0f * t - 1.0f), 0.0f, 1.0f);
        rgb[i * 3 + 0] = (uint8_t)std::nearbyint(r * 255.0f);
        rgb[i * 3 + 1] = (uint8_t)std::nearbyint(g * 255.0f);
        rgb[i * 3 + 2] = (uint8_t)std::nearbyint(b * 255.0f);
    }
    return stbi_write_png(out_path.c_str(), width, height, 3, rgb.data(), width * 3) != 0;
}

}  // namespace yolo
