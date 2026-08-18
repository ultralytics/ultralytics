// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "image_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#if defined(YOLO_USE_OPENMP)
#include <omp.h>
#endif

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

static std::vector<float> resize_bilinear_float(const float* src, int sw, int sh, int dw, int dh) {
    std::vector<float> dst((size_t)dw * dh);
    const float fx = (float)sw / dw;
    const float fy = (float)sh / dh;
    std::vector<int> x0(dw), x1(dw), y0(dh), y1(dh);
    std::vector<float> wx(dw), wy(dh);
    for (int x = 0; x < dw; x++) {
        const float sx = (x + 0.5f) * fx - 0.5f;
        const int ix = (int)std::floor(sx);
        x0[x] = std::clamp(ix, 0, sw - 1);
        x1[x] = std::clamp(ix + 1, 0, sw - 1);
        wx[x] = sx - ix;
    }
    for (int y = 0; y < dh; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int iy = (int)std::floor(sy);
        y0[y] = std::clamp(iy, 0, sh - 1);
        y1[y] = std::clamp(iy + 1, 0, sh - 1);
        wy[y] = sy - iy;
    }
#if defined(YOLO_USE_OPENMP)
    const int resize_threads = std::min(8, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(resize_threads) if (dh >= 64)
#endif
    for (int y = 0; y < dh; y++) {
        const int yc0 = y0[y], yc1 = y1[y];
        const float wyv = wy[y];
        for (int x = 0; x < dw; x++) {
            const float wxv = wx[x];
            const float v0 = src[(size_t)yc0 * sw + x0[x]] * (1.0f - wxv) +
                             src[(size_t)yc0 * sw + x1[x]] * wxv;
            const float v1 = src[(size_t)yc1 * sw + x0[x]] * (1.0f - wxv) +
                             src[(size_t)yc1 * sw + x1[x]] * wxv;
            dst[(size_t)y * dw + x] = v0 * (1.0f - wyv) + v1 * wyv;
        }
    }
    return dst;
}

void letterbox_image(const Image& img, int imgsz, LetterboxInfo& info, std::vector<float>& out) {
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

    const size_t plane = (size_t)canvas_w * canvas_h;
    out.resize(3 * plane);
    if (left || right || top || bottom) {
        constexpr float pad = 114.0f / 255.0f;
        for (int c = 0; c < 3; c++) {
            float* channel = out.data() + (size_t)c * plane;
            std::fill(channel, channel + (size_t)top * canvas_w, pad);
            std::fill(channel + (size_t)(top + new_h) * canvas_w, channel + plane, pad);
            for (int y = top; y < top + new_h; y++) {
                float* row = channel + (size_t)y * canvas_w;
                std::fill(row, row + left, pad);
                std::fill(row + left + new_w, row + canvas_w, pad);
            }
        }
    }

    const float fx = (float)img.w / new_w;
    const float fy = (float)img.h / new_h;
    std::vector<int> x0(new_w), x1(new_w);
    std::vector<float> wx(new_w);
    for (int x = 0; x < new_w; x++) {
        const float sx = (x + 0.5f) * fx - 0.5f;
        const int ix0 = (int)std::floor(sx);
        x0[x] = std::clamp(ix0, 0, img.w - 1);
        x1[x] = std::clamp(ix0 + 1, 0, img.w - 1);
        wx[x] = sx - ix0;
    }

#if defined(YOLO_USE_OPENMP)
    const int resize_threads = std::min(8, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(resize_threads) if (new_h >= 64)
#endif
    for (int y = 0; y < new_h; y++) {
        const float sy = (y + 0.5f) * fy - 0.5f;
        const int iy0 = (int)std::floor(sy);
        const int yc0 = std::clamp(iy0, 0, img.h - 1);
        const int yc1 = std::clamp(iy0 + 1, 0, img.h - 1);
        const float wy = sy - iy0;
        for (int x = 0; x < new_w; x++) {
            const size_t p00 = ((size_t)yc0 * img.w + x0[x]) * 3;
            const size_t p01 = ((size_t)yc0 * img.w + x1[x]) * 3;
            const size_t p10 = ((size_t)yc1 * img.w + x0[x]) * 3;
            const size_t p11 = ((size_t)yc1 * img.w + x1[x]) * 3;
            const size_t dst = (size_t)(y + top) * canvas_w + x + left;
            for (int c = 0; c < 3; c++) {
                const float v0 = img.rgb[p00 + c] + (img.rgb[p01 + c] - img.rgb[p00 + c]) * wx[x];
                const float v1 = img.rgb[p10 + c] + (img.rgb[p11 + c] - img.rgb[p10 + c]) * wx[x];
                const uint8_t value = (uint8_t)(v0 + (v1 - v0) * wy + 0.5f);
                out[(size_t)c * plane + dst] = value / 255.0f;
            }
        }
    }
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

static const uint8_t* glyph_rows(char ch) {
    static constexpr uint8_t glyphs[][7] = {
        {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E},  // 0
        {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E},  // 1
        {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F},  // 2
        {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E},  // 3
        {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02},  // 4
        {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E},  // 5
        {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E},  // 6
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08},  // 7
        {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E},  // 8
        {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E},  // 9
        {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},  // A
        {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E},  // B
        {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E},  // C
        {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E},  // D
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F},  // E
        {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10},  // F
        {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F},  // G
        {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11},  // H
        {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E},  // I
        {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C},  // J
        {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11},  // K
        {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F},  // L
        {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11},  // M
        {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11},  // N
        {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},  // O
        {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10},  // P
        {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D},  // Q
        {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11},  // R
        {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E},  // S
        {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04},  // T
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E},  // U
        {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04},  // V
        {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A},  // W
        {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11},  // X
        {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04},  // Y
        {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F},  // Z
    };
    static constexpr uint8_t blank[7] = {};
    static constexpr uint8_t dot[7] = {0, 0, 0, 0, 0, 0x0C, 0x0C};
    static constexpr uint8_t dash[7] = {0, 0, 0, 0x0E, 0, 0, 0};
    static constexpr uint8_t underscore[7] = {0, 0, 0, 0, 0, 0, 0x1F};
    static constexpr uint8_t question[7] = {0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04};

    unsigned char c = (unsigned char)ch;
    if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
    if (c >= '0' && c <= '9') return glyphs[c - '0'];
    if (c >= 'A' && c <= 'Z') return glyphs[10 + c - 'A'];
    if (c == '.') return dot;
    if (c == '-') return dash;
    if (c == '_') return underscore;
    if (c == ' ') return blank;
    return question;
}

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
        const char* cname = d.class_id >= 0 && d.class_id < (int)names.size() ? names[d.class_id].c_str() : "?";
        snprintf(label, sizeof(label), "%s %.2f", cname, d.score);
        constexpr int scale = 2, glyph_h = 7 * scale, advance = 6 * scale, pad = 2;
        constexpr int label_h = 2 * pad + glyph_h;
        const int available = img.w - x1 - 2 * pad;
        const size_t visible_chars = std::min(std::strlen(label), (size_t)std::max(0, (available + scale) / advance));
        if (visible_chars == 0 || img.h < label_h) continue;
        const int label_w = 2 * pad + (int)visible_chars * advance - scale;
        const int ly = y1 >= label_h ? y1 - label_h : std::min(y1 + t, img.h - label_h);
        for (int y = ly; y < ly + label_h; y++)
            for (int x = x1; x < x1 + label_w; x++)
                for (int c = 0; c < 3; c++) img.rgb[(size_t)(y * img.w + x) * 3 + c] = col[c];
        for (size_t i = 0; i < visible_chars; i++) {
            const uint8_t* glyph = glyph_rows(label[i]);
            const int ox = x1 + pad + (int)i * advance, oy = ly + pad;
            for (int gy = 0; gy < 7; gy++)
                for (int gx = 0; gx < 5; gx++)
                    if (glyph[gy] & (1 << (4 - gx)))
                        for (int sy = 0; sy < scale; sy++)
                            for (int sx = 0; sx < scale; sx++)
                                for (int c = 0; c < 3; c++)
                                    img.rgb[(size_t)((oy + gy * scale + sy) * img.w + ox + gx * scale + sx) * 3 + c] =
                                        255;
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
