// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "common.hpp"

#include <string>
#include <vector>

namespace yolo {

struct Image {
    int w = 0, h = 0, c = 3;
    std::vector<uint8_t> rgb;  // interleaved RGB8
};

// Load an image (jpg/png/bmp). Returns false on failure.
bool load_image(const std::string& path, Image& img);

// Ultralytics-equivalent LetterBox(auto=True, stride=32): resize keeping aspect
// then pad to a stride-multiple rectangle inside imgsz x imgsz. Bilinear
// resampling matches cv2.INTER_LINEAR bit-for-bit.
void letterbox_image(const Image& img, int imgsz, LetterboxInfo& info, std::vector<float>& out);

// Map boxes from the letterboxed canvas back to original image pixels.
void unscale_boxes(std::vector<Detection>& dets, const LetterboxInfo& info);

// Draw detections onto the image (in place) and write a PNG.
bool draw_detections(const std::string& out_path, Image& img,
                     const std::vector<Detection>& dets, const std::vector<std::string>& names);

// Restore a model-resolution depth map to the original image size, matching
// DepthPredictor's bilinear resize, letterbox crop, and final resize.
std::vector<float> restore_depth(const std::vector<float>& depth, int depth_w, int depth_h,
                                 const LetterboxInfo& info, int image_w, int image_h);

// Write a colorized depth preview. The float depth values remain in meters;
// the PNG is display-only and scaled to max_depth (or the 95th percentile).
bool write_depth_png(const std::string& out_path, const std::vector<float>& depth, int width, int height,
                     float max_depth = 0.0f);

}  // namespace yolo
