// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "common.hpp"

#include <vector>

namespace yolo {

struct PostprocConfig {
    float conf_thres = 0.25f;  // predict conf from cfg/default.yaml
    float iou_thres = 0.7f;    // predict iou from cfg/default.yaml
    int max_det = 300;
    int max_nms = 30000;
};

/* Decode the raw detect output into boxes in letterbox-canvas coordinates.
 *
 * raw is [no, na] with element (c, a) at raw[c * na + a]; no = 4*reg_max + nc.
 * anchors is [na*2] holding (x+0.5, y+0.5) per anchor (unscaled), strides is
 * [na]. Mirrors ultralytics Detect._inference + non_max_suppression:
 *   - v8 heads: per-anchor max sigmoid class + conf filter, DFL softmax over
 *     reg_max, dist2bbox(xywh) * stride, xywh->xyxy, greedy class-aware NMS,
 *     top max_det (score descending, ties by anchor index).
 *   - end2end heads (yolo26): dist2bbox(xyxy) * stride, per-anchor max
 *     sigmoid class + conf filter, top max_det by score. No NMS.
 */
std::vector<Detection> postprocess(const std::vector<float>& raw, int no, int na, const ModelMeta& meta,
                                   const float* anchors, const float* strides, const PostprocConfig& cfg);

}  // namespace yolo
