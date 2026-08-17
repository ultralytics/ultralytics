// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace yolo {

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Exclusive-boundary IoU, identical to torchvision.ops.nms / box_iou.
inline float box_iou(const Detection& a, const Detection& b) {
    const float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    const float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    const float w = std::max(0.0f, xx2 - xx1), h = std::max(0.0f, yy2 - yy1);
    const float inter = w * h;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float u = area_a + area_b - inter;
    return u > 0.0f ? inter / u : 0.0f;
}

struct Cand {
    int a;
    float score;
    int cls;
};

}  // namespace

std::vector<Detection> postprocess(const std::vector<float>& raw, int no, int na, const ModelMeta& meta,
                                   const float* anchors, const float* strides, const PostprocConfig& cfg) {
    const int nc = meta.nc;
    const int rm = meta.reg_max;
    const int box_ch = 4 * rm;
    if (no != box_ch + nc) {
        YOLO_LOG_ERROR("output channels %d != 4*reg_max + nc (%d)", no, box_ch + nc);
        return {};
    }

    // Per-anchor best class, confidence filtered (shared by both heads). sigmoid is
    // monotonic, so argmax and the conf filter run on the raw logits — sigmoid itself
    // is applied only to the few surviving candidates. The class loop is outer so the
    // anchor walk streams raw[c] contiguously instead of hopping na*4B per class.
    const float logit_thr = -std::log(1.0f / cfg.conf_thres - 1.0f);
    const float* cls_base = raw.data() + (size_t)box_ch * na;
    std::vector<Cand> cands;
    cands.reserve(na / 8);
    if (nc > 1) {
        std::vector<float> best(na, -INFINITY);
        std::vector<int> bc(na, 0);
        for (int c = 0; c < nc; c++) {
            const float* row = cls_base + (size_t)c * na;
            for (int a = 0; a < na; a++) {
                if (row[a] > best[a]) {
                    best[a] = row[a];
                    bc[a] = c;
                }
            }
        }
        for (int a = 0; a < na; a++) {
            if (best[a] > logit_thr) cands.push_back({a, sigmoid(best[a]), bc[a]});
        }
    } else {
        for (int a = 0; a < na; a++) {
            if (cls_base[a] > logit_thr) cands.push_back({a, sigmoid(cls_base[a]), 0});
        }
    }

    // Cap the NMS input by confidence like ultralytics max_nms.
    if ((int)cands.size() > cfg.max_nms) {
        std::partial_sort(cands.begin(), cands.begin() + cfg.max_nms, cands.end(),
                          [](const Cand& x, const Cand& y) { return x.score > y.score; });
        cands.resize(cfg.max_nms);
    }

    // DFL softmax scratch (reg_max may be any size, e.g. 16 for v8, 1 for yolo26).
    std::vector<float> probs(rm > 0 ? rm : 1);

    std::vector<Detection> dets;
    dets.reserve(cands.size());
    for (const Cand& cd : cands) {
        const int a = cd.a;
        const float ax = anchors[2 * a], ay = anchors[2 * a + 1], st = strides[a];
        float d[4];
        if (rm > 1) {
            // torch DFL: x.view(B, 4, reg_max, A) — channel k = j*rm + t (edge j outer, bin t inner).
            for (int j = 0; j < 4; j++) {
                float m = -INFINITY;
                for (int t = 0; t < rm; t++) m = std::max(m, raw[(size_t)(j * rm + t) * na + a]);
                float sum = 0.0f, val = 0.0f;
                for (int t = 0; t < rm; t++) {
                    probs[t] = std::exp(raw[(size_t)(j * rm + t) * na + a] - m);
                    sum += probs[t];
                }
                for (int t = 0; t < rm; t++) val += probs[t] * t;
                d[j] = val / sum;
            }
        } else {
            for (int j = 0; j < 4; j++) d[j] = raw[(size_t)j * na + a];
        }

        if (meta.end2end) {
            // dist2bbox xyxy: anchor -/+ dist, then scaled by stride.
            dets.push_back({(ax - d[0]) * st, (ay - d[1]) * st, (ax + d[2]) * st, (ay + d[3]) * st, cd.score, cd.cls});
        } else {
            // dist2bbox xywh: center = (lt + rb) / 2, wh = lt + rb, scaled by stride.
            const float x1 = ax - d[0], y1 = ay - d[1], x2 = ax + d[2], y2 = ay + d[3];
            const float cx = (x1 + x2) * 0.5f * st, cy = (y1 + y2) * 0.5f * st;
            const float w = (x2 - x1) * st, h = (y2 - y1) * st;
            dets.push_back({cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f, cd.score, cd.cls});
        }
    }

    if (meta.end2end) {
        // Detect.postprocess topk (score descending) then the conf filter in the
        // non_max_suppression end2end branch keeps the order; cap at max_det.
        std::stable_sort(dets.begin(), dets.end(),
                         [](const Detection& x, const Detection& y) { return x.score > y.score; });
        if ((int)dets.size() > cfg.max_det) dets.resize(cfg.max_det);
        return dets;
    }

    // Greedy class-aware NMS, matching torchvision.ops.nms with the class offset
    // trick: only same-class boxes can suppress each other, IoU > threshold.
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int x, int y) { return dets[x].score > dets[y].score; });
    std::vector<int> keep;
    keep.reserve(std::min((size_t)cfg.max_det, dets.size()));
    for (int i : order) {
        bool suppressed = false;
        for (int k : keep) {
            if (dets[k].class_id == dets[i].class_id && box_iou(dets[k], dets[i]) > cfg.iou_thres) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keep.push_back(i);
        if ((int)keep.size() == cfg.max_det) break;
    }

    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int k : keep) out.push_back(dets[k]);
    return out;
}

}  // namespace yolo
