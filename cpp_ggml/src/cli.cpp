// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "backend.hpp"
#include "common.hpp"
#include "image_io.hpp"
#include "postprocess.hpp"
#include "yolo_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using SessionPtr = std::unique_ptr<yolo::Session, decltype(&yolo::free_session)>;

void usage() {
    fprintf(stderr,
            "usage:\n"
            "  yolo-cli info   --model M.gguf\n"
            "  yolo-cli detect --model M.gguf --source IMG [--out OUT.png] [--conf 0.25] [--iou 0.7]\n"
            "                 [--max-det 300] [--threads N] [--input-f32 IN.bin] [--dump-raw OUT.bin]\n"
            "                 [--dump-input OUT.bin]\n"
            "  yolo-cli depth  --model M.gguf --source IMG [--out OUT.png] [--raw OUT.bin] [--max-depth M]\n"
            "                 [--threads N] [--dump-input OUT.bin]\n"
            "  yolo-cli bench  --model M.gguf --source IMG [--warmup 20] [--iters 100] [--threads N]\n"
            "\n"
            "raw binary formats (little endian, for pytorch parity tests):\n"
            "  --input-f32 / --dump-input: 8b magic \"YINP0001\", 3x i32 (C,H,W), then f32 CHW pixels\n"
            "  --dump-raw:                  8b magic \"YRAW0001\", 2x i32 (no,na), then f32 [no,na]\n"
            "  depth --raw:                 8b magic \"YDEP0001\", 2x i32 (H,W), then f32 meters\n");
}

using Args = std::unordered_map<std::string, std::string>;

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 2; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || k.size() <= 2) {
            fprintf(stderr, "unexpected argument '%s'\n", argv[i]);
            return {};
        }
        k = k.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
            args[k] = argv[++i];
        } else {
            args[k] = "1";  // boolean flag
        }
    }
    return args;
}

std::string arg_s(const Args& a, const char* k, const std::string& def = "") {
    auto it = a.find(k);
    return it == a.end() ? def : it->second;
}

double arg_f(const Args& a, const char* k, double def) {
    auto it = a.find(k);
    return it == a.end() ? def : atof(it->second.c_str());
}

int arg_i(const Args& a, const char* k, int def) {
    auto it = a.find(k);
    return it == a.end() ? def : atoi(it->second.c_str());
}

// ---- raw f32 dump helpers (tensor-level parity with pytorch) ----------------

bool dump_f32(const char* path, const char* magic, const std::vector<int32_t>& dims, const float* data,
              size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(magic, 1, 8, f);
    fwrite(dims.data(), sizeof(int32_t), dims.size(), f);
    const bool ok = fwrite(data, sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

bool read_f32(const char* path, const char* magic, std::vector<int32_t>& dims, std::vector<float>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    char m[8];
    if (fread(m, 1, 8, f) != 8 || memcmp(m, magic, 8) || dims.empty() ||
        fread(dims.data(), sizeof(int32_t), dims.size(), f) != dims.size()) {
        fclose(f);
        return false;
    }
    size_t n = 1;
    for (int32_t v : dims) {
        if (v <= 0 || n > std::numeric_limits<size_t>::max() / (size_t)v) {
            fclose(f);
            return false;
        }
        n *= (size_t)v;
    }
    if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
        fclose(f);
        return false;
    }
    const long data_pos = ftell(f);
    if (data_pos < 0 || fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    const long file_end = ftell(f);
    if (file_end < data_pos || (uint64_t)n > (uint64_t)(file_end - data_pos) / sizeof(float) ||
        fseek(f, data_pos, SEEK_SET) != 0) {
        fclose(f);
        return false;
    }
    out.resize(n);
    const bool ok = fread(out.data(), sizeof(float), n, f) == n;
    fclose(f);
    return ok;
}

// ---- info --------------------------------------------------------------------

int cmd_info(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    if (model_path.empty()) {
        fprintf(stderr, "--model is required\n");
        return 1;
    }
    auto model = yolo::load_gguf(model_path);
    if (!model) return 1;
    const auto& m = model->meta;

    printf("name       : %s\n", m.name.c_str());
    printf("task       : %s\n", m.task.c_str());
    printf("dtype      : %s\n", m.dtype.c_str());
    printf("imgsz      : %d\n", m.imgsz);
    printf("nc         : %d\n", m.nc);
    printf("layers     : %d (strides:", m.nl);
    for (float s : m.strides) printf(" %g", s);
    printf(")\n");
    printf("reg_max    : %d\n", m.reg_max);
    printf("end2end    : %s\n", m.end2end ? "true" : "false");
    printf("max_det    : %d\n", m.max_det);
    printf("ops        : %zu\n", model->ops.size());
    printf("tensors    : %zu\n", model->tensors.size());

    std::map<std::string, int> hist;
    for (const auto& op : model->ops) hist[op.type]++;
    printf("op types   :");
    for (const auto& kv : hist) printf(" %s=%d", kv.first.c_str(), kv.second);
    printf("\n");

    printf("classes    : %d [", (int)m.class_names.size());
    for (size_t i = 0; i < m.class_names.size() && i < 5; i++) printf("%s,", m.class_names[i].c_str());
    if (m.class_names.size() > 5) printf("...");
    printf("]\n");
    return 0;
}

// ---- detect ------------------------------------------------------------------

int cmd_detect(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    const std::string in_f32 = arg_s(args, "input-f32");
    if (model_path.empty() || (source.empty() && in_f32.empty())) {
        fprintf(stderr, "--model and (--source | --input-f32) are required\n");
        return 1;
    }

    // Preprocess first: the letterboxed canvas decides the graph input shape.
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "detect") {
        fprintf(stderr, "detect requires a detect model, got task=%s\n", meta.task.c_str());
        return 1;
    }

    yolo::Image img;
    yolo::LetterboxInfo info{};
    std::vector<float> input;
    int canvas_w = meta.imgsz, canvas_h = meta.imgsz;
    if (!in_f32.empty()) {
        std::vector<int32_t> in_dims = {3, canvas_h, canvas_w};  // updated from file header
        if (!read_f32(in_f32.c_str(), "YINP0001", in_dims, input)) {
            fprintf(stderr, "failed to read --input-f32 %s\n", in_f32.c_str());
            return 1;
        }
        canvas_h = in_dims[1];
        canvas_w = in_dims[2];
        if (in_dims[0] != 3) {
            fprintf(stderr, "--input-f32 must contain three channels\n");
            return 1;
        }
        info = yolo::LetterboxInfo{1.0f, 0, 0, canvas_w, canvas_h, canvas_w, canvas_h};
    } else {
        if (!yolo::load_image(source, img)) return 1;
        yolo::letterbox_image(img, meta.imgsz, info, input);
        canvas_w = info.imgsz_w;
        canvas_h = info.imgsz_h;
    }

    const std::string dump_ops = arg_s(args, "dump-ops");
    SessionPtr session(yolo::create_session(model_path, arg_i(args, "threads", 0), canvas_w, canvas_h,
                                            !dump_ops.empty()),
                       yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;

    const std::string dump_in = arg_s(args, "dump-input");
    if (!dump_in.empty() &&
        !dump_f32(dump_in.c_str(), "YINP0001", {3, canvas_h, canvas_w}, input.data(), input.size())) {
        fprintf(stderr, "failed to write --dump-input %s\n", dump_in.c_str());
        return 1;
    }

    if (!yolo::session_run(s, input.data())) return 1;

    std::vector<float> raw;
    int no = 0, na = 0;
    if (!yolo::session_read_output(s, raw, no, na)) return 1;

    const std::string dump_raw = arg_s(args, "dump-raw");
    if (!dump_raw.empty() && !dump_f32(dump_raw.c_str(), "YRAW0001", {no, na}, raw.data(), raw.size())) {
        fprintf(stderr, "failed to write --dump-raw %s\n", dump_raw.c_str());
        return 1;
    }

    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }

    yolo::PostprocConfig cfg;
    cfg.conf_thres = (float)arg_f(args, "conf", 0.25);
    cfg.iou_thres = (float)arg_f(args, "iou", 0.7);
    cfg.max_det = arg_i(args, "max-det", s->model.meta.max_det);
    if (!(cfg.conf_thres > 0.0f && cfg.conf_thres < 1.0f) || !(cfg.iou_thres >= 0.0f && cfg.iou_thres <= 1.0f) ||
        cfg.max_det <= 0) {
        fprintf(stderr, "--conf must be in (0,1), --iou in [0,1], and --max-det positive\n");
        return 1;
    }
    std::vector<yolo::Detection> dets =
        yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
    yolo::unscale_boxes(dets, info);

    const auto& names = s->model.meta.class_names;
    printf("%d detection%s (%s, %s, %dx%d, backend=%s)\n", (int)dets.size(), dets.size() == 1 ? "" : "s",
           s->model.meta.name.c_str(), s->model.meta.dtype.c_str(), canvas_w, canvas_h,
           yolo::backend_name(s->backend));
    for (const auto& d : dets) {
        const char* cname = d.class_id < (int)names.size() ? names[d.class_id].c_str() : "?";
        printf("  %-12s %.2f  [%.1f, %.1f, %.1f, %.1f]\n", cname, d.score, d.x1, d.y1, d.x2, d.y2);
    }

    const std::string out = arg_s(args, "out");
    if (!out.empty()) {
        if (in_f32.empty()) {
            yolo::draw_detections(out, img, dets, names);
        } else {
            fprintf(stderr, "note: --out skipped with --input-f32 (no source image)\n");
        }
    }

    return 0;
}

// ---- depth -------------------------------------------------------------------

int cmd_depth(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }

    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "depth") {
        fprintf(stderr, "depth requires a depth model, got task=%s\n", meta.task.c_str());
        return 1;
    }

    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    yolo::letterbox_image(img, meta.imgsz, info, input);
    const std::string dump_ops = arg_s(args, "dump-ops");
    SessionPtr session(yolo::create_session(model_path, arg_i(args, "threads", 0), info.imgsz_w, info.imgsz_h,
                                            !dump_ops.empty()),
                       yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;

    if (!yolo::session_run(s, input.data())) {
        return 1;
    }
    std::vector<float> raw;
    int depth_w = 0, depth_h = 0;
    if (!yolo::session_read_depth(s, raw, depth_w, depth_h)) {
        return 1;
    }
    if (!dump_ops.empty() && !yolo::session_dump_ops(s, dump_ops)) {
        fprintf(stderr, "failed to write --dump-ops %s\n", dump_ops.c_str());
        return 1;
    }
    std::vector<float> depth = yolo::restore_depth(raw, depth_w, depth_h, info, img.w, img.h);
    if (depth.empty()) return 1;

    const std::string dump_in = arg_s(args, "dump-input");
    if (!dump_in.empty() &&
        !dump_f32(dump_in.c_str(), "YINP0001", {3, info.imgsz_h, info.imgsz_w}, input.data(), input.size())) {
        fprintf(stderr, "failed to write --dump-input %s\n", dump_in.c_str());
        return 1;
    }
    const std::string raw_path = arg_s(args, "raw");
    if (!raw_path.empty() &&
        !dump_f32(raw_path.c_str(), "YDEP0001", {img.h, img.w}, depth.data(), depth.size())) {
        fprintf(stderr, "failed to write --raw %s\n", raw_path.c_str());
        return 1;
    }
    const std::string out = arg_s(args, "out");
    if (!out.empty() && !yolo::write_depth_png(out, depth, img.w, img.h, (float)arg_f(args, "max-depth", 0.0))) {
        fprintf(stderr, "failed to write --out %s\n", out.c_str());
        return 1;
    }

    const auto [lo, hi] = std::minmax_element(depth.begin(), depth.end());
    const double mean = std::accumulate(depth.begin(), depth.end(), 0.0) / depth.size();
    printf("depth %dx%d meters (min=%.3f mean=%.3f max=%.3f, model=%s, dtype=%s)\n", img.w, img.h, *lo, mean,
           *hi, meta.name.c_str(), meta.dtype.c_str());
    return 0;
}

// ---- bench -------------------------------------------------------------------

struct Stats {
    std::vector<double> ms;
    double mean = 0, min = 0, p50 = 0, p90 = 0, max = 0;
    void finish() {
        std::sort(ms.begin(), ms.end());
        const int n = (int)ms.size();
        if (!n) return;
        double sum = 0;
        for (double v : ms) sum += v;
        mean = sum / n;
        min = ms.front();
        p50 = ms[n / 2];
        p90 = ms[std::min(n - 1, (int)(n * 0.9))];
        max = ms.back();
    }
};

int cmd_bench(const Args& args) {
    const std::string model_path = arg_s(args, "model");
    const std::string source = arg_s(args, "source");
    if (model_path.empty() || source.empty()) {
        fprintf(stderr, "--model and --source are required\n");
        return 1;
    }
    const int warmup = std::max(1, arg_i(args, "warmup", 20));
    const int iters = std::max(1, arg_i(args, "iters", 100));

    // Preprocess first: the letterboxed canvas decides the graph input shape.
    const yolo::ModelMeta meta = yolo::read_gguf_meta(model_path);
    if (meta.imgsz <= 0) return 1;
    if (meta.task != "detect" && meta.task != "depth") {
        fprintf(stderr, "bench supports detect and depth models, got task=%s\n", meta.task.c_str());
        return 1;
    }
    yolo::Image img;
    yolo::LetterboxInfo info{};
    if (!yolo::load_image(source, img)) return 1;
    std::vector<float> input;
    yolo::letterbox_image(img, meta.imgsz, info, input);

    SessionPtr session(yolo::create_session(model_path, arg_i(args, "threads", 0), info.imgsz_w, info.imgsz_h),
                       yolo::free_session);
    yolo::Session* s = session.get();
    if (!s) return 1;

    std::vector<float> raw;
    int no = 0, na = 0;
    yolo::PostprocConfig cfg;
    cfg.max_det = s->model.meta.max_det;
    int depth_w = 0, depth_h = 0;
    std::vector<float> depth;
    Stats preprocess, graph, post, e2e;

    for (int i = 0; i < warmup; i++) {
        yolo::letterbox_image(img, meta.imgsz, info, input);
        if (!yolo::session_run(s, input.data())) return 1;
        if (meta.task == "detect") {
            if (!yolo::session_read_output(s, raw, no, na)) return 1;
        } else if (!yolo::session_read_depth(s, raw, depth_w, depth_h)) {
            return 1;
        }
    }
    for (int i = 0; i < iters; i++) {
        yolo::Clock ce;
        yolo::Clock c0;
        yolo::letterbox_image(img, meta.imgsz, info, input);
        preprocess.ms.push_back(c0.ms_since());
        yolo::Clock c1;
        if (!yolo::session_run(s, input.data())) return 1;
        if (meta.task == "detect") {
            if (!yolo::session_read_output(s, raw, no, na)) return 1;
        } else if (!yolo::session_read_depth(s, raw, depth_w, depth_h)) {
            return 1;
        }
        graph.ms.push_back(c1.ms_since());
        yolo::Clock c2;
        std::vector<yolo::Detection> dets;
        if (meta.task == "detect") {
            dets = yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(), s->anchor_strides.data(), cfg);
            yolo::unscale_boxes(dets, info);
        } else {
            depth = yolo::restore_depth(raw, depth_w, depth_h, info, img.w, img.h);
        }
        post.ms.push_back(c2.ms_since());
        e2e.ms.push_back(ce.ms_since());
        if (i == iters - 1) {
            if (meta.task == "detect") {
                YOLO_LOG_INFO("sanity: %d detections, top score %.3f", (int)dets.size(),
                              dets.empty() ? 0.0f : dets.front().score);
            } else {
                const auto [lo, hi] = std::minmax_element(depth.begin(), depth.end());
                YOLO_LOG_INFO("sanity: depth range %.3f..%.3f meters", *lo, *hi);
            }
        }
    }
    preprocess.finish();
    graph.finish();
    post.finish();
    e2e.finish();

    printf("{\"backend\":\"%s\",\"model\":\"%s\",\"task\":\"%s\",\"dtype\":\"%s\",\"imgsz\":[%d,%d],\"threads\":%d,"
           "\"warmup\":%d,\"iters\":%d,"
           "\"preprocess_ms\":{\"mean\":%.3f,\"p50\":%.3f,\"p90\":%.3f},"
           "\"graph_ms\":{\"mean\":%.3f,\"min\":%.3f,\"p50\":%.3f,\"p90\":%.3f,\"max\":%.3f},"
           "\"post_ms\":{\"mean\":%.3f,\"p50\":%.3f},"
           "\"e2e_ms\":{\"mean\":%.3f,\"min\":%.3f,\"p50\":%.3f,\"p90\":%.3f,\"max\":%.3f}}\n",
           yolo::backend_name(s->backend), s->model.meta.name.c_str(), s->model.meta.task.c_str(),
           s->model.meta.dtype.c_str(), s->input_w, s->input_h, s->backend.n_threads, warmup, iters, preprocess.mean,
           preprocess.p50, preprocess.p90, graph.mean, graph.min, graph.p50, graph.p90, graph.max, post.mean, post.p50,
           e2e.mean, e2e.min, e2e.p50, e2e.p90, e2e.max);

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 1;
    }
    const std::string cmd = argv[1];
    const Args args = parse_args(argc, argv);
    if (args.empty() && argc > 2) return 1;
    if (cmd == "info") return cmd_info(args);
    if (cmd == "detect") return cmd_detect(args);
    if (cmd == "depth") return cmd_depth(args);
    if (cmd == "bench") return cmd_bench(args);
    usage();
    return 1;
}
