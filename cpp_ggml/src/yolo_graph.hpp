// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "backend.hpp"
#include "gguf_loader.hpp"

#include <string>
#include <vector>

namespace yolo {

/* An inference session: parsed GGUF model + built ggml graph on a backend. */
struct Session {
    ModelDef model;
    BackendCtx backend;
    ggml_context* gctx = nullptr;              // graph tensors
    ggml_context* wctx = nullptr;              // weight tensors
    ggml_backend_buffer_t wbuf = nullptr;
    ggml_tensor* input = nullptr;              // external [W, H, 3] input; F16 on Vulkan F16 models
    ggml_tensor* output = nullptr;             // raw detect [A,no] or metric depth [W,H,1,1]
    ggml_cgraph* graph = nullptr;
    std::vector<ggml_fp16_t> input_f16;         // Vulkan F16 upload scratch, allocated once per session
    std::vector<ggml_fp16_t> output_f16;        // F16 readback scratch, allocated once per session
    int input_w = 640;                          // letterboxed canvas dims (stride-multiple,
    int input_h = 640;                          // non-square under LetterBox auto=True)

    // Postprocess constants (mirrors ultralytics make_anchors).
    std::vector<float> anchors;                // [A*2] (x+0.5, y+0.5) per anchor
    std::vector<float> anchor_strides;         // [A]
    int anchor_total = 0;
    std::vector<float> dfl_proj;               // [reg_max]

    // Debug: per-op output tensors, parallel to model.ops (parity testing).
    std::vector<ggml_tensor*> op_values;
};

// Create a session for a GGUF model. `threads` <= 0 means hardware default.
// `input_w`/`input_h` are the letterboxed canvas dims; 0 defaults to the
// square imgsz stored in the GGUF metadata. `keep_all_ops` marks every op
// output as a graph leaf (no gallocr buffer reuse) so --dump-ops reads valid
// data; costs extra memory, debug only.
Session* create_session(const std::string& gguf_path, int threads, int input_w = 0, int input_h = 0,
                        bool keep_all_ops = false);
void free_session(Session* s);

// Copy a CHW float image into the input tensor, run the graph.
bool session_run(Session* s, const float* chw_image);

// Read back the raw output [no, A] (row-major: no rows x A anchors).
bool session_read_output(Session* s, std::vector<float>& out, int& no, int& na);

// Read back a metric depth map in meters, row-major [height, width].
bool session_read_depth(Session* s, std::vector<float>& out, int& width, int& height);

// Dump every per-op output tensor to `dir` as YLYR0001 bins (4x i32 dims + f32).
bool session_dump_ops(const Session* s, const std::string& dir);

}  // namespace yolo
