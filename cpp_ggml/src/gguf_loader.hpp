// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#pragma once

#include "common.hpp"

#include "ggml.h"
#include "gguf.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace yolo {

// One node of the op-graph vocabulary written by scripts/convert_yolo_to_gguf.py.
struct OpDef {
    std::string type;  // conv|dwconv|maxpool|concat|upsample|interpolate|conv_transpose|add|slice|psa_attention|detect|depth
    std::vector<int> inputs;        // op indices; -1 = graph input image
    std::map<std::string, int64_t> iparams;    // ints
    std::map<std::string, double> fparams;     // floats
    std::map<std::string, std::vector<int64_t>> aparams;  // int arrays (s/p/d)
    std::map<std::string, std::string> sparams; // strings (act)
    std::vector<std::string> tensor_names;      // w, b, qkv_w, ...

    int64_t ip(const std::string& k, int64_t def = 0) const {
        auto it = iparams.find(k);
        return it == iparams.end() ? def : it->second;
    }
    int64_t ai(const std::string& k, int idx, int64_t def = 0) const {
        auto it = aparams.find(k);
        if (it == aparams.end() || idx >= (int)it->second.size()) return def;
        return it->second[idx];
    }
};

// Host-side weight: ggml_type + raw block data + logical shape (torch order).
struct HostTensor {
    std::vector<uint8_t> data;
    ggml_type type = GGML_TYPE_F32;
    int64_t ne[4] = {1, 1, 1, 1};  // ggml order: ne[0] fastest
    std::string name;
};

struct ModelDef {
    ModelMeta meta;
    std::vector<OpDef> ops;
    std::map<std::string, HostTensor> tensors;

    // Flattened per-level head info, taken from the single detect op.
    bool has_detect = false;
    int detect_op_index = -1;
};

// Read only the metadata header (cheap: no tensor data).
ModelMeta read_gguf_meta(const std::string& path);

// Load and parse a GGUF file. Returns nullptr and logs on failure.
std::unique_ptr<ModelDef> load_gguf(const std::string& path);

}  // namespace yolo
