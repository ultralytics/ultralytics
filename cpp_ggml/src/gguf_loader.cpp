// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "gguf_loader.hpp"

#include <cstring>

namespace yolo {

namespace {

int64_t key_or(const gguf_context* g, const char* key, int64_t def) {
    int64_t id = gguf_find_key(g, key);
    return id >= 0 ? (int64_t)gguf_get_val_u32(g, id) : def;
}

std::string str_or(const gguf_context* g, const char* key, const char* def) {
    int64_t id = gguf_find_key(g, key);
    return id >= 0 ? gguf_get_val_str(g, id) : def;
}

// Read an i32/u32/f32 KV scalar into int64 depending on stored type.
int64_t scalar_kv(const gguf_context* g, int64_t id) {
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_UINT32: return (int64_t)gguf_get_val_u32(g, id);
        case GGUF_TYPE_INT32:  return (int64_t)gguf_get_val_i32(g, id);
        case GGUF_TYPE_FLOAT32: return (int64_t)gguf_get_val_f32(g, id);  // truncated
        default: return 0;
    }
}

}  // namespace

static void parse_meta(const gguf_context* g, ModelMeta& meta) {
    meta.name = str_or(g, "general.name", "yolo");
    meta.task = str_or(g, "yolo.task", "detect");
    meta.dtype = str_or(g, "yolo.dtype", "?");
    meta.nc = (int)key_or(g, "yolo.nc", 80);
    meta.nl = (int)key_or(g, "yolo.nl", 3);
    meta.imgsz = (int)key_or(g, "yolo.imgsz", 640);

    if (int64_t id = gguf_find_key(g, "yolo.strides"); id >= 0) {
        size_t n = gguf_get_arr_n(g, id);
        const uint8_t* p = (const uint8_t*)gguf_get_arr_data(g, id);
        for (size_t i = 0; i < n; i++) meta.strides.push_back(((const float*)p)[i]);
    }
    if (int64_t id = gguf_find_key(g, "yolo.class_names"); id >= 0) {
        size_t n = gguf_get_arr_n(g, id);
        for (size_t i = 0; i < n; i++) meta.class_names.emplace_back(gguf_get_arr_str(g, id, i));
    }
    if (meta.strides.empty()) {
        for (int i = 0; i < meta.nl; i++) meta.strides.push_back(float(8 << i));
    }
}

ModelMeta read_gguf_meta(const std::string& path) {
    gguf_init_params ip{};  // no_alloc: header only, no tensor mapping
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        YOLO_LOG_ERROR("failed to open GGUF: %s", path.c_str());
        return {};
    }
    ModelMeta meta;
    parse_meta(g, meta);
    gguf_free(g);
    return meta;
}

std::unique_ptr<ModelDef> load_gguf(const std::string& path) {
    ggml_context* weight_ctx = nullptr;
    gguf_init_params ip{};
    ip.no_alloc = false;      // map tensor data directly
    ip.ctx = &weight_ctx;

    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) {
        YOLO_LOG_ERROR("failed to open GGUF: %s", path.c_str());
        return nullptr;
    }

    auto model = std::make_unique<ModelDef>();

    // ---- metadata ----
    parse_meta(g, model->meta);
    const int64_t graph_version = key_or(g, "yolo.op_graph_version", 0);
    if (graph_version < 1 || graph_version > 2) {
        YOLO_LOG_ERROR("unsupported yolo.op_graph_version: %lld", (long long)graph_version);
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }

    // ---- op graph ----
    const int64_t n_ops = key_or(g, "yolo.op.count", 0);
    if (n_ops <= 0 || n_ops > 10000) {
        YOLO_LOG_ERROR("invalid yolo.op.count: %lld", (long long)n_ops);
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    model->ops.resize(n_ops);
    for (int64_t i = 0; i < n_ops; i++) {
        OpDef& op = model->ops[i];
        std::string prefix = "op." + std::to_string(i);
        op.type = str_or(g, (prefix + ".type").c_str(), "");
        if (op.type.empty()) {
            YOLO_LOG_ERROR("missing op type at index %lld", (long long)i);
            gguf_free(g);
            ggml_free(weight_ctx);
            return nullptr;
        }

        if (int64_t id = gguf_find_key(g, (prefix + ".inputs").c_str()); id >= 0) {
            size_t n = gguf_get_arr_n(g, id);
            const int32_t* p = (const int32_t*)gguf_get_arr_data(g, id);
            op.inputs.assign(p, p + n);
            for (int input : op.inputs) {
                if (input < -1 || input >= i) {
                    YOLO_LOG_ERROR("invalid input %d for op %lld", input, (long long)i);
                    gguf_free(g);
                    ggml_free(weight_ctx);
                    return nullptr;
                }
            }
        }

        // Scan all keys under this op prefix for params / tensor references.
        const int64_t n_kv = gguf_get_n_kv(g);
        for (int64_t kv = 0; kv < n_kv; kv++) {
            const char* k = gguf_get_key(g, kv);
            if (strncmp(k, prefix.c_str(), prefix.size()) != 0 || k[prefix.size()] != '.') continue;
            const char* field = k + prefix.size() + 1;
            if (!strcmp(field, "type") || !strcmp(field, "inputs")) continue;
            switch (gguf_get_kv_type(g, kv)) {
                case GGUF_TYPE_STRING:
                    op.sparams[field] = gguf_get_val_str(g, kv);
                    break;
                case GGUF_TYPE_ARRAY: {
                    size_t n = gguf_get_arr_n(g, kv);
                    auto& vec = op.aparams[field];
                    vec.resize(n);
                    const uint8_t* p = (const uint8_t*)gguf_get_arr_data(g, kv);
                    for (size_t j = 0; j < n; j++) vec[j] = ((const uint32_t*)p)[j];
                    break;
                }
                case GGUF_TYPE_FLOAT32: {
                    float f = gguf_get_val_f32(g, kv);
                    if (f == (int64_t)f && (field[0] == 'p' || field[0] == 's' || field[0] == 'd'))
                        op.iparams[field] = (int64_t)f;  // pad/stride/dilation stored as float
                    else
                        op.fparams[field] = f;
                    break;
                }
                default:
                    op.iparams[field] = scalar_kv(g, kv);
                    break;
            }
        }
        if (op.type == "detect") {
            model->has_detect = true;
            model->detect_op_index = (int)i;
            model->meta.reg_max = (int)key_or(g, (prefix + ".reg_max").c_str(), 16);
            model->meta.end2end = key_or(g, (prefix + ".end2end").c_str(), 0) != 0;
            model->meta.max_det = (int)key_or(g, (prefix + ".max_det").c_str(), 300);
        }
    }
    if ((model->meta.task == "detect" && !model->has_detect) ||
        (model->meta.task == "depth" && model->ops.back().type != "depth")) {
        YOLO_LOG_ERROR("op graph does not contain the declared %s output", model->meta.task.c_str());
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    if (model->meta.task != "detect" && model->meta.task != "depth") {
        YOLO_LOG_ERROR("unsupported task: %s", model->meta.task.c_str());
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }
    if (model->meta.nl <= 0 || model->meta.strides.size() < (size_t)model->meta.nl) {
        YOLO_LOG_ERROR("invalid feature-level metadata");
        gguf_free(g);
        ggml_free(weight_ctx);
        return nullptr;
    }

    // ---- tensors: reference the mapped ggml context data ----
    const int64_t n_tensors = gguf_get_n_tensors(g);
    for (int64_t t = 0; t < n_tensors; t++) {
        const char* name = gguf_get_tensor_name(g, t);
        ggml_tensor* cur = ggml_get_tensor(weight_ctx, name);
        if (!cur) {
            YOLO_LOG_ERROR("tensor %s missing from ggml context", name);
            gguf_free(g);
            ggml_free(weight_ctx);
            return nullptr;
        }
        HostTensor ht;
        ht.name = name;
        ht.type = cur->type;
        for (int d = 0; d < 4; d++) ht.ne[d] = cur->ne[d];
        ht.data.resize(ggml_nbytes(cur));
        memcpy(ht.data.data(), cur->data, ggml_nbytes(cur));
        model->tensors[name] = std::move(ht);
    }

    YOLO_LOG_INFO("loaded %s: %lld ops, %lld tensors, dtype=%s, nc=%d, end2end=%d",
                  path.c_str(), (long long)n_ops, (long long)n_tensors,
                  model->meta.dtype.c_str(), model->meta.nc, (int)model->meta.end2end);

    gguf_free(g);
    ggml_free(weight_ctx);
    return model;
}

}  // namespace yolo
