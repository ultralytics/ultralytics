// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#include "backend.hpp"
#include "common.hpp"

#include "ggml-cpu.h"

#if defined(YOLO_USE_CUDA)
#include "ggml-cuda.h"
#endif
#if defined(YOLO_USE_METAL)
#include "ggml-metal.h"
#endif
#if defined(YOLO_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <cstdio>
#include <thread>
#include <vector>

namespace yolo {

static void forward_ggml_log(enum ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_DEBUG) return;
    std::fputs(text, stderr);
    std::fflush(stderr);
}

/* Try to create a GPU backend if one was compiled in and a device exists.
 * Returns nullptr (not an error) when no GPU backend is built or no device
 * is present — the caller falls back to CPU-only. */
static ggml_backend_t try_init_gpu_backend() {
#if defined(YOLO_USE_CUDA)
    int n = ggml_backend_cuda_get_device_count();
    if (n > 0) {
        ggml_backend_t b = ggml_backend_cuda_init(0);
        if (b) {
            YOLO_LOG_INFO("GPU backend: CUDA device 0 (%d available)", n);
            return b;
        }
        YOLO_LOG_WARN("ggml_backend_cuda_init(0) failed; using CPU");
    }
    return nullptr;
#elif defined(YOLO_USE_METAL)
    ggml_backend_t b = ggml_backend_metal_init();
    if (b) {
        YOLO_LOG_INFO("GPU backend: Metal");
        return b;
    }
    YOLO_LOG_WARN("ggml_backend_metal_init failed; using CPU");
    return nullptr;
#elif defined(YOLO_USE_VULKAN)
    int n = ggml_backend_vk_get_device_count();
    if (n > 0) {
        ggml_backend_t b = ggml_backend_vk_init(0);
        if (b) {
            YOLO_LOG_INFO("GPU backend: Vulkan device 0 (%d available)", n);
            return b;
        }
        YOLO_LOG_WARN("ggml_backend_vk_init(0) failed; using CPU");
    }
    return nullptr;
#else
    return nullptr;  // CPU-only build
#endif
}

BackendCtx init_backend_ctx(int n_threads) {
    ggml_log_set(forward_ggml_log, nullptr);
    BackendCtx ctx{};
    if (n_threads <= 0) {
        // Hardware default. The backend and the persistent threadpool below
        // MUST agree on the thread count: ggml_graph_plan takes n_threads from
        // the backend while compute indexes threadpool->workers[omp_thread],
        // so any mismatch overruns the workers array.
        n_threads = (int)std::thread::hardware_concurrency();
        if (n_threads <= 0) n_threads = 4;
    }
    ctx.n_threads = n_threads;

    ctx.cpu = ggml_backend_cpu_init();
    if (!ctx.cpu) {
        YOLO_LOG_ERROR("ggml_backend_cpu_init returned null");
        return ctx;
    }
    ggml_backend_cpu_set_n_threads(ctx.cpu, n_threads);

    /* Attach a persistent threadpool to the CPU backend — amortizes the
     * disposable-threadpool setup ggml would otherwise pay per call. */
    {
        ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
        ctx.threadpool = ggml_threadpool_new(&tpp);
        if (ctx.threadpool) {
            ggml_backend_cpu_set_threadpool(ctx.cpu, ctx.threadpool);
        } else {
            YOLO_LOG_WARN("ggml_threadpool_new failed; per-call threadpool");
        }
    }

    /* Try a GPU backend. If present, build a scheduler spanning [gpu, cpu]
     * so ops the GPU can't run fall back to CPU automatically. */
    ctx.gpu = try_init_gpu_backend();
    if (ctx.gpu) {
        std::vector<ggml_backend_t> backends = {ctx.gpu, ctx.cpu};
        std::vector<ggml_backend_buffer_type_t> bufts = {
            ggml_backend_get_default_buffer_type(ctx.gpu),
            ggml_backend_get_default_buffer_type(ctx.cpu),
        };
        ctx.sched = ggml_backend_sched_new(
            backends.data(), bufts.data(), (int)backends.size(),
            /*graph_size*/ 8192, /*parallel*/ false, /*op_offload*/ true);
        if (!ctx.sched) {
            YOLO_LOG_WARN("ggml_backend_sched_new failed; CPU-only");
            ggml_backend_free(ctx.gpu);
            ctx.gpu = nullptr;
        }
    }
    return ctx;
}

ggml_backend_buffer_type_t backend_weight_buft(const BackendCtx& ctx) {
    if (ctx.gpu) {
        return ggml_backend_get_default_buffer_type(ctx.gpu);
    }
    return ggml_backend_get_default_buffer_type(ctx.cpu);
}

void free_backend_ctx(BackendCtx& ctx) {
    if (ctx.galloc) {
        ggml_gallocr_free(ctx.galloc);
        ctx.galloc = nullptr;
    }
    if (ctx.sched) {
        ggml_backend_sched_free(ctx.sched);
        ctx.sched = nullptr;
    }
    if (ctx.gpu) {
        ggml_backend_free(ctx.gpu);
        ctx.gpu = nullptr;
    }
    if (ctx.cpu) {
        ggml_backend_free(ctx.cpu);
        ctx.cpu = nullptr;
    }
    if (ctx.threadpool) {
        ggml_threadpool_free(ctx.threadpool);
        ctx.threadpool = nullptr;
    }
}

bool backend_graph_alloc(BackendCtx& ctx, ggml_cgraph* graph) {
    if (ctx.sched) {
        if (!ggml_backend_sched_alloc_graph(ctx.sched, graph)) {
            YOLO_LOG_ERROR("backend_graph_alloc: sched alloc failed");
            return false;
        }
        return true;
    }
    /* CPU path: persistent gallocr. */
    if (!ctx.galloc) {
        ctx.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.cpu));
        if (!ctx.galloc) {
            YOLO_LOG_ERROR("backend_graph_alloc: gallocr_new failed");
            return false;
        }
    }
    if (!ggml_gallocr_alloc_graph(ctx.galloc, graph)) {
        YOLO_LOG_ERROR("backend_graph_alloc: gallocr_alloc_graph failed");
        return false;
    }
    return true;
}

int backend_graph_compute(BackendCtx& ctx, ggml_cgraph* graph) {
    if (ctx.sched) {
        ggml_status st = ggml_backend_sched_graph_compute(ctx.sched, graph);
        return (int)st;
    }
    ggml_status st = ggml_backend_graph_compute(ctx.cpu, graph);
    return (int)st;
}

const char* backend_name(const BackendCtx& ctx) {
    if (ctx.gpu) return ggml_backend_name(ctx.gpu);
    if (ctx.cpu) return ggml_backend_name(ctx.cpu);
    return "none";
}

}  // namespace yolo
