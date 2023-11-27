#pragma once

#include <memory>

namespace trt {

    struct ModelConfig {
        std::string weights_path;
        int max_batch_size; 
        int input_h;     /* cfg.INPUT.SIZE_TRAIN[0] */
        int input_w;     /* cfg.INPUT.SIZE_TRAIN[1] */
        int output_size; /* final embedding dims. Could be cfg.MODEL.BACKBONE.FEAT_DIM or cfg.MODEL.HEADS.EMBEDDING_DIM(if you modified. default=0) */
        int device_id;   /* cuda device id(0, 1, 2, ...) */   
    };

    struct EngineConfig : ModelConfig {
        std::string input_name;
        std::string output_name; 
        std::shared_ptr<char> trtModelStream;
        int stream_size;
    };

}

namespace fastrt {

#define FASTBACKBONE_TABLE \
        X(r50, "r50") \
        X(r50_distill, "r50_distill") \
        X(r34, "r34") \
        X(r34_distill, "r34_distill") \
        X(r18_distill, "r18_distill") 

#define X(a, b) a,
        enum FastreidBackboneType { FASTBACKBONE_TABLE };
#undef X

#define FASTHEAD_TABLE \
        X(EmbeddingHead, "EmbeddingHead")

#define X(a, b) a,
    enum FastreidHeadType { FASTHEAD_TABLE };
#undef X

#define FASTPOOLING_TABLE \
        X(maxpool, "maxpool") \
        X(avgpool, "avgpool") \
        X(gempool, "gempool") \
        X(gempoolP, "gempoolP") 

#define X(a, b) a,
    enum FastreidPoolingType { FASTPOOLING_TABLE };
#undef X

    struct FastreidConfig {
        FastreidBackboneType backbone; /* cfg.MODEL.BACKBONE.DEPTH and cfg.MODEL.META_ARCHITECTURE */
        FastreidHeadType head;         /* cfg.MODEL.HEADS.NAME */
        FastreidPoolingType pooling;   /* cfg.MODEL.HEADS.POOL_LAYER */
        int last_stride;               /* cfg.MODEL.BACKBONE.LAST_STRIDE */
        bool with_ibna;                /* cfg.MODEL.BACKBONE.WITH_IBN */
        bool with_nl;                  /* cfg.MODEL.BACKBONE.WITH_NL */
        int embedding_dim;             /* cfg.MODEL.HEADS.EMBEDDING_DIM (Default = 0) */ 
    };

}