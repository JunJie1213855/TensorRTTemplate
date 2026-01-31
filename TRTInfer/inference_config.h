#ifndef INFERENCE_CONFIG_H
#define INFERENCE_CONFIG_H

#include <cstdint>

namespace inference
{

    struct InferenceConfig
    {
        int num_streams = 4;
        bool enable_async = true;
        int max_pending_tasks = 100;
        bool use_pinned_memory = true;
        bool enable_profiling = false;
        int device_id = 0;
        bool enable_dynamic_batch = false;
    };

    constexpr int DEFAULT_NUM_STREAMS = 4;
    constexpr int MAX_STREAMS = 16;
    constexpr int MIN_STREAMS = 1;
    constexpr int DEFAULT_MAX_PENDING = 100;
    constexpr int CUDA_MAX_DEVICES = 8;

}

#endif
