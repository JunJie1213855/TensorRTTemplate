#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include <mutex>
#include <map>
#include <unordered_map>
#include <string>
#include "config.h"
#include "inference_config.h"

namespace inference
{

    struct MemoryBlock
    {
        void *ptr;
        size_t size;
        int stream_id;
        bool in_use;

        MemoryBlock(void *p, size_t s, int sid)
            : ptr(p), size(s), stream_id(sid), in_use(false) {}
    };

    class TRTInfer_API MemoryPool
    {
    public:
        explicit MemoryPool(const std::unordered_map<std::string, size_t> &input_sizes,
                            const std::unordered_map<std::string, size_t> &output_sizes,
                            int num_streams = DEFAULT_NUM_STREAMS);
        ~MemoryPool();

        void *allocate(const std::string &name, size_t size, int stream_id);
        void deallocate(void *ptr);

        void *get_input_binding(const std::string &name, int stream_id);
        void *get_output_binding(const std::string &name, int stream_id);

        void synchronize_all();
        void reset();

        size_t total_allocated() const { return total_allocated_; }

    private:
        int num_streams_;
        std::unordered_map<std::string, size_t> input_sizes_;
        std::unordered_map<std::string, size_t> output_sizes_;

        std::map<std::string, std::vector<MemoryBlock>> input_buffers_;
        std::map<std::string, std::vector<MemoryBlock>> output_buffers_;

        std::mutex mutex_;
        size_t total_allocated_;

        void allocate_buffers();
        void deallocate_buffers();

        MemoryBlock *find_available_block(std::vector<MemoryBlock> &blocks, size_t size, int stream_id);
    };

}

#endif
