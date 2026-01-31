#ifndef STREAM_POOL_H
#define STREAM_POOL_H

#include <cuda_runtime_api.h>
#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include "config.h"
#include "inference_config.h"
#include <atomic>
namespace inference {

class TRTInfer_API StreamHandle {
public:
    StreamHandle(cudaStream_t stream, int id);
    ~StreamHandle();
    
    cudaStream_t get() const { return stream_; }
    int id() const { return id_; }
    
private:
    cudaStream_t stream_;
    int id_;
};

class TRTInfer_API StreamPool {
public:
    explicit StreamPool(int num_streams = DEFAULT_NUM_STREAMS);
    ~StreamPool();
    
    std::shared_ptr<StreamHandle> acquire();
    void release(std::shared_ptr<StreamHandle> stream);
    
    size_t size() const { return streams_.size(); }
    size_t available() const;
    int num_streams() const { return num_streams_; }
    
    void synchronize_all();
    
private:
    std::atomic<bool> b_stop_;
    int num_streams_;
    std::vector<std::shared_ptr<StreamHandle>> streams_;
    std::vector<std::shared_ptr<StreamHandle>> available_streams_;
    std::mutex mutex_;
    std::condition_variable cv_;
    
    void create_streams();
    void destroy_streams();
};

}

#endif
