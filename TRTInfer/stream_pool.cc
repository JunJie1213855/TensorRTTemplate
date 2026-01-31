#include "stream_pool.h"
#include <iostream>

namespace inference
{

    StreamHandle::StreamHandle(cudaStream_t stream, int id)
        : stream_(stream), id_(id)
    {
    }

    StreamHandle::~StreamHandle()
    {
    }

    StreamPool::StreamPool(int num_streams)
        : num_streams_(num_streams), b_stop_(false)
    {
        if (num_streams < MIN_STREAMS)
        {
            num_streams_ = MIN_STREAMS;
        }
        else if (num_streams > MAX_STREAMS)
        {
            num_streams_ = MAX_STREAMS;
        }

        create_streams();
    }

    StreamPool::~StreamPool()
    {
        b_stop_ = true;
        cv_.notify_all();
        destroy_streams();
    }

    void StreamPool::create_streams()
    {
        for (int i = 0; i < num_streams_; ++i)
        {
            cudaStream_t stream;
            cudaError_t err = cudaStreamCreate(&stream);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to create CUDA stream " << i << ": "
                          << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }

            auto handle = std::make_shared<StreamHandle>(stream, i);
            streams_.push_back(handle);
            available_streams_.push_back(handle);
        }

        std::cout << "Created " << num_streams_ << " CUDA streams" << std::endl;
    }

    void StreamPool::destroy_streams()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &handle : streams_)
        {
            if (handle && handle->get() != nullptr)
            {
                cudaStreamDestroy(handle->get());
            }
        }

        streams_.clear();
        available_streams_.clear();

        std::cout << "Destroyed all CUDA streams" << std::endl;
    }

    std::shared_ptr<StreamHandle> StreamPool::acquire()
    {
        std::unique_lock<std::mutex> lock(mutex_);

        cv_.wait(lock, [this]()
                 { return b_stop_ || !available_streams_.empty(); });
        if (b_stop_)
            return nullptr;
        auto stream = available_streams_.back();
        available_streams_.pop_back();

        return stream;
    }

    void StreamPool::release(std::shared_ptr<StreamHandle> stream)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (b_stop_)
            return;
        available_streams_.push_back(stream);
        cv_.notify_one();
    }

    size_t StreamPool::available() const
    {
        std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(mutex_));
        return available_streams_.size();
    }

    void StreamPool::synchronize_all()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto &handle : streams_)
        {
            if (handle && handle->get() != nullptr)
            {
                cudaStreamSynchronize(handle->get());
            }
        }
    }

}
