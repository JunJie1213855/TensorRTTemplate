#include "memory_pool.h"
#include <iostream>
#include <unordered_map>

namespace inference {

MemoryPool::MemoryPool(const std::unordered_map<std::string, size_t>& input_sizes,
                       const std::unordered_map<std::string, size_t>& output_sizes,
                       int num_streams)
    : num_streams_(num_streams),
      input_sizes_(input_sizes),
      output_sizes_(output_sizes),
      total_allocated_(0) {
    
    if (num_streams < MIN_STREAMS) {
        num_streams_ = MIN_STREAMS;
    } else if (num_streams > MAX_STREAMS) {
        num_streams_ = MAX_STREAMS;
    }
    // allocate memory
    allocate_buffers();
}

MemoryPool::~MemoryPool() {
    deallocate_buffers();
}

void MemoryPool::allocate_buffers() {
    for (const auto& [name, size] : input_sizes_) {
        input_buffers_[name].reserve(num_streams_);
        
        for (int i = 0; i < num_streams_; ++i) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate input buffer for " << name 
                          << " stream " << i << ": " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            
            input_buffers_[name].emplace_back(ptr, size, i);
            total_allocated_ += size;
        }
    }
    
    for (const auto& [name, size] : output_sizes_) {
        output_buffers_[name].reserve(num_streams_);
        
        for (int i = 0; i < num_streams_; ++i) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate output buffer for " << name 
                          << " stream " << i << ": " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            
            output_buffers_[name].emplace_back(ptr, size, i);
            total_allocated_ += size;
        }
    }
    
    std::cout << "Memory pool allocated: " << total_allocated_ / (1024.0 * 1024.0) 
              << " MB for " << num_streams_ << " streams" << std::endl;
}

void MemoryPool::deallocate_buffers() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [name, blocks] : input_buffers_) {
        for (auto& block : blocks) {
            if (block.ptr != nullptr) {
                cudaFree(block.ptr);
            }
        }
        blocks.clear();
    }
    input_buffers_.clear();
    
    for (auto& [name, blocks] : output_buffers_) {
        for (auto& block : blocks) {
            if (block.ptr != nullptr) {
                cudaFree(block.ptr);
            }
        }
        blocks.clear();
    }
    output_buffers_.clear();
    
    total_allocated_ = 0;
    std::cout << "Memory pool deallocated" << std::endl;
}

void* MemoryPool::allocate(const std::string& name, size_t size, int stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = input_buffers_.find(name);
    if (it != input_buffers_.end()) {
        auto* block = find_available_block(it->second, size, stream_id);
        if (block) {
            block->in_use = true;
            return block->ptr;
        }
    }
    
    auto it2 = output_buffers_.find(name);
    if (it2 != output_buffers_.end()) {
        auto* block = find_available_block(it2->second, size, stream_id);
        if (block) {
            block->in_use = true;
            return block->ptr;
        }
    }
    
    return nullptr;
}

void MemoryPool::deallocate(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [name, blocks] : input_buffers_) {
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    for (auto& [name, blocks] : output_buffers_) {
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
}

void* MemoryPool::get_input_binding(const std::string& name, int stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = input_buffers_.find(name);
    if (it != input_buffers_.end()) {
        if (stream_id < static_cast<int>(it->second.size())) {
            return it->second[stream_id].ptr;
        }
    }
    
    return nullptr;
}

void* MemoryPool::get_output_binding(const std::string& name, int stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = output_buffers_.find(name);
    if (it != output_buffers_.end()) {
        if (stream_id < static_cast<int>(it->second.size())) {
            return it->second[stream_id].ptr;
        }
    }
    
    return nullptr; 
}

MemoryBlock* MemoryPool::find_available_block(std::vector<MemoryBlock>& blocks, 
                                               size_t size, int stream_id) {
    if (stream_id < static_cast<int>(blocks.size())) {
        if (!blocks[stream_id].in_use && blocks[stream_id].size == size) {
            return &blocks[stream_id];
        }
    }
    
    for (auto& block : blocks) {
        if (!block.in_use && block.size == size) {
            return &block;
        }
    }
    
    return nullptr;
}

void MemoryPool::synchronize_all() {
    cudaDeviceSynchronize();
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [name, blocks] : input_buffers_) {
        for (auto& block : blocks) {
            block.in_use = false;
        }
    }
    
    for (auto& [name, blocks] : output_buffers_) {
        for (auto& block : blocks) {
            block.in_use = false;
        }
    }
}

}
