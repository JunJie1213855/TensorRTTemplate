#ifndef STREAMPOOL_H
#define STREAMPOOL_H

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <memory>
#include <NvInfer.h>
#include <queue>
/**
 * @brief Stream 与 Context 配对结构
 * @details 用于同时管理 Stream 和对应的 ExecutionContext
 */
struct StreamContextPair
{
    cudaStream_t stream;                                                   // CUDA 流
    nvinfer1::IExecutionContext *context;                                  // 执行上下文（裸指针）
    std::unordered_map<std::string, void *> inputBindings, outputBindings; // 输入输出的CUDA内存
    std::unordered_map<std::string, std::shared_ptr<char[]>> outputBlobs;  // 输出的主机内存

    // 可以移动
    StreamContextPair(StreamContextPair&&) = default;
    StreamContextPair& operator=(StreamContextPair&&) = default;


    StreamContextPair() : stream(nullptr), context(nullptr) {}
    /* @brief 重载 ()
     *  @example
     *      pair = pool.acquire();
     *      if(!pair) return false;
     */
    bool operator!() const
    {
        return stream == nullptr || context == nullptr;
    }


};

/**
 * @brief CUDA Stream 池 - 线程安全的 Stream + Context 管理器
 * @details 采用归还策略，支持多线程并行获取/归还 Stream + Context 配对
 *
 * 使用示例:
 *   // 初始化（传入 Engine）
 *   auto pool = StreamPool::GetInstance(engine, 4);
 *
 *   // 获取配对
 *   auto pair = pool->acquire();
 *
 *   // 使用推理
 *   pair.context->setInputTensorAddress("input", gpu_ptr);
 *   pair.context->enqueueV3(pair.stream);
 *
 *   // 归还
 *   pool->release(pair);
 */
class StreamPool
{
public:
    // 禁止拷贝和移动
    StreamPool(const StreamPool &) = delete;
    StreamPool(StreamPool &&) = default;
    StreamPool &operator=(const StreamPool &) = delete;
    StreamPool &operator=(StreamPool &&) = default;

    /**
     * @brief 获取 StreamPool 单例实例（延迟初始化）
     * @param engine TensorRT Engine（首次调用时传入）
     * @param num_streams Stream 池大小（默认 4，仅首次有效）
     * @return StreamPool 智能指针
     */
    StreamPool(nvinfer1::ICudaEngine *engine = nullptr, int num_streams = 4);

    /**
     * @brief 初始化 StreamPool（需要在获取前调用）
     * @param engine TensorRT Engine
     * @param num_streams Stream 数量
     */
    void init(nvinfer1::ICudaEngine *engine, int num_streams);

    /**
     * @brief 获取一个可用的 Stream + Context 配对（阻塞直到有可用）
     * @return StreamContextPair 可用的配对
     * @note 如果所有配对都在使用中，会阻塞等待
     */
    StreamContextPair acquire();

    /**
     * @brief 归还 Stream + Context 配对（归还策略核心）
     * @param pair 要归还的配对
     */
    void release(StreamContextPair&& pair);

    /**
     * @brief 非阻塞尝试获取配对
     * @return StreamContextPair 可用的配对，如果没有则 stream 为 nullptr
     */
    StreamContextPair tryAcquire();

    /**
     * @brief 等待所有配对变为可用
     */
    void syncAll();

    /**
     * @brief 关闭 Stream 池，唤醒所有等待线程
     */
    void shutdown();

    /**
     * @brief 获取当前可用配对数量
     */
    int availableCount() const;

    /**
     * @brief 获取总配对数量
     */
    int size() const;

    /**
     * @brief 检查是否正在关闭
     */
    bool isShuttingDown() const;

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const;

    /**
     * @brief 析构函数 - 销毁所有资源
     */
    ~StreamPool();

protected:
    /**
     * @brief 构造函数
     */
    StreamPool();

private:
    // 检查是否有可用的配对
    bool hasAvailable() const;

    // 检查是否所有配对都不在使用中
    bool hasAvailableInUse() const;

    // 创建 Context
    nvinfer1::IExecutionContext *createContext(nvinfer1::ICudaEngine *engine);

    std::queue<StreamContextPair> pool_; // Stream + Context 配对池

    nvinfer1::ICudaEngine *engine_; // TensorRT Engine 引用

    std::condition_variable cond_; // 条件变量
    mutable std::mutex mutex_;     // 互斥锁
    bool shutting_down_ = false;   // 关闭标志
    bool initialized_ = false;     // 初始化标志
    int num_streams_ = 4;          // Stream 数量
};

#endif // STREAMPOOL_H
