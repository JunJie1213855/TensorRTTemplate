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
 * @brief StreamContextPair 结构体
 *
 * 用于同时管理 CUDA Stream 和对应的 TensorRT ExecutionContext 的数据结构。
 * 包含流、上下文、输入输出绑定的 GPU 显存指针以及输出结果的主机内存。
 */
struct StreamContextPair
{
    cudaStream_t stream;                                                   /**< @brief CUDA 流 */
    nvinfer1::IExecutionContext *context;                                  /**< @brief TensorRT 执行上下文（裸指针） */
    std::unordered_map<std::string, void *> inputBindings, outputBindings; /**< @brief 输入输出的 CUDA 显存指针 */
    std::unordered_map<std::string, std::shared_ptr<char[]>> outputBlobs;  /**< @brief 输出的主机端内存 */

    /// 移动构造函数
    StreamContextPair(StreamContextPair &&) = default;
    /// 移动赋值运算符
    StreamContextPair &operator=(StreamContextPair &&) = default;

    /// 默认构造函数，初始化为空
    StreamContextPair() : stream(nullptr), context(nullptr) {}

    /**
     * @brief 布尔运算符重载
     * @return bool 如果 stream 或 context 为空返回 true，否则返回 false
     *
     * 用于检查配对是否有效:
     * @code
     * auto pair = pool.acquire();
     * if (!pair) return false;
     * @endcode
     */
    bool operator!() const
    {
        return stream == nullptr || context == nullptr;
    }
};

/**
 * @brief StreamPool 类
 *
 * CUDA Stream 资源池管理器，采用线程安全的获取/归还策略。
 * 用于支持多线程并行推理，每个线程可以独立获取一个 Stream + Context 配对。
 *
 * 设计特点:
 * - 线程安全: 使用互斥锁和条件变量同步
 * - 归还策略: 支持资源复用，减少创建销毁开销
 * - 延迟初始化: 首次获取时才创建资源
 *
 * 使用示例:
 * @code
 * // 初始化（传入 Engine）
 * auto pool = std::make_shared<StreamPool>(engine, 4);
 *
 * // 获取配对
 * auto pair = pool->acquire();
 * if (!pair) return false;
 *
 * // 使用推理
 * pair.context->setInputTensorAddress("input", gpu_ptr);
 * pair.context->enqueueV3(pair.stream);
 *
 * // 归还
 * pool->release(std::move(pair));
 * @endcode
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
     * @brief 构造函数
     *
     * @param  engine         TensorRT Engine 指针
     * @param  num_streams   Stream 池大小，默认为 4
     *
     * 注意: 此构造函数仅保存参数，实际资源在 init() 或首次 acquire() 时创建。
     */
    StreamPool(nvinfer1::ICudaEngine *engine = nullptr, int num_streams = 4);

    /**
     * @brief 初始化 StreamPool
     *
     * @param  engine         TensorRT Engine 指针
     * @param  num_streams   Stream 数量
     *
     * 预先创建指定数量的 Stream + Context 配对。必须在获取前调用一次。
     */
    void init(nvinfer1::ICudaEngine *engine, int num_streams);

    /**
     * @brief 获取一个可用的 Stream + Context 配对
     *
     * @return StreamContextPair 可用的配对，如果池为空则阻塞等待
     *
     * 如果所有配对都在使用中，调用线程会被阻塞直到有配对被归还。
     * 如果池未初始化，会自动调用 init() 进行初始化。
     *
     * @warning 使用完毕后必须调用 release() 归还配对
     */
    StreamContextPair acquire();

    /**
     * @brief 归还 Stream + Context 配对
     *
     * @param  pair  要归还的配对（使用移动语义）
     *
     * 将使用完毕的配对归还到池中，唤醒等待获取的线程。
     * 归还后 pair 变为空，不应再使用。
     */
    void release(StreamContextPair &&pair);

    /**
     * @brief 非阻塞尝试获取配对
     *
     * @return StreamContextPair 可用的配对，如果没有可用则 stream 为 nullptr
     *
     * 立即返回，不会阻塞。如果池为空或所有配对都在使用中，返回无效配对。
     */
    StreamContextPair tryAcquire();

    /**
     * @brief 等待所有配对变为可用
     *
     * 阻塞直到池中所有配对都被归还。通常用于同步所有工作线程。
     */
    void syncAll();

    /**
     * @brief 关闭 Stream 池
     *
     * 设置关闭标志，唤醒所有等待线程。
     * 调用后 acquire() 会返回无效配对。
     */
    void shutdown();

    /**
     * @brief 获取当前可用配对数量
     *
     * @return int 池中可用（未使用）的配对数量
     */
    int availableCount() const;

    /**
     * @brief 获取总配对数量
     *
     * @return int 池中总配对数量（无论是否在使用中）
     */
    int size() const;

    /**
     * @brief 检查是否正在关闭
     *
     * @return bool 如果正在关闭返回 true
     */
    bool isShuttingDown() const;

    /**
     * @brief 检查是否已初始化
     *
     * @return bool 如果已初始化返回 true
     */
    bool isInitialized() const;

    /**
     * @brief 析构函数
     *
     * 销毁所有 CUDA Stream、Context 和显存资源。
     */
    ~StreamPool();

protected:
    /**
     * @brief 默认构造函数
     */
    StreamPool();

private:
    /**
     * @brief 检查是否有可用的配对
     *
     * @return bool 如果池不为空返回 true
     */
    bool hasAvailable() const;

    /**
     * @brief 检查是否有使用中的配对
     *
     * @return bool 如果有配对在使用中返回 true
     */
    bool hasAvailableInUse() const;

    /**
     * @brief 创建 ExecutionContext
     *
     * @param  engine TensorRT Engine 指针
     * @return nvinfer1::IExecutionContext* 创建的上下文指针
     */
    nvinfer1::IExecutionContext *createContext(nvinfer1::ICudaEngine *engine);

    std::queue<StreamContextPair> pool_; /**< @brief Stream + Context 配对池 */
    nvinfer1::ICudaEngine *engine_;      /**< @brief TensorRT Engine 引用 */
    std::condition_variable cond_;       /**< @brief 条件变量，用于线程同步 */
    mutable std::mutex mutex_;           /**< @brief 互斥锁，保护资源池 */
    bool shutting_down_ = false;         /**< @brief 关闭标志 */
    bool initialized_ = false;           /**< @brief 初始化标志 */
    int num_streams_ = 4;                /**< @brief Stream 数量 */
};

#endif // STREAMPOOL_H
