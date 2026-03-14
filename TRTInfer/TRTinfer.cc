#include "TRTinfer.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include "utility.h"
#include "StreamPool.h"

/**
 * @brief 输出 nvinfer1::Dims 维度的流操作符
 */
std::ostream &operator<<(std::ostream &cout, const nvinfer1::Dims &dim)
{
    for (int i = 0; i < dim.nbDims; i++)
    {
        if (i < dim.nbDims - 1)
        {
            cout << dim.d[i] << " X ";
        }
        else
            cout << dim.d[i];
    }
    return cout;
}

/**
 * @brief 输出 nvinfer1::DataType 数据类型的流操作符
 */
std::ostream &operator<<(std::ostream &cout, const nvinfer1::DataType &type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBF16:
        cout << "kBF16";
        break;
    case nvinfer1::DataType::kBOOL:
        cout << "kBOOL";
        break;
    case nvinfer1::DataType::kFLOAT:
        cout << "kFLOAT";
        break;
    case nvinfer1::DataType::kFP8:
        cout << "kFP8";
        break;
    case nvinfer1::DataType::kHALF:
        cout << "kHALF";
        break;
    case nvinfer1::DataType::kINT32:
        cout << "kINT32";
        break;
    case nvinfer1::DataType::kINT4:
        cout << "kINT4";
        break;
    case nvinfer1::DataType::kINT64:
        cout << "kINT64";
        break;
    case nvinfer1::DataType::kINT8:
        cout << "kINT8";
        break;
    case nvinfer1::DataType::kUINT8:
        cout << "kUINT8";
        break;
    default:
        break;
    }
    return cout;
}

/**
 * @brief TensorRT 日志记录器
 *
 * 过滤 INFO 级别日志，只输出 WARNING 及以上级别
 */
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};

namespace TRT
{

    /**
     * @brief TRTInfer 实现类 (Pimpl 模式)
     */
    class TRTInfer::Impl
    {

    private:
        /**
         * @brief 推理任务封装类
         *
         * 封装推理函数和参数，通过 promise/future 返回结果
         */
        class InferTask
        {
        public:
            InferTask(std::function<BlobType(const BlobType &)> F,
                      const BlobType &args) : task_(F), args_(args) {}

            void execute()
            {
                try
                {
                    promise_.set_value(std::invoke(task_, args_));
                }
                catch (...)
                {
                    promise_.set_exception(std::current_exception());
                }
            }

            std::future<BlobType> get_future()
            {
                return promise_.get_future();
            }

        private:
            std::function<BlobType(const BlobType &)> task_;
            const BlobType &args_;
            std::promise<BlobType> promise_;
        };

    public:
        /** @brief 构造函数 */
        Impl(const std::string &engine_path, int num_thread, TRTInfer *parent);

        /** @brief 析构函数 */
        ~Impl();

        /** @brief 推送任务到队列 */
        std::future<BlobType> PostQueue(const BlobType &input_blob);

        /** @brief 同步推理 */
        BlobType infer(const BlobType &input_blob);

        /** @brief 实际推理任务执行 */
        BlobType infer_task(const BlobType &input_blob);

        /** @brief 初始化引擎和资源 */
        void Initialized();

    public:
        /** @brief 获取输入张量名称 */
        std::vector<std::string> getInputNames() const { return input_names_; }

        /** @brief 获取输出张量名称 */
        std::vector<std::string> getOutputNames() const { return output_names_; }

        /** @brief 获取输入张量形状 */
        std::vector<int> getInputShapeVec(const std::string &name) const
        {
            auto it = current_input_shapes_.find(name);
            return (it != current_input_shapes_.end()) ? it->second : std::vector<int>();
        }

        /** @brief 获取输出张量形状 */
        std::vector<int> getOutputShapeVec(const std::string &name) const
        {
            auto it = output_shape_.find(name);
            return (it != output_shape_.end()) ? it->second : std::vector<int>();
        }

    private:
        /** @brief 工作线程函数 */
        void workThread();

        /** @brief 从文件加载引擎 */
        void LoadEngine(const std::string &engine_path);

        /** @brief 获取输入张量属性 */
        void getInputProperty();

        /** @brief 获取输出张量属性 */
        void getOutputProperty();

        /** @brief 分配 Stream/Context/内存 */
        void allocatePair();

        /** @brief 创建工作线程 */
        void createWorkthreads();

        /** @brief 分配输入输出显存 */
        void allocBindings(std::unordered_map<std::string, void *> &inputBindings,
                           std::unordered_map<std::string, void *> &outputBindings,
                           nvinfer1::IExecutionContext *context);

        /** @brief 分配输出主机内存 */
        void allocOutBlob(std::unordered_map<std::string, std::shared_ptr<char[]>> &outputBlob);

        /** @brief 上传输入数据到 GPU */
        void uploadInput(const std::string &name,
                         const cv::Mat &mat,
                         std::unordered_map<std::string, void *> &inputBindings,
                         cudaStream_t stream,
                         nvinfer1::IExecutionContext *context);

        /** @brief 下载输出数据到 CPU */
        void downloadOutput(std::unordered_map<std::string, std::shared_ptr<char[]>> &output_blob,
                            cudaStream_t stream,
                            nvinfer1::IExecutionContext *context,
                            std::unordered_map<std::string, void *> &OutputBindings);

    private:
        std::string engine_path_;                /**< @brief 引擎文件路径 */
        bool initialized_ = false;               /**< @brief 初始化标志 */

        TRTInfer *parent_;                        /**< @brief 父类指针 */

        std::unique_ptr<nvinfer1::IRuntime> runtime_;   /**< @brief TensorRT 运行时 */
        std::unique_ptr<nvinfer1::ICudaEngine> engine_; /**< @brief TensorRT 引擎 */

        std::unordered_map<std::string, std::vector<int>> current_input_shapes_; /**< @brief 当前输入形状 */

        std::vector<std::string> input_names_, output_names_;                    /**< @brief 输入输出名称 */
        std::unordered_map<std::string, size_t> input_size_, output_size_;     /**< @brief 输入输出字节大小 */
        std::unordered_map<std::string, std::vector<int>> output_shape_;       /**< @brief 输出形状 */
        Logger logger; /**< @brief 日志记录器 */

    private:
        std::queue<std::unique_ptr<InferTask>> task_queues_; /**< @brief 任务队列 */

        std::shared_ptr<StreamPool> streampool_; /**< @brief Stream 池 */

        int num_threads_;                              /**< @brief 线程数量 */
        bool b_stop_ = false;                         /**< @brief 停止标志 */
        std::vector<std::thread> thread_pool;        /**< @brief 线程池 */

        std::condition_variable cond_; /**< @brief 条件变量 */
        std::mutex task_mutext_;      /**< @brief 任务互斥锁 */
    };

    // TRTInfer 公共接口实现

    TRTInfer::TRTInfer(const std::string &engine_path, int num_thread)
        : pImpl(std::make_unique<Impl>(engine_path, num_thread, this))
    {
    }

    TRTInfer::~TRTInfer() = default;

    BlobType TRTInfer::operator()(const BlobType &input_blob)
    {
        return pImpl->infer(input_blob);
    }

    std::future<BlobType> TRTInfer::PostQueue(const BlobType &input_blob)
    {
        return pImpl->PostQueue(input_blob);
    }

    std::vector<std::string> TRTInfer::getInputNames() const
    {
        return pImpl->getInputNames();
    }

    std::vector<std::string> TRTInfer::getOutputNames() const
    {
        return pImpl->getOutputNames();
    }

    TensorShape TRTInfer::getInputShape(const std::string &name) const
    {
        return utility::vectorToShape(pImpl->getInputShapeVec(name));
    }

    TensorShape TRTInfer::getOutputShape(const std::string &name) const
    {
        return utility::vectorToShape(pImpl->getOutputShapeVec(name));
    }

    // TRTInfer::Impl 实现

    TRTInfer::Impl::Impl(const std::string &engine_path, int num_thread, TRTInfer *parent)
        : engine_path_(engine_path), num_threads_(num_thread), parent_(parent), logger()
    {
    }

    /**
     * @brief 初始化引擎：加载引擎、获取张量信息、分配资源、创建线程
     */
    void TRTInfer::Impl::Initialized()
    {
        LoadEngine(engine_path_);
        getInputProperty();
        getOutputProperty();
        allocatePair();
        createWorkthreads();
    }

    /**
     * @brief 析构函数：停止线程并释放资源
     */
    TRTInfer::Impl::~Impl()
    {
        std::cout << "[TRTInfer::Impl] 开始关闭" << std::endl;
        b_stop_ = true;
        cond_.notify_all();
        for (auto &thread : thread_pool)
            thread.join();
        std::cout << "[TRTInfer::Impl] 释放" << std::endl;
    }

    /**
     * @brief 从文件加载 TensorRT 引擎
     */
    void TRTInfer::Impl::LoadEngine(const std::string &engine_path)
    {
        // 读取引擎文件
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good())
        {
            file.close();
            std::cerr << "[TRTInfer::Impl] Error reading engine file" << std::endl;
            throw std::runtime_error("Error reading engine file: " + engine_path);
        }

        file.seekg(0, file.end);
        const size_t fsize = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engineData(fsize);
        file.read(engineData.data(), fsize);
        file.close();

        // 创建运行时
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_)
        {
            std::cerr << "[TRTInfer::Impl] Failed to create runtime" << std::endl;
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        // 初始化插件并反序列化引擎
        initLibNvInferPlugins(&logger, "TRT");
        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), fsize));
        if (!engine_)
        {
            std::cerr << "[TRTInfer::Impl] Failed to create engine" << std::endl;
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
    }

    /**
     * @brief 获取所有输入张量信息
     */
    void TRTInfer::Impl::getInputProperty()
    {
        for (int i = 0; i < engine_->getNbIOTensors(); i++)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                std::cout << "[TRTInfer::Impl] input tensor name : " << name
                          << ", tensor shape : " << engine_->getTensorShape(name)
                          << ", tensor type : " << engine_->getTensorDataType(name)
                          << ", tensor format : " << engine_->getTensorFormatDesc(name)
                          << std::endl;

                input_names_.emplace_back(std::string(name));
                input_size_[std::string(name)] = utility::getTensorbytes(
                    engine_->getTensorShape(name), engine_->getTensorDataType(name));

                // 保存形状
                nvinfer1::Dims dims = engine_->getTensorShape(name);
                std::vector<int> dim;
                dim.reserve(dims.nbDims);
                for (int i = 0; i < dims.nbDims; i++)
                    dim.emplace_back(dims.d[i]);
                current_input_shapes_[name] = std::move(dim);
            }
        }
    }

    /**
     * @brief 获取所有输出张量信息
     */
    void TRTInfer::Impl::getOutputProperty()
    {
        for (int i = 0; i < engine_->getNbIOTensors(); i++)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::cout << "[TRTInfer::Impl] output tensor name : " << name
                          << ", tensor shape : " << engine_->getTensorShape(name)
                          << ", tensor type : " << engine_->getTensorDataType(name)
                          << ", tensor format : " << engine_->getTensorFormatDesc(name)
                          << std::endl;

                output_names_.emplace_back(std::string(name));
                output_size_[std::string(name)] = utility::getTensorbytes(
                    engine_->getTensorShape(name), engine_->getTensorDataType(name));

                nvinfer1::Dims dims = engine_->getTensorShape(name);
                std::vector<int> dim;
                dim.reserve(dims.nbDims);
                for (int i = 0; i < dims.nbDims; i++)
                    dim.emplace_back(dims.d[i]);
                output_shape_[std::string(name)] = dim;
            }
        }
    }

    /**
     * @brief 为输入输出分配 GPU 显存
     */
    void TRTInfer::Impl::allocBindings(std::unordered_map<std::string, void *> &inputBindings,
                                       std::unordered_map<std::string, void *> &outputBindings,
                                       nvinfer1::IExecutionContext *context)
    {
        // 分配输入显存
        for (int i = 0; i < input_names_.size(); i++)
        {
            void *ptr = utility::safeCudaMalloc(input_size_[input_names_[i]]);
            if (!ptr)
                throw std::runtime_error("Failed to allocate GPU memory");
            inputBindings[input_names_[i]] = ptr;
            context->setInputTensorAddress(input_names_[i].c_str(), inputBindings[input_names_[i]]);
        }

        // 分配输出显存
        for (int i = 0; i < output_names_.size(); i++)
        {
            void *ptr = utility::safeCudaMalloc(output_size_[output_names_[i]]);
            if (!ptr)
                throw std::runtime_error("Failed to allocate GPU memory");
            outputBindings[output_names_[i]] = ptr;
            context->setOutputTensorAddress(output_names_[i].c_str(), outputBindings[output_names_[i]]);
        }
    }

    /**
     * @brief 分配输出主机内存
     */
    void TRTInfer::Impl::allocOutBlob(std::unordered_map<std::string, std::shared_ptr<char[]>> &outputBlob)
    {
        for (const auto &name : output_names_)
        {
            size_t datasize = output_size_[name];
            outputBlob[name] = std::shared_ptr<char[]>(new char[datasize]);
        }
    }

    /**
     * @brief 同步推理
     */
    BlobType TRTInfer::Impl::infer(const BlobType &input_blob)
    {
        auto future = this->PostQueue(input_blob);
        BlobType results = std::move(future.get());
        return results;
    }

    /**
     * @brief 执行推理任务
     */
    BlobType TRTInfer::Impl::infer_task(const BlobType &input_blob)
    {
        // 获取资源
        auto pair = streampool_->acquire();
        if (!pair)
        {
            std::cout << "[TRTInfer] pair empty!" << std::endl;
            return BlobType{};
        }

        // 延迟归还
        utility::Defer defer([&pair, this]()
                             { this->streampool_->release(std::move(pair)); });

        // 上传输入
        for (const auto &[name, mat] : input_blob)
        {
            uploadInput(name, mat, pair.inputBindings, pair.stream, pair.context);
        }

        // 执行推理
        pair.context->enqueueV3(pair.stream);

        // 下载输出
        downloadOutput(pair.outputBlobs, pair.stream, pair.context, pair.outputBindings);

        // 等待完成
        cudaStreamSynchronize(pair.stream);

        // 封装结果
        BlobType tmp_results;
        for (auto &name : output_names_)
        {
            cv::Mat temp(
                output_shape_[name].size(),
                output_shape_[name].data(),
                utility::typeRt2Cv(engine_->getTensorDataType(name.c_str())),
                pair.outputBlobs[name].get());
            tmp_results[name] = temp.clone();
        }
        return tmp_results;
    }

    /**
     * @brief 上传输入数据到 GPU
     */
    void TRTInfer::Impl::uploadInput(
        const std::string &name,
        const cv::Mat &mat,
        std::unordered_map<std::string, void *> &inputBindings,
        cudaStream_t stream,
        nvinfer1::IExecutionContext *context)
    {
        cv::Mat cpu_ptr = mat;

        // 类型转换
        if (utility::typeCv2Rt(cpu_ptr.type()) != engine_->getTensorDataType(name.c_str()))
        {
            cpu_ptr.convertTo(cpu_ptr, utility::typeRt2Cv(engine_->getTensorDataType(name.c_str())));
        }

        auto iter = inputBindings.find(name);
        if (iter == inputBindings.end())
            return;

        void *cuda_ptr = iter->second;

        // 计算大小并验证
        nvinfer1::Dims dims;
        dims.nbDims = current_input_shapes_[name].size();
        for (size_t i = 0; i < current_input_shapes_[name].size(); i++)
        {
            dims.d[i] = current_input_shapes_[name][i];
        }
        size_t data_size = utility::getTensorbytes(dims, engine_->getTensorDataType(name.c_str()));

        size_t mat_size = cpu_ptr.total() * cpu_ptr.elemSize();
        if (data_size != mat_size)
        {
            std::cerr << "[TRTInfer::Impl - ERROR] Input tensor size mismatch for '" << name << "': "
                      << "required " << data_size << " bytes, "
                      << "but cv::Mat has " << mat_size << " bytes. "
                      << "Mat shape: " << cpu_ptr.size[0] << "x" << cpu_ptr.size[1] << "x" << cpu_ptr.size[2] << "x" << cpu_ptr.size[3]
                      << ", expected tensor shape: ";
            for (size_t i = 0; i < current_input_shapes_.at(name).size(); i++)
            {
                std::cerr << current_input_shapes_.at(name)[i] << (i < current_input_shapes_.at(name).size() - 1 ? "x" : "");
            }
            std::cerr << std::endl;
            throw std::runtime_error("[TRTInfer::Impl] Input tensor size mismatch");
        }

        // 检查连续性
        if (!cpu_ptr.isContinuous())
        {
            std::cerr << "[TRTInfer::Impl - WARNING] Input cv::Mat for '" << name << "' is not continuous" << std::endl;
        }

        // 拷贝到 GPU
        cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr.data, data_size, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
        {
            std::cerr << "[TRTInfer::Impl] CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }
        context->setInputTensorAddress(name.c_str(), cuda_ptr);
    }

    /**
     * @brief 下载输出数据到 CPU
     */
    void TRTInfer::Impl::downloadOutput(std::unordered_map<std::string, std::shared_ptr<char[]>> &output_blob,
                                        cudaStream_t stream,
                                        nvinfer1::IExecutionContext *context,
                                        std::unordered_map<std::string, void *> &OutputBindings)
    {
        for (const auto &name : output_names_)
        {
            // 获取实际输出形状
            nvinfer1::Dims out_shape = context->getTensorShape(name.c_str());
            size_t actual_size = utility::getTensorbytes(out_shape, engine_->getTensorDataType(name.c_str()));

            // 验证缓冲区
            if (actual_size != output_size_[name])
            {
                std::cerr << "[ERROR] Output buffer size insufficient for '" << name << "': "
                          << "required " << actual_size << " bytes, "
                          << "but only " << output_size_[name] << " bytes allocated" << std::endl;
                throw std::runtime_error("Output buffer size insufficient");
            }

            // 拷贝到主机
            void *ptr = static_cast<void *>(output_blob[name].get());
            cudaError_t err = cudaMemcpyAsync(ptr, OutputBindings[name], actual_size, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "[TRTInfer::Impl] CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    /**
     * @brief 推送任务到队列
     */
    std::future<BlobType> TRTInfer::Impl::PostQueue(const BlobType &input_blob)
    {
        auto task = std::make_unique<InferTask>(
            std::bind(&TRTInfer::Impl::infer_task, this, std::placeholders::_1), static_cast<const BlobType &>(input_blob));
        auto future = task->get_future();

        {
            std::lock_guard<std::mutex> lock(task_mutext_);
            task_queues_.push(std::move(task));
            cond_.notify_one();
        }
        return future;
    }

    /**
     * @brief 工作线程函数
     */
    void TRTInfer::Impl::workThread()
    {
        while (true)
        {
            std::unique_ptr<InferTask> task;
            {
                std::unique_lock<std::mutex> lock(task_mutext_);
                cond_.wait(lock, [this]()
                           { return b_stop_ || !task_queues_.empty(); });
                if (b_stop_)
                    return;
                task = std::move(task_queues_.front());
                task_queues_.pop();
            }

            // 执行任务
            if (task)
            {
                try
                {
                    task->execute();
                }
                catch (const std::exception &e)
                {
                    std::cerr << "[TRTInfer::Impl] Task execution failed: " << e.what() << "\n";
                }
            }
        }
    }

    /**
     * @brief 分配 Stream 池资源
     */
    void TRTInfer::Impl::allocatePair()
    {
        streampool_ = std::make_shared<StreamPool>(this->engine_.get(), num_threads_);
        for (int i = 0; i < num_threads_; i++)
        {
            auto pair = streampool_->acquire();
            allocBindings(pair.inputBindings, pair.outputBindings, pair.context);
            allocOutBlob(pair.outputBlobs);
            streampool_->release(std::move(pair));
        }
    }

    /**
     * @brief 创建工作线程
     */
    void TRTInfer::Impl::createWorkthreads()
    {
        for (int i = 0; i < num_threads_; i++)
            thread_pool.emplace_back([this]()
                                     { workThread(); });
    }

    /**
     * @brief 初始化引擎
     */
    void TRTInfer::Init()
    {
        pImpl->Initialized();
    }

    /**
     * @brief 工厂方法：创建 TRTInfer 实例
     */
    std::shared_ptr<TRTInfer> TRTInfer::create(const std::string &engine_path, int num_thread)
    {
        auto instance_ = std::shared_ptr<TRTInfer>(new TRTInfer(engine_path, num_thread));
        instance_->Init();
        return instance_;
    }
}
