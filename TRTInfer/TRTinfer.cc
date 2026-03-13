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

// Stream operators - 用于方便地输出 TensorRT 类型信息
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

// 重载 << 操作符，用于输出 nvinfer1::DataType（数据类型）
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

// Logger 类 - TensorRT 日志记录器实现
class Logger : public nvinfer1::ILogger
{
public:
    // 重写 log 方法，过滤掉 INFO 级别的日志，只输出 WARNING 及以上级别
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

    // TRTInfer的Implementation类
    class TRTInfer::Impl
    {

    private:
        // 内部类：推理任务封装
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
        // 构造函数：加载引擎文件并初始化所有资源
        Impl(const std::string &engine_path, int num_thread, TRTInfer *parent);

        // 析构函数：释放 CUDA 资源
        ~Impl();

        // 推送数据到队列中
        std::future<BlobType> PostQueue(const BlobType &input_blob);

        // 线程工作队列
        void workThread();

        // 推理接口：接收 cv::Mat 类型的输入数据（自动处理类型转换）
        BlobType infer(
            const BlobType &input_blob);

        // 推理任务
        BlobType infer_task(const BlobType &input_blob);

    public:
        // 公共 getter 方法（供 TRTInfer 公共接口使用）
        std::vector<std::string> getInputNames() const { return input_names; }

        std::vector<std::string> getOutputNames() const { return output_names; }

        std::vector<int> getInputShapeVec(const std::string &name) const
        {
            auto it = current_input_shapes.find(name);
            return (it != current_input_shapes.end()) ? it->second : std::vector<int>();
        }

        std::vector<int> getOutputShapeVec(const std::string &name) const
        {
            auto it = output_shape.find(name);
            return (it != output_shape.end()) ? it->second : std::vector<int>();
        }

    private:
        // 私有方法 - 初始化  runtime、engine
        void load_engine(const std::string &engine_path); // 从文件加载 TensorRT 引擎

        void get_InputProperty(); // 获取所有输入张量名称和信息

        void get_OutputProperty(); // 获取所有输出张量名称和信息

        void allocate_pair(); // 多线程：给streampool分配stream、context、mem

        void create_workthreads(); // 多线程：创建工作线程

        void allocBindings(std::unordered_map<std::string, void *> &inputBindings,
                           std::unordered_map<std::string, void *> &outputBindings,
                           nvinfer1::IExecutionContext *context); // 为输入输出分配 CUDA 显存

        void allocOutBlob(std::unordered_map<std::string, std::shared_ptr<char[]>> &outputBlob); // 设置输出张量的显存地址

        // 上传输入：类型转换、验证尺寸、拷贝至 GPU
        void uploadInput(const std::string &name,
                         const cv::Mat &mat,
                         std::unordered_map<std::string, void *> &inputBindings_,
                         cudaStream_t stream_,
                         nvinfer1::IExecutionContext *context_);

        // 下载输出：分配空间、拷贝至 CPU、包装为 cv::Mat
        void downloadOutput(std::unordered_map<std::string, std::shared_ptr<char[]>> &output_blob,
                            cudaStream_t stream_,
                            nvinfer1::IExecutionContext *context_,
                            std::unordered_map<std::string, void *> &OutputBindings_);

    private: // 共用数据
             // 指向父类的指针（用于回调）
        TRTInfer *parent_;
        // TensorRT 核心对象, 多线程共用单个runtime和engine
        std::unique_ptr<nvinfer1::IRuntime> runtime = nullptr;   // 运行时，用于反序列化引擎
        std::unique_ptr<nvinfer1::ICudaEngine> engine = nullptr; // CUDA 引擎，包含优化后的网络结构

        // 形状支持
        std::unordered_map<std::string, std::vector<int>> current_input_shapes; // 当前使用的输入形状

        // 张量元数据, 多线程模式下只读, 不需要加锁
        std::vector<std::string> input_names, output_names;              // 输入输出张量名称列表
        std::unordered_map<std::string, size_t> input_size, output_size; // 张量字节大小
        std::unordered_map<std::string, std::vector<int>> output_shape;  // 输出张量形状
        Logger logger;                                                   // 日志记录器

    private: // 多线程模式
        // 任务队列
        std::queue<std::unique_ptr<InferTask>> task_queues_;

        // <stream, context, cuda_data> 池
        std::shared_ptr<StreamPool> streampool;

        // thread 池 -> 生产者消费者模式
        int num_threads_;
        bool b_stop_ = false;
        std::vector<std::thread> thread_pool;

        std::condition_variable cond_; // 池获取数据
        std::mutex task_mutext_;
    };

    // TRTInfer 公共接口 - 使用 Pimpl 模式将实现委托给 Impl 类
    TRTInfer::TRTInfer(const std::string &engine_path, int num_thread)
        : pImpl(std::make_unique<Impl>(engine_path, num_thread, this))
    {
    }

    // 析构函数：使用默认实现（unique_ptr 自动释放 Impl）
    TRTInfer::~TRTInfer() = default;

    // 重载 () 运算符：cv::Mat 版本的推理接口
    BlobType TRTInfer::operator()(
        const BlobType &input_blob)
    {
        return pImpl->infer(input_blob);
    }

    std::future<BlobType> TRTInfer::PostQueue(const BlobType &input_blob)
    {
        return pImpl->PostQueue(input_blob);
    }

    // 辅助函数：将 vector 转为 TensorShape

    // 获取所有输入张量名称
    std::vector<std::string> TRTInfer::getInputNames() const
    {
        return pImpl->getInputNames();
    }

    // 获取所有输出张量名称
    std::vector<std::string> TRTInfer::getOutputNames() const
    {
        return pImpl->getOutputNames();
    }

    // 获取指定输入张量形状
    TensorShape TRTInfer::getInputShape(const std::string &name) const
    {
        return utility::vectorToShape(pImpl->getInputShapeVec(name));
    }

    // 获取指定输出张量形状
    TensorShape TRTInfer::getOutputShape(const std::string &name) const
    {
        return utility::vectorToShape(pImpl->getOutputShapeVec(name));
    }

    // TRTInfer::Impl 实现
    TRTInfer::Impl::Impl(const std::string &engine_path, int num_thread, TRTInfer *parent)
        : parent_(parent), num_threads_(num_thread), logger()
    {
        // 按正确顺序初始化引擎和相关资源
        load_engine(engine_path); // 1. 加载引擎文件
        get_InputProperty();      // 2. 获取输入张量信息
        get_OutputProperty();     // 3. 获取输出张量信息
        allocate_pair();          // 4. 创建 stream、context、内存
        create_workthreads();     // 5. 创建接收队列
    }

    // 析构函数：释放所有 CUDA 资源
    TRTInfer::Impl::~Impl()
    {
        // 设置停止标志
        b_stop_ = true;
        cond_.notify_all();
        // 等待线程结束
        for (auto &thread : thread_pool)
            thread.join();
        std::cout << "[ TRT释放 ]" << std::endl;
    }

    // 从文件加载 TensorRT 引擎并进行反序列化
    void TRTInfer::Impl::load_engine(const std::string &engine_path)
    {
        // 以二进制模式读取引擎文件
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good())
        {
            file.close();
            std::cerr << "Error reading engine file" << std::endl;
            throw std::runtime_error("Error reading engine file");
        }
        // 获取文件大小
        file.seekg(0, file.end);
        const size_t fsize = file.tellg();
        file.seekg(0, file.beg);
        // 读取整个文件到内存
        std::vector<char> engineData(fsize);
        file.read(engineData.data(), fsize);
        file.close();

        // 创建 TensorRT 运行时
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime)
        {
            std::cerr << "Failed to create runtime" << std::endl;
            throw std::runtime_error("Failed to create runtime");
        }

        // 初始化 TensorRT 插件（支持自定义层）
        initLibNvInferPlugins(&logger, "TRT");
        // 反序列化引擎
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize));
        if (!engine)
        {
            std::cerr << "Failed to create engine" << std::endl;
            throw std::runtime_error("Failed to create engine");
        }
    }

    // 获取引擎中所有输入张量的信息（名称、形状、数据类型等）
    void TRTInfer::Impl::get_InputProperty()
    {
        // 遍历所有 IO 张量
        for (int i = 0; i < engine->getNbIOTensors(); i++)
        {
            const char *name = engine->getIOTensorName(i);
            // 只处理输入类型的张量
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            {
                // 打印张量信息
                std::cout << "input tensor name : " << name
                          << ", tensor shape : " << engine->getTensorShape(name)
                          << ", tensor type : " << engine->getTensorDataType(name)
                          << ", tensor format : " << engine->getTensorFormatDesc(name)
                          << std::endl;
                // 存储张量的名称
                input_names.emplace_back(std::string(name));
                // 计算并存储张量的字节大小
                input_size[std::string(name)] = utility::getTensorbytes(engine->getTensorShape(name), engine->getTensorDataType(name));

                // 保存当前张量尺寸
                nvinfer1::Dims dims = engine->getTensorShape(name);
                std::vector<int> dim;
                dim.reserve(dims.nbDims);
                for (int i = 0; i < dims.nbDims; i++)
                    dim.emplace_back(dims.d[i]);
                current_input_shapes[name] = std::move(dim);
            }
        }
    }

    // 获取引擎中所有输出张量的信息（名称、形状、数据类型等）
    void TRTInfer::Impl::get_OutputProperty()
    {
        // 遍历所有 IO 张量
        for (int i = 0; i < engine->getNbIOTensors(); i++)
        {
            const char *name = engine->getIOTensorName(i);
            // 只处理输出类型的张量
            if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
            {
                // 打印张量信息
                std::cout << "output tensor name : " << name
                          << ", tensor shape : " << engine->getTensorShape(name)
                          << ", tensor type : " << engine->getTensorDataType(name)
                          << ", tensor format : " << engine->getTensorFormatDesc(name)
                          << std::endl;
                output_names.emplace_back(std::string(name));
                // 计算并存储张量的字节大小
                output_size[std::string(name)] = utility::getTensorbytes(engine->getTensorShape(name), engine->getTensorDataType(name));
                // 存储输出张量的形状信息
                nvinfer1::Dims dims = engine->getTensorShape(name);
                std::vector<int> dim;
                dim.reserve(dims.nbDims);
                for (int i = 0; i < dims.nbDims; i++)
                    dim.emplace_back(dims.d[i]);
                output_shape[std::string(name)] = dim;
            }
        }
    }
    void TRTInfer::Impl::allocBindings(std::unordered_map<std::string, void *> &inputBindings,
                                       std::unordered_map<std::string, void *> &outputBindings,
                                       nvinfer1::IExecutionContext *context)
    {
        // 在设备端预分配输入、输出缓冲区
        for (int i = 0; i < input_names.size(); i++)
        {
            inputBindings[input_names[i]] = utility::safeCudaMalloc(input_size[input_names[i]]);   // 输入
            context->setInputTensorAddress(input_names[i].c_str(), inputBindings[input_names[i]]); // 上下文绑定地址
        }
        // 为每个输出张量分配显存
        for (int i = 0; i < output_names.size(); i++)
        {
            outputBindings[output_names[i]] = utility::safeCudaMalloc(output_size[output_names[i]]);   // 输出
            context->setOutputTensorAddress(output_names[i].c_str(), outputBindings[output_names[i]]); // 上下文绑定地址
        }
    }

    void TRTInfer::Impl::allocOutBlob(std::unordered_map<std::string, std::shared_ptr<char[]>> &outputBlob)
    {
        // 在主机端CPU预分配输出数据缓冲区（用于接收推理结果）
        for (const auto &name : output_names)
        {
            size_t datasize = output_size[name];
            outputBlob[name] = std::shared_ptr<char[]>(new char[datasize]);
        }
    }

    // 推理函数：接收 cv::Mat 类型的输入数据，自动处理类型转换，返回 cv::Mat 格式的输出
    BlobType TRTInfer::Impl::infer(
        const BlobType &input_blob)
    {
        auto future = this->PostQueue(input_blob);
        BlobType results = std::move(future.get());
        return results;
    }

    // 推理任务
    BlobType TRTInfer::Impl::infer_task(const BlobType &input_blob)
    {
        // 获取池数据
        auto pair = streampool->acquire();
        // 析构延迟归还
        utility::Defer defer([&pair, this]()
                             { this->streampool->release(std::move(pair)); });
        // 1.上传数据
        for (const auto &[name, mat] : input_blob)
        {
            uploadInput(name, mat, pair.inputBindings, pair.stream, pair.context);
        }

        // 2.异步执行
        pair.context->enqueueV3(pair.stream);

        // 3. 处理输出：分配空间、拷贝至 CPU、包装为 cv::Mat
        downloadOutput(pair.outputBlobs, pair.stream, pair.context, pair.outputBindings);
        
        // 等待cuda stream操作完成
        cudaStreamSynchronize(pair.stream);

        BlobType tmp_results;
        for (auto &name : output_names)
        {
            cv::Mat temp(
                output_shape[name].size(),
                output_shape[name].data(),
                utility::typeRt2Cv(engine->getTensorDataType(name.c_str())),
                pair.outputBlobs[name].get());
            tmp_results[name] = temp.clone();
        }
        // NRVO 触发移动赋值
        return tmp_results;
    }

    // 上传输入：类型转换、验证尺寸、拷贝至 GPU
    void TRTInfer::Impl::uploadInput(
        const std::string &name,
        const cv::Mat &mat,
        std::unordered_map<std::string, void *> &inputBindings_,
        cudaStream_t stream_,
        nvinfer1::IExecutionContext *context_)
    {
        cv::Mat cpu_ptr = mat;

        // 类型转换
        if (utility::typeCv2Rt(cpu_ptr.type()) != engine->getTensorDataType(name.c_str()))
        {
            cpu_ptr.convertTo(cpu_ptr, utility::typeRt2Cv(engine->getTensorDataType(name.c_str())));
        }

        auto iter = inputBindings_.find(name);
        if (iter == inputBindings_.end())
            return;

        void *cuda_ptr = iter->second;

        // 计算 TensorRT 期望的数据大小
        nvinfer1::Dims dims;
        dims.nbDims = current_input_shapes[name].size();
        for (size_t i = 0; i < current_input_shapes[name].size(); i++)
        {
            dims.d[i] = current_input_shapes[name][i];
        }
        size_t data_size = utility::getTensorbytes(dims, engine->getTensorDataType(name.c_str()));

        // 验证尺寸
        size_t mat_size = cpu_ptr.total() * cpu_ptr.elemSize();
        if (data_size != mat_size)
        {
            std::cerr << "[ERROR] Input tensor size mismatch for '" << name << "': "
                      << "required " << data_size << " bytes, "
                      << "but cv::Mat has " << mat_size << " bytes. "
                      << "Mat shape: " << cpu_ptr.size[0] << "x" << cpu_ptr.size[1] << "x" << cpu_ptr.size[2] << "x" << cpu_ptr.size[3]
                      << ", expected tensor shape: ";
            for (size_t i = 0; i < current_input_shapes.at(name).size(); i++)
            {
                std::cerr << current_input_shapes.at(name)[i] << (i < current_input_shapes.at(name).size() - 1 ? "x" : "");
            }
            std::cerr << std::endl;
            throw std::runtime_error("Input tensor size mismatch");
        }

        // 检查连续性
        if (!cpu_ptr.isContinuous())
        {
            std::cerr << "[WARNING] Input cv::Mat for '" << name << "' is not continuous" << std::endl;
        }

        // 拷贝至 GPU
        cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr.data, data_size, cudaMemcpyHostToDevice, stream_);
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }
        context_->setInputTensorAddress(name.c_str(), cuda_ptr);
    }

    // 下载输出：分配空间、拷贝至 CPU、包装为 cv::Mat

    void TRTInfer::Impl::downloadOutput(std::unordered_map<std::string, std::shared_ptr<char[]>> &output_blob,
                                        cudaStream_t stream_,
                                        nvinfer1::IExecutionContext *context_,
                                        std::unordered_map<std::string, void *> &OutputBindings_)
    {
        for (const auto &name : output_names)
        {
            // 获取输出形状
            nvinfer1::Dims out_shape = context_->getTensorShape(name.c_str());
            size_t actual_size = utility::getTensorbytes(out_shape, engine->getTensorDataType(name.c_str()));

            // 验证缓冲区
            if (actual_size != output_size[name])
            {
                std::cerr << "[ERROR] Output buffer size insufficient for '" << name << "': "
                          << "required " << actual_size << " bytes, "
                          << "but only " << output_size[name] << " bytes allocated" << std::endl;
                throw std::runtime_error("Output buffer size insufficient");
            }

            // 拷贝至 CPU
            void *ptr = static_cast<void *>(output_blob[name].get());
            cudaError_t err = cudaMemcpyAsync(ptr, OutputBindings_[name], actual_size, cudaMemcpyDeviceToHost, stream_);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    std::future<BlobType> TRTInfer::Impl::PostQueue(const BlobType &input_blob)
    {
        // 创建任务
        auto task = std::make_unique<InferTask>(
            std::bind(&TRTInfer::Impl::infer_task, this, std::placeholders::_1), static_cast<const BlobType &>(input_blob));
        // 获取返回值
        auto future = task->get_future();
        // 任务入队
        {
            std::lock_guard<std::mutex> lock(task_mutext_);
            task_queues_.push(std::move(task));
            cond_.notify_one();
        }
        return future;
    }

    void TRTInfer::Impl::workThread()
    {
        while (true)
        {
            std::unique_ptr<InferTask> task;
            // 获取任务
            {
                std::unique_lock<std::mutex> lock(task_mutext_);
                cond_.wait(lock, [this]()
                           { return b_stop_ || !task_queues_.empty(); });
                // 停止直接拜拜
                if (b_stop_)
                    return;
                // 移动语义
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
                    std::cerr << "Task execution failed: " << e.what() << "\n";
                }
            }
        }
    }
    void TRTInfer::Impl::allocate_pair()
    {
        streampool = std::make_shared<StreamPool>(this->engine.get(), num_threads_);
        for (int i = 0; i < num_threads_; i++)
        {
            auto pair = streampool->acquire();
            allocBindings(pair.inputBindings, pair.outputBindings, pair.context);
            allocOutBlob(pair.outputBlobs);
            streampool->release(std::move(pair));
        }
    }
    void TRTInfer::Impl::create_workthreads()
    {
        for (int i = 0; i < num_threads_; i++)
            thread_pool.emplace_back([this]()
                                     { workThread(); });
    }
}