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

// ============================================================================
// Logger 类 - TensorRT 日志记录器实现
// 用于接收和输出 TensorRT 引擎的日志信息
// ============================================================================
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

// ============================================================================
// TRTInfer::Impl 类定义 - 使用 Pimpl 模式隐藏实现细节
// 封装了 TensorRT 引擎加载、推理和动态形状管理的完整逻辑
// ============================================================================
class TRTInfer::Impl
{
public:
    // 构造函数：加载引擎文件并初始化所有资源
    Impl(const std::string &engine_path, TRTInfer *parent);
    // 析构函数：释放 CUDA 资源
    ~Impl();

    // 公共 API 的实现方法
    // 推理接口：接收 void* 指针类型的输入数据
    std::unordered_map<std::string, std::shared_ptr<char[]>> infer(
        const std::unordered_map<std::string, void *> &input_blob);
    // 推理接口：接收 cv::Mat 类型的输入数据（自动处理类型转换）
    std::unordered_map<std::string, cv::Mat> infer(
        const std::unordered_map<std::string, cv::Mat> &input_blob);
    // 动态设置输入形状（用于动态形状引擎）
    void setInputShape(const std::string &input_name, const std::vector<int> &shape);

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
    // 私有方法 - 初始化流程
    void load_engine(const std::string &engine_path); // 从文件加载 TensorRT 引擎
    void get_InputNames();                            // 获取所有输入张量名称和信息
    void get_OutputNames();                           // 获取所有输出张量名称和信息
    void get_bindings();                              // 为输入输出分配 CUDA 显存
    void get_OptimizationProfiles();                  // 解析动态形状的优化配置文件
    void set_OutputBlob();                            // 设置输出张量的显存地址
    // 动态内存分配：如果当前内存不足，重新分配更大的显存
    size_t allocateDynamicMemory(
        const std::string &name,
        const nvinfer1::Dims &dims,
        nvinfer1::DataType dtype,
        std::unordered_map<std::string, void *> &bindings,
        std::unordered_map<std::string, size_t> &max_sizes);

private:
    TRTInfer *parent_; // 指向父类的指针（用于回调）

    // TensorRT 核心对象
    std::unique_ptr<nvinfer1::IRuntime> runtime;          // 运行时，用于反序列化引擎
    std::unique_ptr<nvinfer1::ICudaEngine> engine;        // CUDA 引擎，包含优化后的网络结构
    std::unique_ptr<nvinfer1::IExecutionContext> context; // 执行上下文，用于实际推理
    cudaStream_t stream;                                  // CUDA 流，用于异步操作
    Logger logger;                                        // 日志记录器

    // 输出数据存储（主机端）
    std::unordered_map<std::string, std::shared_ptr<char[]>> output_blob_ptr;

    // 张量元数据
    std::vector<std::string> input_names, output_names;              // 输入输出张量名称列表
    std::unordered_map<std::string, size_t> input_size, output_size; // 张量字节大小
    std::unordered_map<std::string, std::vector<int>> output_shape;  // 输出张量形状
    cv::Size size;                                                   // 图像尺寸（用于某些特定模型）

    // 动态形状支持相关
    std::unordered_map<std::string, std::vector<int>> current_input_shapes;                         // 当前使用的输入形状
    std::unordered_map<std::string, nvinfer1::Dims> input_min_dims, input_opt_dims, input_max_dims; // MIN/OPT/MAX 形状
    std::unordered_map<std::string, size_t> input_max_size, output_max_size;                        // 已分配的最大显存大小

    // 显存绑定（设备端）
    std::unordered_map<std::string, cv::Mat> input_Bindings, output_Bindings; // 未使用的绑定
    std::unordered_map<std::string, void *> inputBindings, outputBindings;    // CUDA 显存指针映射
};

// ============================================================================
// Stream operators - 用于方便地输出 TensorRT 类型信息
// ============================================================================
// 重载 << 操作符，用于输出 nvinfer1::Dims（张量形状）
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

// ============================================================================
// TRTInfer 公共接口 - 使用 Pimpl 模式将实现委托给 Impl 类
// Pimpl（Pointer to Implementation）模式隐藏实现细节，减少编译依赖
// ============================================================================
// 构造函数：创建 Impl 实例并加载引擎
TRTInfer::TRTInfer(const std::string &engine_path)
    : pImpl(std::make_unique<Impl>(engine_path, this))
{
}

// @brief 析构函数：使用默认实现（unique_ptr 自动释放 Impl）
TRTInfer::~TRTInfer() = default;

// 重载 () 运算符：void* 指针版本的推理接口
std::unordered_map<std::string, std::shared_ptr<char[]>> TRTInfer::operator()(
    const std::unordered_map<std::string, void *> &input_blob)
{
    return pImpl->infer(input_blob);
}

// 重载 () 运算符：cv::Mat 版本的推理接口
std::unordered_map<std::string, cv::Mat> TRTInfer::operator()(
    const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    return pImpl->infer(input_blob);
}

// 设置动态输入形状的公共接口
void TRTInfer::setInputShape(const std::string &input_name, const std::vector<int> &shape)
{
    pImpl->setInputShape(input_name, shape);
}

// 辅助函数：将 vector 转为 TensorShape
static TensorShape vectorToShape(const std::vector<int> &vec)
{
    TensorShape shape;
    if (vec.size() == 4)
    {
        shape.d = 0;
        shape.n = vec[0];
        shape.c = vec[1];
        shape.h = vec[2];
        shape.w = vec[3];
    } else {
                
        shape.n = vec[0];
        shape.d = vec[1];
        shape.c = vec[2];
        shape.h = vec[3];
        shape.w = vec[4];
    }
    return shape;
}

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
    return vectorToShape(pImpl->getInputShapeVec(name));
}

// 获取指定输出张量形状
TensorShape TRTInfer::getOutputShape(const std::string &name) const
{
    return vectorToShape(pImpl->getOutputShapeVec(name));
}

// ============================================================================
// TRTInfer::Impl 实现
// ============================================================================
TRTInfer::Impl::Impl(const std::string &engine_path, TRTInfer *parent)
    : parent_(parent), logger()
{
    // 按正确顺序初始化引擎和相关资源
    load_engine(engine_path);   // 1. 加载引擎文件
    get_InputNames();           // 2. 获取输入张量信息
    get_OutputNames();          // 3. 获取输出张量信息
    get_OptimizationProfiles(); // 4. 解析动态形状配置
    get_bindings();             // 5. 分配 CUDA 显存
    set_OutputBlob();           // 6. 设置输出地址
    cudaStreamCreate(&stream);  // 7. 创建 CUDA 流
}

// 析构函数：释放所有 CUDA 资源
TRTInfer::Impl::~Impl()
{
    cudaStreamDestroy(stream); // 销毁 CUDA 流

    // 释放所有输入输出张量的 CUDA 显存
    for (auto &data : inputBindings)
        utility::safeCudaFree(data.second);
    for (auto &data : outputBindings)
        utility::safeCudaFree(data.second);
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
    initLibNvInferPlugins(&logger, "");
    // 反序列化引擎
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize));
    if (!engine)
    {
        std::cerr << "Failed to create engine" << std::endl;
        throw std::runtime_error("Failed to create engine");
    }
    // 创建执行上下文（用于推理）
    context.reset(engine->createExecutionContext());
}

// 获取引擎中所有输入张量的信息（名称、形状、数据类型等）
void TRTInfer::Impl::get_InputNames()
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
                      << ",tensor shape : " << engine->getTensorShape(name)
                      << ",tensor type : " << engine->getTensorDataType(name)
                      << ",tensor format : " << engine->getTensorFormatDesc(name)
                      << std::endl;
            input_names.emplace_back(std::string(name));
            // 计算并存储张量的字节大小
            input_size[std::string(name)] = utility::getTensorbytes(engine->getTensorShape(name), engine->getTensorDataType(name));
        }
    }
}

// 获取引擎中所有输出张量的信息（名称、形状、数据类型等）
void TRTInfer::Impl::get_OutputNames()
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
                      << ",tensor shape : " << engine->getTensorShape(name)
                      << ",tensor type : " << engine->getTensorDataType(name)
                      << ",tensor format : " << engine->getTensorFormatDesc(name)
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

// 为所有输入输出张量分配 CUDA 显存
void TRTInfer::Impl::get_bindings()
{
    // 为每个输入张量分配显存
    for (int i = 0; i < input_names.size(); i++)
    {
        inputBindings[input_names[i]] = utility::safeCudaMalloc(input_size[input_names[i]]);
        input_max_size[input_names[i]] = input_size[input_names[i]];
    }
    // 为每个输出张量分配显存
    for (int i = 0; i < output_names.size(); i++)
    {
        outputBindings[output_names[i]] = utility::safeCudaMalloc(output_size[output_names[i]]);
        output_max_size[output_names[i]] = output_size[output_names[i]];
    }
}

// 解析优化配置文件（Optimization Profiles），用于支持动态形状输入
// 动态形状允许同一个引擎处理不同 batch size 或分辨率的输入
void TRTInfer::Impl::get_OptimizationProfiles()
{
    // 获取优化配置文件数量
    int numProfiles = engine->getNbOptimizationProfiles();
    std::cout << "Number of optimization profiles: " << numProfiles << std::endl;

    // 遍历所有 IO 张量
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        // 只处理输入类型的张量
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            std::string tensorName(name);
            nvinfer1::Dims shape = engine->getTensorShape(name);

            // 检查是否有动态维度（值为 -1 表示可变）
            bool hasDynamicDim = false;
            for (int j = 0; j < shape.nbDims; j++)
            {
                if (shape.d[j] == -1)
                {
                    hasDynamicDim = true;
                    break;
                }
            }

            // 如果有动态维度且有优化配置文件
            if (hasDynamicDim && numProfiles > 0)
            {
                // 定义三个优化配置选择器：最小值、最优值、最大值
                nvinfer1::OptProfileSelector selectors[] = {
                    nvinfer1::OptProfileSelector::kMIN,
                    nvinfer1::OptProfileSelector::kOPT,
                    nvinfer1::OptProfileSelector::kMAX};

                // 获取该张量的 MIN/OPT/MAX 形状
                input_min_dims[tensorName] = engine->getProfileShape(name, 0, selectors[0]);
                input_opt_dims[tensorName] = engine->getProfileShape(name, 0, selectors[1]);
                input_max_dims[tensorName] = engine->getProfileShape(name, 0, selectors[2]);

                // 初始化当前输入形状为 MAX（后续可通过 setInputShape 动态调整）
                current_input_shapes[tensorName] = std::vector<int>();
                for (int j = 0; j < input_max_dims[tensorName].nbDims; j++)
                {
                    current_input_shapes[tensorName].push_back(input_max_dims[tensorName].d[j]);
                }

                // 打印动态形状信息
                std::cout << "Dynamic input tensor: " << tensorName << std::endl;
                std::cout << "  Min shape: " << input_min_dims[tensorName] << std::endl;
                std::cout << "  Opt shape: " << input_opt_dims[tensorName] << std::endl;
                std::cout << "  Max shape: " << input_max_dims[tensorName] << std::endl;

                // 根据 MAX 形状计算所需显存大小
                input_size[tensorName] = utility::getTensorbytes(input_max_dims[tensorName],
                                                                 engine->getTensorDataType(name));
            }
            else
            {
                // 固定形状：直接使用引擎中的形状
                current_input_shapes[tensorName] = std::vector<int>();
                for (int j = 0; j < shape.nbDims; j++)
                {
                    current_input_shapes[tensorName].push_back(shape.d[j]);
                }
            }
        }
    }
}

// 动态设置输入张量的形状（用于动态形状推理）
// 需要在每次推理前调用，确保输入尺寸在 MIN-MAX 范围内
void TRTInfer::Impl::setInputShape(const std::string &input_name, const std::vector<int> &shape)
{
    // 检查输入张量是否存在
    if (current_input_shapes.find(input_name) == current_input_shapes.end())
    {
        std::cerr << "Input tensor '" << input_name << "' not found" << std::endl;
        throw std::runtime_error("Input tensor not found");
    }

    // 构造 TensorRT 的 Dims 结构
    nvinfer1::Dims dims;
    dims.nbDims = shape.size();
    for (size_t i = 0; i < shape.size(); i++)
    {
        dims.d[i] = shape[i];
    }

    // 设置执行上下文的输入形状（如果形状超出范围会返回 false）
    if (!context->setInputShape(input_name.c_str(), dims))
    {
        std::cerr << "Failed to set input shape for '" << input_name << "'" << std::endl;
        throw std::runtime_error("Failed to set input shape");
    }

    // 更新当前形状记录
    current_input_shapes[input_name] = shape;
    // 如果新形状需要更大内存，重新分配显存
    allocateDynamicMemory(input_name, dims, engine->getTensorDataType(input_name.c_str()),
                          inputBindings, input_max_size);
}

// 动态内存分配函数：检查当前显存是否足够，不足则重新分配
// 参数：name-张量名, dims-形状, dtype-数据类型, bindings-显存绑定, max_sizes-已分配的最大尺寸
size_t TRTInfer::Impl::allocateDynamicMemory(const std::string &name, const nvinfer1::Dims &dims,
                                             nvinfer1::DataType dtype,
                                             std::unordered_map<std::string, void *> &bindings,
                                             std::unordered_map<std::string, size_t> &max_sizes)
{
    // 计算所需内存大小
    size_t required_size = utility::getTensorbytes(dims, dtype);

    // 如果当前内存不足，需要重新分配
    if (required_size > max_sizes[name])
    {
        // 先释放旧内存
        if (bindings.find(name) != bindings.end() && bindings[name] != nullptr)
        {
            utility::safeCudaFree(bindings[name]);
        }

        // 分配新内存
        bindings[name] = utility::safeCudaMalloc(required_size);
        max_sizes[name] = required_size;

        std::cout << "Reallocated memory for '" << name << "': " << required_size << " bytes" << std::endl;
    }

    return required_size;
}

// 设置输出张量的显存地址，并在主机端预分配输出数据缓冲区
void TRTInfer::Impl::set_OutputBlob()
{
    // 将输出张量的显存地址绑定到执行上下文
    for (int i = 0; i < output_names.size(); i++)
    {
        context->setOutputTensorAddress(output_names[i].c_str(), outputBindings[output_names[i]]);
    }

    // 在主机端预分配输出数据缓冲区（用于接收推理结果）
    for (const auto &name : output_names)
    {
        size_t datasize = output_size[name];
        output_blob_ptr[name] = std::shared_ptr<char[]>(new char[datasize]);
    }
}

// 推理函数：接收 void* 指针类型的输入数据，返回原始字节指针的输出
// 流程：输入 H2D → 推理 → 输出 D2H
std::unordered_map<std::string, std::shared_ptr<char[]>> TRTInfer::Impl::infer(
    const std::unordered_map<std::string, void *> &input_blob)
{
    // ===== 第一阶段：将输入数据从主机复制到设备 =====
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        void *cpu_ptr = input_data.second;
        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;
            // 根据当前输入形状计算数据大小
            nvinfer1::Dims current_dims;
            current_dims.nbDims = current_input_shapes[key].size();
            for (size_t i = 0; i < current_input_shapes[key].size(); i++)
            {
                current_dims.d[i] = current_input_shapes[key][i];
            }
            size_t data_size = utility::getTensorbytes(current_dims, engine->getTensorDataType(key.c_str()));

            // 异步复制：主机 → 设备
            cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr, data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            // 设置输入张量的显存地址
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }

    // ===== 第二阶段：执行推理 =====
    context->enqueueV3(stream);

    // ===== 第三阶段：将输出数据从设备复制到主机 =====
    for (const auto &names : output_names)
    {
        // 获取输出张量的实际形状（动态形状下可能变化）
        nvinfer1::Dims out_shape = context->getTensorShape(names.c_str());
        size_t actual_size = utility::getTensorbytes(out_shape, engine->getTensorDataType(names.c_str()));

        // 如果输出显存不足，重新分配
        if (actual_size > output_max_size[names])
        {
            utility::safeCudaFree(outputBindings[names]);
            outputBindings[names] = utility::safeCudaMalloc(actual_size);
            output_max_size[names] = actual_size;
            context->setOutputTensorAddress(names.c_str(), outputBindings[names]);
        }

        // 如果主机端缓冲区不足，重新分配
        if (actual_size > output_size[names])
        {
            output_blob_ptr[names] = std::shared_ptr<char[]>(new char[actual_size]);
            output_size[names] = actual_size;
        }

        // 更新输出形状记录
        output_shape[names].clear();
        for (int i = 0; i < out_shape.nbDims; i++)
        {
            output_shape[names].push_back(out_shape.d[i]);
        }

        // 异步复制：设备 → 主机
        void *ptr = static_cast<void *>(output_blob_ptr[names].get());
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            cudaError_t err = cudaMemcpyAsync(ptr, iter->second, actual_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // 等待所有 CUDA 操作完成
    cudaStreamSynchronize(stream);
    return output_blob_ptr;
}

// 推理函数：接收 cv::Mat 类型的输入数据，自动处理类型转换，返回 cv::Mat 格式的输出
// 流程：类型转换 → 输入 H2D → 推理 → 输出 D2H → 包装为 cv::Mat
std::unordered_map<std::string, cv::Mat> TRTInfer::Impl::infer(
    const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    // ===== 第一阶段：处理输入数据并复制到设备 =====
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        cv::Mat cpu_ptr = input_data.second;

        // 如果 cv::Mat 的数据类型与引擎要求不匹配，自动转换
        if (utility::typeCv2Rt(cpu_ptr.type()) != engine->getTensorDataType(key.c_str()))
            cpu_ptr.convertTo(cpu_ptr, utility::typeRt2Cv(engine->getTensorDataType(key.c_str())));
        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;
            // 根据当前输入形状计算数据大小
            nvinfer1::Dims current_dims;
            current_dims.nbDims = current_input_shapes[key].size();
            for (size_t i = 0; i < current_input_shapes[key].size(); i++)
            {
                current_dims.d[i] = current_input_shapes[key][i];
            }
            size_t data_size = utility::getTensorbytes(current_dims, engine->getTensorDataType(key.c_str()));

            // 异步复制：主机 → 设备
            cudaError_t err = cudaMemcpyAsync(cuda_ptr, static_cast<void *>(cpu_ptr.data), data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            // 设置输入张量的显存地址
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }

    // ===== 第二阶段：执行推理 =====
    context->enqueueV3(stream);

    // ===== 第三阶段：将输出数据从设备复制到主机 =====
    for (const auto &names : output_names)
    {
        // 获取输出张量的实际形状（动态形状下可能变化）
        nvinfer1::Dims out_shape = context->getTensorShape(names.c_str());
        size_t actual_size = utility::getTensorbytes(out_shape, engine->getTensorDataType(names.c_str()));

        // 如果输出显存不足，重新分配
        if (actual_size > output_max_size[names])
        {
            utility::safeCudaFree(outputBindings[names]);
            outputBindings[names] = utility::safeCudaMalloc(actual_size);
            output_max_size[names] = actual_size;
            context->setOutputTensorAddress(names.c_str(), outputBindings[names]);
        }

        // 如果主机端缓冲区不足，重新分配
        if (actual_size > output_size[names])
        {
            output_blob_ptr[names] = std::shared_ptr<char[]>(new char[actual_size]);
            output_size[names] = actual_size;
        }

        // 更新输出形状记录
        output_shape[names].clear();
        for (int i = 0; i < out_shape.nbDims; i++)
        {
            output_shape[names].push_back(out_shape.d[i]);
        }

        // 异步复制：设备 → 主机
        void *ptr = static_cast<void *>(output_blob_ptr[names].get());
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            cudaError_t err = cudaMemcpyAsync(ptr, iter->second, actual_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // 等待所有 CUDA 操作完成
    cudaStreamSynchronize(stream);

    // ===== 第四阶段：将输出数据包装为 cv::Mat 格式返回 =====
    std::unordered_map<std::string, cv::Mat> output_blob;
    for (const auto &names : output_names)
    {
        // 使用共享的原始指针创建 cv::Mat（不复制数据），然后 clone 创建独立副本
        cv::Mat temp(
            output_shape[names].size(),
            output_shape[names].data(),
            utility::typeRt2Cv(engine->getTensorDataType(names.c_str())),
            output_blob_ptr[names].get());
        output_blob[names] = temp.clone();
    }

    return output_blob;
}
