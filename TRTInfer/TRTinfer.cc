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
// Logger 类 - TensorRT 日志记录器
// 接收 TensorRT 引擎日志，过滤 INFO 级别，只输出 WARNING 及以上
// ============================================================================
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

// ============================================================================
// TRTInfer::Impl 实现类 - Pimpl 模式隐藏 TensorRT 实现细节
// ============================================================================
class TRTInfer::Impl
{
public:
    Impl(const std::string &engine_path, TRTInfer *parent);
    ~Impl();

    // 推理接口
    std::unordered_map<std::string, std::shared_ptr<char[]>> infer(
        const std::unordered_map<std::string, void *> &input_blob);
    std::unordered_map<std::string, cv::Mat> infer(
        const std::unordered_map<std::string, cv::Mat> &input_blob);

    // 动态形状设置
    void setInputShape(const std::string &input_name, const std::vector<int> &shape);

    // 查询接口
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

    void init();

private:
    // 初始化流程
    void load_engine(const std::string &engine_path);
    void get_InputNames();
    void get_OutputNames();
    void get_bindings();
    void get_OptimizationProfiles();
    void set_OutputBlob();

    // 动态内存分配
    size_t allocateDynamicMemory(
        const std::string &name,
        const nvinfer1::Dims &dims,
        nvinfer1::DataType dtype,
        std::unordered_map<std::string, void *> &bindings,
        std::unordered_map<std::string, size_t> &max_sizes);

private:
    TRTInfer *parent_;

    std::string engine_path_;

    // TensorRT 核心对象
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    Logger logger;

    // 输出数据（主机端）
    std::unordered_map<std::string, std::shared_ptr<char[]>> output_blob_ptr;

    // 张量元数据
    std::vector<std::string> input_names, output_names;
    std::unordered_map<std::string, size_t> input_size, output_size;
    std::unordered_map<std::string, std::vector<int>> output_shape;
    cv::Size size;

    // 动态形状支持
    std::unordered_map<std::string, std::vector<int>> current_input_shapes;
    std::unordered_map<std::string, nvinfer1::Dims> input_min_dims, input_opt_dims, input_max_dims;
    std::unordered_map<std::string, size_t> input_max_size, output_max_size;

    // 显存绑定（设备端）
    std::unordered_map<std::string, cv::Mat> input_Bindings, output_Bindings;
    std::unordered_map<std::string, void *> inputBindings, outputBindings;
};

// ============================================================================
// 流操作符重载 - 方便输出 TensorRT 类型信息
// ============================================================================
std::ostream &operator<<(std::ostream &cout, const nvinfer1::Dims &dim)
{
    for (int i = 0; i < dim.nbDims; i++)
    {
        if (i < dim.nbDims - 1)
            cout << dim.d[i] << " X ";
        else
            cout << dim.d[i];
    }
    return cout;
}

std::ostream &operator<<(std::ostream &cout, const nvinfer1::DataType &type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBF16: cout << "kBF16"; break;
    case nvinfer1::DataType::kBOOL: cout << "kBOOL"; break;
    case nvinfer1::DataType::kFLOAT: cout << "kFLOAT"; break;
    case nvinfer1::DataType::kFP8: cout << "kFP8"; break;
    case nvinfer1::DataType::kHALF: cout << "kHALF"; break;
    case nvinfer1::DataType::kINT32: cout << "kINT32"; break;
    case nvinfer1::DataType::kINT64: cout << "kINT64"; break;
    case nvinfer1::DataType::kINT8: cout << "kINT8"; break;
    case nvinfer1::DataType::kUINT8: cout << "kUINT8"; break;
    default: break;
    }
    return cout;
}

// ============================================================================
// TRTInfer 公共接口实现
// ============================================================================

/**
 * @brief 工厂方法：创建 TRTInfer 实例
 * @param engine_path 引擎文件路径
 * @return 智能指针管理的实例
 */
std::shared_ptr<TRTInfer> TRTInfer::create(const std::string &engine_path)
{
    auto model = std::shared_ptr<TRTInfer>(new TRTInfer(engine_path));
    model->Init();
    return model;
}

/**
 * @brief 初始化引擎和资源
 */
void TRTInfer::Init()
{
    pImpl->init();
}

// 构造函数（私有）
TRTInfer::TRTInfer(const std::string &engine_path)
    : pImpl(std::make_unique<Impl>(engine_path, this))
{
}

// 析构函数
TRTInfer::~TRTInfer() = default;

/**
 * @brief 同步推理 (void* 版本)
 * @param input_blob 输入张量 map，值为 GPU 指针
 * @return 输出张量 map，值为原始指针
 */
std::unordered_map<std::string, std::shared_ptr<char[]>> TRTInfer::operator()(
    const std::unordered_map<std::string, void *> &input_blob)
{
    return pImpl->infer(input_blob);
}

/**
 * @brief 同步推理 (cv::Mat 版本)
 * @param input_blob 输入张量 map，值为 cv::Mat
 * @return 输出张量 map，值为 cv::Mat
 */
std::unordered_map<std::string, cv::Mat> TRTInfer::operator()(
    const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    return pImpl->infer(input_blob);
}

/**
 * @brief 设置动态输入形状
 * @param input_name 输入张量名称
 * @param shape 形状向量
 */
void TRTInfer::setInputShape(const std::string &input_name, const std::vector<int> &shape)
{
    pImpl->setInputShape(input_name, shape);
}

// 辅助函数：vector 转 TensorShape
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
    }
    else
    {
        shape.n = vec[0];
        shape.d = vec[1];
        shape.c = vec[2];
        shape.h = vec[3];
        shape.w = vec[4];
    }
    return shape;
}

/**
 * @brief 获取所有输入张量名称
 */
std::vector<std::string> TRTInfer::getInputNames() const
{
    return pImpl->getInputNames();
}

/**
 * @brief 获取所有输出张量名称
 */
std::vector<std::string> TRTInfer::getOutputNames() const
{
    return pImpl->getOutputNames();
}

/**
 * @brief 获取指定输入张量形状
 */
TensorShape TRTInfer::getInputShape(const std::string &name) const
{
    return vectorToShape(pImpl->getInputShapeVec(name));
}

/**
 * @brief 获取指定输出张量形状
 */
TensorShape TRTInfer::getOutputShape(const std::string &name) const
{
    return vectorToShape(pImpl->getOutputShapeVec(name));
}

// ============================================================================
// TRTInfer::Impl 实现
// ============================================================================

TRTInfer::Impl::Impl(const std::string &engine_path, TRTInfer *parent)
    : engine_path_(engine_path), parent_(parent), logger()
{
}

/**
 * @brief 初始化引擎和资源（按正确顺序）
 */
void TRTInfer::Impl::init()
{
    load_engine(engine_path_);
    get_InputNames();
    get_OutputNames();
    get_OptimizationProfiles();
    get_bindings();
    set_OutputBlob();
    cudaStreamCreate(&stream);
}

// 析构函数：释放 CUDA 资源
TRTInfer::Impl::~Impl()
{
    cudaStreamDestroy(stream);

    for (auto &data : inputBindings)
        utility::safeCudaFree(data.second);
    for (auto &data : outputBindings)
        utility::safeCudaFree(data.second);
}

// 加载 TensorRT 引擎
void TRTInfer::Impl::load_engine(const std::string &engine_path)
{
    // 读取引擎文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        file.close();
        throw std::runtime_error("Error reading engine file: " + engine_path);
    }

    file.seekg(0, file.end);
    const size_t fsize = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    file.close();

    // 创建运行时
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime)
        throw std::runtime_error("Failed to create TensorRT runtime");

    // 初始化插件
    initLibNvInferPlugins(&logger, "");

    // 反序列化引擎
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize));
    if (!engine)
        throw std::runtime_error("Failed to deserialize TensorRT engine");

    // 创建执行上下文
    context.reset(engine->createExecutionContext());
}

// 获取输入张量信息
void TRTInfer::Impl::get_InputNames()
{
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            std::cout << "input tensor name : " << name
                      << ", tensor shape : " << engine->getTensorShape(name)
                      << ", tensor type : " << engine->getTensorDataType(name)
                      << ", tensor format : " << engine->getTensorFormatDesc(name)
                      << std::endl;
            input_names.emplace_back(std::string(name));
            input_size[std::string(name)] = utility::getTensorbytes(
                engine->getTensorShape(name), engine->getTensorDataType(name));
        }
    }
}

// 获取输出张量信息
void TRTInfer::Impl::get_OutputNames()
{
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            std::cout << "output tensor name : " << name
                      << ", tensor shape : " << engine->getTensorShape(name)
                      << ", tensor type : " << engine->getTensorDataType(name)
                      << ", tensor format : " << engine->getTensorFormatDesc(name)
                      << std::endl;
            output_names.emplace_back(std::string(name));
            output_size[std::string(name)] = utility::getTensorbytes(
                engine->getTensorShape(name), engine->getTensorDataType(name));

            // 存储输出形状
            nvinfer1::Dims dims = engine->getTensorShape(name);
            std::vector<int> dim;
            dim.reserve(dims.nbDims);
            for (int i = 0; i < dims.nbDims; i++)
                dim.emplace_back(dims.d[i]);
            output_shape[std::string(name)] = dim;
        }
    }
}

// 分配 CUDA 显存
void TRTInfer::Impl::get_bindings()
{
    for (int i = 0; i < input_names.size(); i++)
    {
        inputBindings[input_names[i]] = utility::safeCudaMalloc(input_size[input_names[i]]);
        input_max_size[input_names[i]] = input_size[input_names[i]];
    }
    for (int i = 0; i < output_names.size(); i++)
    {
        outputBindings[output_names[i]] = utility::safeCudaMalloc(output_size[output_names[i]]);
        output_max_size[output_names[i]] = output_size[output_names[i]];
    }
}

// 解析优化配置文件（支持动态形状）
void TRTInfer::Impl::get_OptimizationProfiles()
{
    int numProfiles = engine->getNbOptimizationProfiles();
    std::cout << "Number of optimization profiles: " << numProfiles << std::endl;

    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            std::string tensorName(name);
            nvinfer1::Dims shape = engine->getTensorShape(name);

            // 检查是否有动态维度
            bool hasDynamicDim = false;
            for (int j = 0; j < shape.nbDims; j++)
            {
                if (shape.d[j] == -1)
                {
                    hasDynamicDim = true;
                    break;
                }
            }

            if (hasDynamicDim && numProfiles > 0)
            {
                // 获取 MIN/OPT/MAX 形状
                input_min_dims[tensorName] = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMIN);
                input_opt_dims[tensorName] = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kOPT);
                input_max_dims[tensorName] = engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);

                // 初始化为 MAX 形状
                current_input_shapes[tensorName] = std::vector<int>();
                for (int j = 0; j < input_max_dims[tensorName].nbDims; j++)
                    current_input_shapes[tensorName].push_back(input_max_dims[tensorName].d[j]);

                // 打印动态形状信息
                std::cout << "Dynamic input tensor: " << tensorName << std::endl;
                std::cout << "  Min shape: " << input_min_dims[tensorName] << std::endl;
                std::cout << "  Opt shape: " << input_opt_dims[tensorName] << std::endl;
                std::cout << "  Max shape: " << input_max_dims[tensorName] << std::endl;

                // 按 MAX 形状分配显存
                input_size[tensorName] = utility::getTensorbytes(
                    input_max_dims[tensorName], engine->getTensorDataType(name));
            }
            else
            {
                // 固定形状
                current_input_shapes[tensorName] = std::vector<int>();
                for (int j = 0; j < shape.nbDims; j++)
                    current_input_shapes[tensorName].push_back(shape.d[j]);
            }
        }
    }
}

/**
 * @brief 设置输入形状（用于动态形状推理）
 */
void TRTInfer::Impl::setInputShape(const std::string &input_name, const std::vector<int> &shape)
{
    if (current_input_shapes.find(input_name) == current_input_shapes.end())
        throw std::runtime_error("Input tensor not found: " + input_name);

    // 构造 Dims
    nvinfer1::Dims dims;
    dims.nbDims = shape.size();
    for (size_t i = 0; i < shape.size(); i++)
        dims.d[i] = shape[i];

    // 设置形状
    if (!context->setInputShape(input_name.c_str(), dims))
        throw std::runtime_error("Failed to set input shape for: " + input_name);

    current_input_shapes[input_name] = shape;

    // 如需更大内存则重新分配
    allocateDynamicMemory(input_name, dims, engine->getTensorDataType(input_name.c_str()),
                          inputBindings, input_max_size);
}

// 动态内存分配
size_t TRTInfer::Impl::allocateDynamicMemory(const std::string &name, const nvinfer1::Dims &dims,
                                             nvinfer1::DataType dtype,
                                             std::unordered_map<std::string, void *> &bindings,
                                             std::unordered_map<std::string, size_t> &max_sizes)
{
    size_t required_size = utility::getTensorbytes(dims, dtype);

    if (required_size > max_sizes[name])
    {
        if (bindings.find(name) != bindings.end() && bindings[name] != nullptr)
            utility::safeCudaFree(bindings[name]);

        bindings[name] = utility::safeCudaMalloc(required_size);
        max_sizes[name] = required_size;
        std::cout << "Reallocated memory for '" << name << "': " << required_size << " bytes" << std::endl;
    }

    return required_size;
}

// 设置输出张量地址
void TRTInfer::Impl::set_OutputBlob()
{
    for (int i = 0; i < output_names.size(); i++)
        context->setOutputTensorAddress(output_names[i].c_str(), outputBindings[output_names[i]]);

    // 预分配主机端缓冲区
    for (const auto &name : output_names)
    {
        size_t datasize = output_size[name];
        output_blob_ptr[name] = std::shared_ptr<char[]>(new char[datasize]);
    }
}

// ============================================================================
// 推理实现
// ============================================================================

/**
 * @brief 推理实现 (void* 版本)
 * @return 输出张量 map
 */
std::unordered_map<std::string, std::shared_ptr<char[]>> TRTInfer::Impl::infer(
    const std::unordered_map<std::string, void *> &input_blob)
{
    // ===== 输入：Host -> Device =====
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        void *cpu_ptr = input_data.second;

        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;

            // 计算当前形状的数据大小
            nvinfer1::Dims current_dims;
            current_dims.nbDims = current_input_shapes[key].size();
            for (size_t i = 0; i < current_input_shapes[key].size(); i++)
                current_dims.d[i] = current_input_shapes[key][i];

            size_t data_size = utility::getTensorbytes(current_dims, engine->getTensorDataType(key.c_str()));

            // H2D 拷贝
            cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr, data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("CUDA memcpyAsync failed: ") + cudaGetErrorString(err));

            // 设置输入地址
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }

    // ===== 推理 =====
    context->enqueueV3(stream);

    // ===== 输出：Device -> Host =====
    for (const auto &names : output_names)
    {
        nvinfer1::Dims out_shape = context->getTensorShape(names.c_str());
        size_t actual_size = utility::getTensorbytes(out_shape, engine->getTensorDataType(names.c_str()));

        // 显存不足则重新分配
        if (actual_size > output_max_size[names])
        {
            utility::safeCudaFree(outputBindings[names]);
            outputBindings[names] = utility::safeCudaMalloc(actual_size);
            output_max_size[names] = actual_size;
            context->setOutputTensorAddress(names.c_str(), outputBindings[names]);
        }

        // 主机缓冲区不足则重新分配
        if (actual_size > output_size[names])
        {
            output_blob_ptr[names] = std::shared_ptr<char[]>(new char[actual_size]);
            output_size[names] = actual_size;
        }

        // 更新输出形状
        output_shape[names].clear();
        for (int i = 0; i < out_shape.nbDims; i++)
            output_shape[names].push_back(out_shape.d[i]);

        // D2H 拷贝
        void *ptr = static_cast<void *>(output_blob_ptr[names].get());
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            if (actual_size != output_size[names])
                throw std::runtime_error("Output buffer size insufficient for: " + names);

            cudaError_t err = cudaMemcpyAsync(ptr, iter->second, actual_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("CUDA memcpyAsync failed: ") + cudaGetErrorString(err));
        }
    }

    cudaStreamSynchronize(stream);
    return output_blob_ptr;
}

/**
 * @brief 推理实现 (cv::Mat 版本)
 * @return 输出张量 map (cv::Mat)
 */
std::unordered_map<std::string, cv::Mat> TRTInfer::Impl::infer(
    const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    // ===== 输入处理：Host -> Device =====
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        cv::Mat cpu_ptr = input_data.second;

        // 自动类型转换
        if (utility::typeCv2Rt(cpu_ptr.type()) != engine->getTensorDataType(key.c_str()))
            cpu_ptr.convertTo(cpu_ptr, utility::typeRt2Cv(engine->getTensorDataType(key.c_str())));

        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;

            // 计算大小
            nvinfer1::Dims current_dims;
            current_dims.nbDims = current_input_shapes[key].size();
            for (size_t i = 0; i < current_input_shapes[key].size(); i++)
                current_dims.d[i] = current_input_shapes[key][i];

            size_t data_size = utility::getTensorbytes(current_dims, engine->getTensorDataType(key.c_str()));

            // 检查数据大小匹配
            size_t mat_actual_size = cpu_ptr.total() * cpu_ptr.elemSize();
            if (data_size != mat_actual_size)
                throw std::runtime_error("Input tensor size mismatch for: " + key);

            // 检查连续性
            if (!cpu_ptr.isContinuous())
                std::cerr << "[WARNING] Input cv::Mat for '" << key << "' is not continuous" << std::endl;

            // H2D 拷贝
            cudaError_t err = cudaMemcpyAsync(cuda_ptr, static_cast<void *>(cpu_ptr.data),
                                               data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("CUDA memcpyAsync failed: ") + cudaGetErrorString(err));

            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }

    // ===== 推理 =====
    context->enqueueV3(stream);

    // ===== 输出处理：Device -> Host =====
    for (const auto &names : output_names)
    {
        nvinfer1::Dims out_shape = context->getTensorShape(names.c_str());
        size_t actual_size = utility::getTensorbytes(out_shape, engine->getTensorDataType(names.c_str()));

        // 显存不足则重新分配
        if (actual_size > output_max_size[names])
        {
            utility::safeCudaFree(outputBindings[names]);
            outputBindings[names] = utility::safeCudaMalloc(actual_size);
            output_max_size[names] = actual_size;
            context->setOutputTensorAddress(names.c_str(), outputBindings[names]);
        }

        // 主机缓冲区不足则重新分配
        if (actual_size > output_size[names])
        {
            output_blob_ptr[names] = std::shared_ptr<char[]>(new char[actual_size]);
            output_size[names] = actual_size;
        }

        // 更新输出形状
        output_shape[names].clear();
        for (int i = 0; i < out_shape.nbDims; i++)
            output_shape[names].push_back(out_shape.d[i]);

        // D2H 拷贝
        void *ptr = static_cast<void *>(output_blob_ptr[names].get());
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            if (actual_size != output_size[names])
                throw std::runtime_error("Output buffer size insufficient for: " + names);

            cudaError_t err = cudaMemcpyAsync(ptr, iter->second, actual_size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("CUDA memcpyAsync failed: ") + cudaGetErrorString(err));
        }
    }

    cudaStreamSynchronize(stream);

    // ===== 转换为 cv::Mat =====
    std::unordered_map<std::string, cv::Mat> output_blob;
    for (const auto &names : output_names)
    {
        cv::Mat temp(
            output_shape[names].size(),
            output_shape[names].data(),
            utility::typeRt2Cv(engine->getTensorDataType(names.c_str())),
            output_blob_ptr[names].get());
        output_blob[names] = temp.clone();
    }

    return output_blob;
}
