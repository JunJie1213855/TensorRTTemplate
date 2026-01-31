#include "TRTinfer.h"
#include "inference_config.h"
#include <stdexcept>
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

// logger
void Logger::log(Severity severity, const char *msg) noexcept
{
    if (severity != Severity::kINFO)
    {
        std::cout << msg << std::endl;
    }
}

// TRTInfer
TRTInfer::TRTInfer(const std::string &engine_path, int num_streams, bool enable_async, bool use_cvMat)
    : logger(), enable_async_(enable_async), use_cvMat_(use_cvMat) {

    load_engine(engine_path);

    get_InputNames();

    get_OutputNames();

    if (num_streams <= 0) {
        num_streams = 1;
    }
    stream_pool_ = std::make_shared<inference::StreamPool>(num_streams);
    memory_pool_ = std::make_shared<inference::MemoryPool>(input_size, output_size, num_streams);

    if (enable_async) {
        if (use_cvMat) {
            async_infer_mat_.reset(new inference::AsyncInfer<std::unordered_map<std::string, cv::Mat>>(
                context.get(), engine.get(), stream_pool_, memory_pool_,
                input_names, output_names, input_size, output_size, output_shape));
            std::cout << "Async inference enabled with " << num_streams << " streams (cv::Mat mode)" << std::endl;
        } else {
            async_infer_ptr_.reset(new inference::AsyncInfer<std::unordered_map<std::string, std::shared_ptr<char[]>>>(
                context.get(), engine.get(), stream_pool_, memory_pool_,
                input_names, output_names, input_size, output_size, {}));
            std::cout << "Async inference enabled with " << num_streams << " streams (shared_ptr mode)" << std::endl;
        }
    } else {
        set_OutputBlob();
        stream = stream_pool_->acquire()->get();
        std::cout << "Sync inference enabled with " << num_streams << " stream" << std::endl;
    }
}
TRTInfer::~TRTInfer()
{
    // MemoryPool and StreamPool handle cleanup automatically
    // No need to delete output_blob_ptr - smart pointers handle this automatically
}

std::unordered_map<std::string, std::shared_ptr<char[]>> TRTInfer::operator()(const std::unordered_map<std::string, void *> &input_blob)
{
    return infer(input_blob);
}

std::unordered_map<std::string, cv::Mat> TRTInfer::operator()(const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    if (enable_async_ && async_infer_mat_) {
        auto future = async_infer_mat_->infer_async(input_blob);
        return future.get();
    }
    return infer(input_blob);
}
void TRTInfer::load_engine(const std::string &engine_path)
{
    // read engine weights
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good())
    {
        file.close();
        std::cerr << "Error reading engine file" << std::endl;
        throw std::runtime_error("Error reading engine file");
    }
    file.seekg(0, file.end);
    const size_t fsize = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(fsize);
    file.read(engineData.data(), fsize);
    file.close();

    // runtime
    runtime.reset(nvinfer1::createInferRuntime(logger));

    if (!runtime)
    {
        std::cerr << "Failed to create runtime" << std::endl;
        throw std::runtime_error("Failed to create runtime");
    }

    // init plugins
    initLibNvInferPlugins(&logger, "");
    // engine
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), fsize));
    if (!engine)
    {
        std::cerr << "Failed to create engine" << std::endl;
        throw std::runtime_error("Failed to create engine");
    }
    context.reset(engine->createExecutionContext());
}

void TRTInfer::get_InputNames()
{
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            std::cout << "input tensor name : " << name
                      << ",tensor shape : " << engine->getTensorShape(name)
                      << ",tensor type : " << engine->getTensorDataType(name)
                      << ",tensor format : " << engine->getTensorFormatDesc(name)
                      << std::endl;
            input_names.emplace_back(std::string(name));
            input_size[std::string(name)] = utility::getTensorbytes(engine->getTensorShape(name), engine->getTensorDataType(name));
        }
    }
}

void TRTInfer::get_OutputNames()
{
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            // first
            std::cout << "output tensor name : " << name
                      << ",tensor shape : " << engine->getTensorShape(name)
                      << ",tensor type : " << engine->getTensorDataType(name)
                      << ",tensor format : " << engine->getTensorFormatDesc(name)
                      << std::endl;
            // second
            output_names.emplace_back(std::string(name));
            output_size[std::string(name)] = utility::getTensorbytes(engine->getTensorShape(name), engine->getTensorDataType(name));
            // third
            // Convert TensorRT dims to OpenCV dims
            nvinfer1::Dims dims = engine->getTensorShape(name);
            std::vector<int> dim;
            dim.reserve(dims.nbDims);
            for (int i = 0; i < dims.nbDims; i++)
                dim.emplace_back(dims.d[i]);
            // Fill type
            output_shape[std::string(name)] = dim;
        }
    }
}

void TRTInfer::set_OutputBlob()
{
    // allocate output memory for cpu, input memory is from user or outside
    for (const auto &name : output_names)
    {
        // allocate memory using shared_ptr
        size_t datasize = output_size[name];
        output_blob_ptr[name] = std::shared_ptr<char[]>(new char[datasize]);
    }
}

std::unordered_map<std::string, std::shared_ptr<char[]>> TRTInfer::infer(const std::unordered_map<std::string, void *> &input_blob)
{
    // input copy
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        void *cpu_ptr = input_data.second;
        void *cuda_ptr = memory_pool_->get_input_binding(key, 0);
        if (cuda_ptr)
        {
            size_t data_size = input_size[key];

            cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr, data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            // set the input tensor
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }

    // set output tensor addresses
    for (const auto &name : output_names)
    {
        void *cuda_ptr = memory_pool_->get_output_binding(name, 0);
        if (cuda_ptr)
        {
            context->setOutputTensorAddress(name.c_str(), cuda_ptr);
        }
    }

    // async execute
    context->enqueueV3(stream);

    // copy the gpu data to cpu data
    for (const auto &names : output_names)
    {
        size_t datasize = output_size[names];
        // Get raw pointer from shared_ptr for cudaMemcpy
        void* ptr = static_cast<void*>(output_blob_ptr[names].get());
        void *cuda_ptr = memory_pool_->get_output_binding(names, 0);
        if (cuda_ptr)
        {
            cudaError_t err = cudaMemcpyAsync(ptr, cuda_ptr, datasize, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // waiting for the stream
    cudaStreamSynchronize(stream);

    return output_blob_ptr; // Smart pointers handle memory automatically
}

std::unordered_map<std::string, cv::Mat> TRTInfer::infer(const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    // input copy
    for (const auto &input_data : input_blob)
    {
        const std::string &key = input_data.first;
        cv::Mat cpu_ptr = input_data.second;

        // Type check, may conversion
        if (utility::typeCv2Rt(cpu_ptr.type()) != engine->getTensorDataType(key.c_str()))
            cpu_ptr.convertTo(cpu_ptr, utility::typeRt2Cv(engine->getTensorDataType(key.c_str())));
        void *cuda_ptr = memory_pool_->get_input_binding(key, 0);
        if (cuda_ptr)
        {
            size_t data_size = input_size[key];

            cudaError_t err = cudaMemcpyAsync(cuda_ptr, static_cast<void*>(cpu_ptr.data), data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            // set input tensor
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
        }
    }

    // set output tensor addresses
    for (const auto &name : output_names)
    {
        void *cuda_ptr = memory_pool_->get_output_binding(name, 0);
        if (cuda_ptr)
        {
            context->setOutputTensorAddress(name.c_str(), cuda_ptr);
        }
    }

    // async execute
    context->enqueueV3(stream);

    // copy gpu data to cpu data
    for (const auto &names : output_names)
    {
        size_t datasize = output_size[names];
        // Get raw pointer from shared_ptr for cudaMemcpy
        void* ptr = static_cast<void*>(output_blob_ptr[names].get());
        void *cuda_ptr = memory_pool_->get_output_binding(names, 0);
        if (cuda_ptr)
        {
            cudaError_t err = cudaMemcpyAsync(ptr, cuda_ptr, datasize, cudaMemcpyDeviceToHost, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // waiting for stream
    cudaStreamSynchronize(stream);

    // Convert output data to cv::Mat
    std::unordered_map<std::string, cv::Mat> output_blob;
    for (const auto &name : output_names)
    {
        nvinfer1::DataType dtype = engine->getTensorDataType(name.c_str());
        const auto &shape = output_shape[name];
        output_blob[name] = cv::Mat(shape.size(), shape.data(), utility::typeRt2Cv(dtype), output_blob_ptr[name].get()).clone();
    }

    return output_blob;
}

std::future<std::unordered_map<std::string, std::shared_ptr<char[]>>>
TRTInfer::infer_async(const std::unordered_map<std::string, void *> &input_blob)
{
    if (!enable_async_ || !async_infer_ptr_) {
        throw std::runtime_error("Async inference not enabled or use_cvMat is set to true. Use constructor with enable_async=true and use_cvMat=false.");
    }
    return async_infer_ptr_->infer_async(input_blob);
}

std::future<std::unordered_map<std::string, cv::Mat>>
TRTInfer::infer_async(const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    if (!enable_async_ || !async_infer_mat_) {
        throw std::runtime_error("Async inference not enabled or use_cvMat is set to false. Use constructor with enable_async=true and use_cvMat=true.");
    }
    return async_infer_mat_->infer_async(input_blob);
}

void TRTInfer::infer_with_callback(const std::unordered_map<std::string, void *> &input_blob,
                                   std::function<void(const std::unordered_map<std::string, std::shared_ptr<char[]>>&)> callback)
{
    if (!enable_async_ || !async_infer_ptr_) {
        throw std::runtime_error("Async inference not enabled or use_cvMat is set to true. Use constructor with enable_async=true and use_cvMat=false.");
    }
    async_infer_ptr_->infer_with_callback(input_blob, callback);
}

void TRTInfer::infer_with_callback(const std::unordered_map<std::string, cv::Mat> &input_blob,
                                   std::function<void(const std::unordered_map<std::string, cv::Mat>&)> callback)
{
    if (!enable_async_ || !async_infer_mat_) {
        throw std::runtime_error("Async inference not enabled or use_cvMat is set to false. Use constructor with enable_async=true and use_cvMat=true.");
    }
    async_infer_mat_->infer_with_callback(input_blob, callback);
}

void TRTInfer::wait_all()
{
    if (enable_async_ && stream_pool_) {
        stream_pool_->synchronize_all();
    }
}

int TRTInfer::num_streams() const
{
    if (enable_async_ && stream_pool_) {
        return stream_pool_->num_streams();
    }
    return 1;
}
