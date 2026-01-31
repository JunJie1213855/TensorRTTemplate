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
TRTInfer::TRTInfer(const std::string &engine_path, int num_streams, bool enable_async)
    : logger(), enable_async_(enable_async) {

    load_engine(engine_path);

    get_InputNames();

    get_OutputNames();

    if (enable_async) {
        if (num_streams <= 0) {
            num_streams = 4;
        }
        stream_pool_ = std::make_shared<inference::StreamPool>(num_streams);
        memory_pool_ = std::make_shared<inference::MemoryPool>(input_size, output_size, num_streams);

        async_infer_ptr_.reset(new inference::AsyncInfer<std::unordered_map<std::string, std::shared_ptr<char[]>>>(
            context.get(), engine.get(), stream_pool_, memory_pool_,
            input_names, output_names, input_size, output_size, {}));

        async_infer_mat_.reset(new inference::AsyncInfer<std::unordered_map<std::string, cv::Mat>>(
            context.get(), engine.get(), stream_pool_, memory_pool_,
            input_names, output_names, input_size, output_size, output_shape));

        std::cout << "Async inference enabled with " << num_streams << " streams" << std::endl;
    } else {
        get_bindings();
        set_OutputBlob();
        cudaStreamCreate(&stream);
    }
}
TRTInfer::~TRTInfer()
{
    // destory stream (only for sync mode)
    if (!enable_async_)
    {
        cudaStreamDestroy(stream);
    }

    // release cuda data
    for (auto &data : inputBindings)
        utility::safeCudaFree(data.second);
    for (auto &data : outputBindings)
        utility::safeCudaFree(data.second);

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

void TRTInfer::get_bindings()
{
    // allocate input memeory for cuda
    for (int i = 0; i < input_names.size(); i++)
    {
        inputBindings[input_names[i]] = utility::safeCudaMalloc(input_size[input_names[i]]);
    }
    // allocate output memeory for cuda
    for (int i = 0; i < output_names.size(); i++)
    {
        outputBindings[output_names[i]] = utility::safeCudaMalloc(output_size[output_names[i]]);
    }
}

void TRTInfer::set_OutputBlob()
{
    // output set is fixed, so we can set it here
    for (int i = 0; i < output_names.size(); i++)
    {
        context->setOutputTensorAddress(output_names[i].c_str(), outputBindings[output_names[i]]);
    }

    // allocate output memory for cpu, the input memory is from user or outside
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
        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;
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


    // async execute
    context->enqueueV3(stream);

    // copy the gpu data to cpu data
    for (const auto &names : output_names)
    {
        size_t datasize = output_size[names];
        // Get raw pointer from shared_ptr for cudaMemcpy
        void* ptr = static_cast<void*>(output_blob_ptr[names].get());
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            cudaError_t err = cudaMemcpyAsync(ptr, iter->second, datasize, cudaMemcpyDeviceToHost);
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
        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;
            size_t data_size = input_size[key];

            cudaError_t err = cudaMemcpyAsync(cuda_ptr, static_cast<void*>(cpu_ptr.data), data_size, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
            // set the input tensor
            context->setInputTensorAddress(key.c_str(), cuda_ptr);
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
        const auto &iter = outputBindings.find(names);
        if (iter != outputBindings.end())
        {
            cudaError_t err = cudaMemcpyAsync(ptr, iter->second, datasize, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // waiting for the stream
    cudaStreamSynchronize(stream);
    
    // Construct cv::Mat from shared_ptr (deep copy to avoid dangling pointers)
    std::unordered_map<std::string, cv::Mat> output_blob;
    for (const auto &names : output_names)
    {
        // First create cv::Mat using the shared_ptr data
        cv::Mat temp(
            output_shape[names].size(),
            output_shape[names].data(),
            utility::typeRt2Cv(engine->getTensorDataType(names.c_str())),
            output_blob_ptr[names].get()
        );
        
        // Deep copy to ensure independent memory management
        // This prevents dangling pointers when output_ptr is reused
        output_blob[names] = temp.clone();
    }
    
    return output_blob;
}

std::future<std::unordered_map<std::string, std::shared_ptr<char[]>>> 
TRTInfer::infer_async(const std::unordered_map<std::string, void *> &input_blob)
{
    if (!enable_async_ || !async_infer_ptr_) {
        throw std::runtime_error("Async inference not enabled. Use constructor with num_streams parameter.");
    }
    return async_infer_ptr_->infer_async(input_blob);
}

std::future<std::unordered_map<std::string, cv::Mat>> 
TRTInfer::infer_async(const std::unordered_map<std::string, cv::Mat> &input_blob)
{
    if (!enable_async_ || !async_infer_mat_) {
        throw std::runtime_error("Async inference not enabled. Use constructor with num_streams parameter.");
    }
    return async_infer_mat_->infer_async(input_blob);
}

void TRTInfer::infer_with_callback(const std::unordered_map<std::string, void *> &input_blob,
                                   std::function<void(const std::unordered_map<std::string, std::shared_ptr<char[]>>&)> callback)
{
    if (!enable_async_ || !async_infer_ptr_) {
        throw std::runtime_error("Async inference not enabled. Use constructor with num_streams parameter.");
    }
    async_infer_ptr_->infer_with_callback(input_blob, callback);
}

void TRTInfer::infer_with_callback(const std::unordered_map<std::string, cv::Mat> &input_blob,
                                   std::function<void(const std::unordered_map<std::string, cv::Mat>&)> callback)
{
    if (!enable_async_ || !async_infer_mat_) {
        throw std::runtime_error("Async inference not enabled. Use constructor with num_streams parameter.");
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
