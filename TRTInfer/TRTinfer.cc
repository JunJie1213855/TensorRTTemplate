#include "TRTinfer.h"
// for dim
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
TRTInfer::TRTInfer(const std::string &engine_path) : logger()
{

    load_engine(engine_path);

    get_InputNames();

    get_OutputNames();

    get_OptimizationProfiles();

    get_bindings();

    set_OutputBlob();

    cudaStreamCreate(&stream);
}
TRTInfer::~TRTInfer()
{
    // destory stream
    cudaStreamDestroy(stream);

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
        input_max_size[input_names[i]] = input_size[input_names[i]];
    }
    // allocate output memeory for cuda
    for (int i = 0; i < output_names.size(); i++)
    {
        outputBindings[output_names[i]] = utility::safeCudaMalloc(output_size[output_names[i]]);
        output_max_size[output_names[i]] = output_size[output_names[i]];
    }
}

void TRTInfer::get_OptimizationProfiles()
{
    // Get optimization profile information for dynamic shapes
    int numProfiles = engine->getNbOptimizationProfiles();
    std::cout << "Number of optimization profiles: " << numProfiles << std::endl;

    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char *name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            std::string tensorName(name);

            // Check if this tensor has a dynamic shape (contains -1)
            nvinfer1::Dims shape = engine->getTensorShape(name);
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
                // Get dimensions from profile 0 (default profile)
                nvinfer1::OptProfileSelector selectors[] = {
                    nvinfer1::OptProfileSelector::kMIN,
                    nvinfer1::OptProfileSelector::kOPT,
                    nvinfer1::OptProfileSelector::kMAX
                };

                input_min_dims[tensorName] = engine->getProfileShape(name, 0, selectors[0]);
                input_opt_dims[tensorName] = engine->getProfileShape(name, 0, selectors[1]);
                input_max_dims[tensorName] = engine->getProfileShape(name, 0, selectors[2]);

                // Use max dimensions for initial allocation
                current_input_shapes[tensorName] = std::vector<int>();
                for (int j = 0; j < input_max_dims[tensorName].nbDims; j++)
                {
                    current_input_shapes[tensorName].push_back(input_max_dims[tensorName].d[j]);
                }

                std::cout << "Dynamic input tensor: " << tensorName << std::endl;
                std::cout << "  Min shape: " << input_min_dims[tensorName] << std::endl;
                std::cout << "  Opt shape: " << input_opt_dims[tensorName] << std::endl;
                std::cout << "  Max shape: " << input_max_dims[tensorName] << std::endl;

                // Update input_size to max size for initial allocation
                input_size[tensorName] = utility::getTensorbytes(input_max_dims[tensorName],
                                                                  engine->getTensorDataType(name));
            }
            else
            {
                // Static shape, use as-is
                current_input_shapes[tensorName] = std::vector<int>();
                for (int j = 0; j < shape.nbDims; j++)
                {
                    current_input_shapes[tensorName].push_back(shape.d[j]);
                }
            }
        }
    }
}

void TRTInfer::setInputShape(const std::string &input_name, const std::vector<int> &shape)
{
    // Check if input exists
    if (current_input_shapes.find(input_name) == current_input_shapes.end())
    {
        std::cerr << "Input tensor '" << input_name << "' not found" << std::endl;
        throw std::runtime_error("Input tensor not found");
    }

    // Convert vector to Dims
    nvinfer1::Dims dims;
    dims.nbDims = shape.size();
    for (size_t i = 0; i < shape.size(); i++)
    {
        dims.d[i] = shape[i];
    }

    // Set the input shape
    if (!context->setInputShape(input_name.c_str(), dims))
    {
        std::cerr << "Failed to set input shape for '" << input_name << "'" << std::endl;
        throw std::runtime_error("Failed to set input shape");
    }

    // Update current shape
    current_input_shapes[input_name] = shape;

    // Reallocate memory if needed (use max shape to ensure enough memory)
    allocateDynamicMemory(input_name, dims, engine->getTensorDataType(input_name.c_str()),
                         inputBindings, input_max_size);
}

size_t TRTInfer::allocateDynamicMemory(const std::string &name, const nvinfer1::Dims &dims,
                                        nvinfer1::DataType dtype,
                                        std::unordered_map<std::string, void *> &bindings,
                                        std::unordered_map<std::string, size_t> &max_sizes)
{
    size_t required_size = utility::getTensorbytes(dims, dtype);

    // Check if we need to allocate more memory
    if (required_size > max_sizes[name])
    {
        // Free old memory
        if (bindings.find(name) != bindings.end() && bindings[name] != nullptr)
        {
            utility::safeCudaFree(bindings[name]);
        }

        // Allocate new memory
        bindings[name] = utility::safeCudaMalloc(required_size);
        max_sizes[name] = required_size;

        std::cout << "Reallocated memory for '" << name << "': " << required_size << " bytes" << std::endl;
    }

    return required_size;
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
            // Calculate actual size based on current input shape
            nvinfer1::Dims current_dims;
            current_dims.nbDims = current_input_shapes[key].size();
            for (size_t i = 0; i < current_input_shapes[key].size(); i++)
            {
                current_dims.d[i] = current_input_shapes[key][i];
            }
            size_t data_size = utility::getTensorbytes(current_dims, engine->getTensorDataType(key.c_str()));

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

    // Handle dynamic output shapes and reallocate if needed
    for (const auto &names : output_names)
    {
        // Get actual output shape after inference
        nvinfer1::Dims out_shape = context->getTensorShape(names.c_str());
        size_t actual_size = utility::getTensorbytes(out_shape, engine->getTensorDataType(names.c_str()));

        // Reallocate output memory on GPU if needed
        if (actual_size > output_max_size[names])
        {
            utility::safeCudaFree(outputBindings[names]);
            outputBindings[names] = utility::safeCudaMalloc(actual_size);
            output_max_size[names] = actual_size;
            context->setOutputTensorAddress(names.c_str(), outputBindings[names]);
        }

        // Reallocate CPU memory if needed
        if (actual_size > output_size[names])
        {
            output_blob_ptr[names] = std::shared_ptr<char[]>(new char[actual_size]);
            output_size[names] = actual_size;
        }

        // Update output shape
        output_shape[names].clear();
        for (int i = 0; i < out_shape.nbDims; i++)
        {
            output_shape[names].push_back(out_shape.d[i]);
        }

        // Copy the gpu data to cpu data
        void* ptr = static_cast<void*>(output_blob_ptr[names].get());
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

        // Type conversion
        if (utility::typeCv2Rt(cpu_ptr.type()) != engine->getTensorDataType(key.c_str()))
            cpu_ptr.convertTo(cpu_ptr, utility::typeRt2Cv(engine->getTensorDataType(key.c_str())));
        auto iter = inputBindings.find(key);
        if (iter != inputBindings.end())
        {
            void *cuda_ptr = iter->second;
            // Calculate actual size based on current input shape
            nvinfer1::Dims current_dims;
            current_dims.nbDims = current_input_shapes[key].size();
            for (size_t i = 0; i < current_input_shapes[key].size(); i++)
            {
                current_dims.d[i] = current_input_shapes[key][i];
            }
            size_t data_size = utility::getTensorbytes(current_dims, engine->getTensorDataType(key.c_str()));

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

    // Handle dynamic output shapes and reallocate if needed
    for (const auto &names : output_names)
    {
        // Get actual output shape after inference
        nvinfer1::Dims out_shape = context->getTensorShape(names.c_str());
        size_t actual_size = utility::getTensorbytes(out_shape, engine->getTensorDataType(names.c_str()));

        // Reallocate output memory on GPU if needed
        if (actual_size > output_max_size[names])
        {
            utility::safeCudaFree(outputBindings[names]);
            outputBindings[names] = utility::safeCudaMalloc(actual_size);
            output_max_size[names] = actual_size;
            context->setOutputTensorAddress(names.c_str(), outputBindings[names]);
        }

        // Reallocate CPU memory if needed
        if (actual_size > output_size[names])
        {
            output_blob_ptr[names] = std::shared_ptr<char[]>(new char[actual_size]);
            output_size[names] = actual_size;
        }

        // Update output shape
        output_shape[names].clear();
        for (int i = 0; i < out_shape.nbDims; i++)
        {
            output_shape[names].push_back(out_shape.d[i]);
        }

        // Copy the gpu data to cpu data
        void* ptr = static_cast<void*>(output_blob_ptr[names].get());
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
