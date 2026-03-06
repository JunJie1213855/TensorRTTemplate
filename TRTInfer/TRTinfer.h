#ifndef TRTINFER_H
#define TRTINFER_H
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
// #include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include "config.h"

class TRTInfer_API Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override;
};

class TRTInfer_API TRTInfer
{
    // init -> load engine -> allocate cuda memory  -> inference
public:
    TRTInfer() = delete;
    /**
     * @param engine_path The weight path of engine
     */
    TRTInfer(const std::string &engine_path);

    /**
     * @brief Model inference, calling the inner function internally
     * @param input_blob The output blob data consists of the first data being the name of the input tensor, and the second data being the address header of the tensor data
     * @return output blob tensor
     */
    std::unordered_map<std::string, std::shared_ptr<char[]>> operator()(const std::unordered_map<std::string, void *> &input_blob);
    
    
    /**
     * @brief  Model inference based on OpenCV cv::Mat, calling the inner function internally
     * @param input_blob The output blob data consists of the first data being the name of the input tensor, and the second data being the address header of the tensor data
     * @return output blob tensor
     */
    std::unordered_map<std::string, cv::Mat> operator()(const std::unordered_map<std::string, cv::Mat> &input_blob);

    /**
     * @brief Set input shape for dynamic batch inference
     * @param input_name The name of the input tensor
     * @param shape The shape to set (e.g., {batch_size, C, H, W})
     */
    void setInputShape(const std::string &input_name, const std::vector<int> &shape);

    ~TRTInfer();

private:
    void load_engine(const std::string &engine_path);

    void get_InputNames();

    void get_OutputNames();

    void get_bindings();

    void get_OptimizationProfiles();

    void set_OutputBlob();

    size_t allocateDynamicMemory(const std::string &name, const nvinfer1::Dims &dims, nvinfer1::DataType dtype,
                                 std::unordered_map<std::string, void *> &bindings,
                                 std::unordered_map<std::string, size_t> &max_sizes);

    std::unordered_map<std::string, std::shared_ptr<char[]>> infer(const std::unordered_map<std::string, void *> &input_blob);

    // for opencv Mat data
    std::unordered_map<std::string, cv::Mat> infer(const std::unordered_map<std::string, cv::Mat> &input_blob);

private:
    // plugin
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    Logger logger;

    // output blob data
    std::unordered_map<std::string, std::shared_ptr<char[]>> output_blob_ptr;

    // data
    std::vector<std::string> input_names, output_names;
    std::unordered_map<std::string, size_t> input_size, output_size;
    std::unordered_map<std::string, std::vector<int>> output_shape;
    cv::Size size;

    // Dynamic shape support
    std::unordered_map<std::string, std::vector<int>> current_input_shapes;  // Current input shapes
    std::unordered_map<std::string, nvinfer1::Dims> input_min_dims, input_opt_dims, input_max_dims;  // Optimization profile dimensions
    std::unordered_map<std::string, size_t> input_max_size, output_max_size;  // Maximum allocated memory sizes

    // bindings
    std::unordered_map<std::string, cv::Mat> input_Bindings, output_Bindings;

    // for cuda memory, just allocate one time
    std::unordered_map<std::string, void *> inputBindings, outputBindings;
};

#endif
