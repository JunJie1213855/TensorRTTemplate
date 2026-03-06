#ifndef TRTINFER_H
#define TRTINFER_H

// 只保留标准库和 OpenCV 依赖
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <opencv2/opencv.hpp>
#include "config.h"

class TRTInfer_API TRTInfer
{
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

    // 析构函数必须在 .cc 文件中定义
    ~TRTInfer();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;

    // 禁止拷贝
    TRTInfer(const TRTInfer&) = delete;
    TRTInfer& operator=(const TRTInfer&) = delete;

    // 允许移动
    TRTInfer(TRTInfer&&) = default;
    TRTInfer& operator=(TRTInfer&&) = default;
};

#endif
