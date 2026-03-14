#ifndef TRTINFER_H
#define TRTINFER_H

// 只保留标准库和 OpenCV 依赖
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <future>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "utility.h"

namespace TRT
{
    using BlobType = std::unordered_map<std::string, cv::Mat>;

    class TRTInfer_API TRTInfer
    {
    public:
        

        static std::shared_ptr<TRTInfer> create(const std::string &engine_path, int num_thread = 1);
        

        void Init();


        /**
         * @brief  Model inference based on OpenCV cv::Mat, calling the inner function internally
         * @param input_blob The output blob data consists of the first data being the name of the input tensor, and the second data being the address header of the tensor data
         * @return output blob tensor
         */
        BlobType operator()(const BlobType &input_blob);

        /**
         * @brief 多线程运行, 数据推送到队列中
         * @param input_blob 输入数据
         * @return 输出张量
         */
        std::future<BlobType> PostQueue(const BlobType &input_blob);

        // 析构函数必须在 .cc 文件中定义
        ~TRTInfer();

    private:
        TRTInfer() = delete;

        /**
         * @param engine_path The weight path of engine
         */
        TRTInfer(const std::string &engine_path, int num_thread = 1);

    public:
        /**
         * @brief 获取所有输入张量的名称
         * @return 输入张量名称列表
         */
        std::vector<std::string> getInputNames() const;

        /**
         * @brief 获取所有输出张量的名称
         * @return 输出张量名称列表
         */
        std::vector<std::string> getOutputNames() const;

        /**
         * @brief 获取指定输入张量的形状
         * @param name 输入张量名称
         * @return TensorShape 形状信息
         */
        TensorShape getInputShape(const std::string &name) const;

        /**
         * @brief 获取指定输出张量的形状
         * @param name 输出张量名称
         * @return TensorShape 形状信息
         */
        TensorShape getOutputShape(const std::string &name) const;

    private:
        class Impl;
        std::unique_ptr<Impl> pImpl;

        // 禁止拷贝
        TRTInfer(const TRTInfer &) = delete;
        TRTInfer &operator=(const TRTInfer &) = delete;

        // 允许移动
        TRTInfer(TRTInfer &&) = default;
        TRTInfer &operator=(TRTInfer &&) = default;
    };
}

#endif
