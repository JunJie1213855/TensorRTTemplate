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
    /**
     * @brief BlobType 类型别名
     *
     * 用于存储输入输出张量的数据结构，键为张量名称，值为 cv::Mat 格式的张量数据
     */
    using BlobType = std::unordered_map<std::string, cv::Mat>;

    /**
     * @brief TRTInfer 类
     *
     * TensorRT 推理引擎的封装类，提供同步和异步推理接口。
     * 使用 Pimpl 模式隐藏实现细节，支持多线程异步推理。
     *
     * 使用示例:
     * @code
     * auto model = TRTInfer::create("model.engine", 4);
     * auto future = model->PostQueue(input_blob);
     * auto output = future.get();
     * @endcode
     */
    class TRTInfer_API TRTInfer
    {
    public:
        /**
         * @brief 工厂方法，创建 TRTInfer 实例
         *
         * @param  engine_path   TensorRT 引擎文件路径
         * @param  num_thread    工作线程数量，默认为 1
         * @return std::shared_ptr<TRTInfer> 返回智能指针管理的实例
         *
         * 该方法采用延迟初始化策略，构造对象时不会立即加载引擎，
         * 只有在实际调用推理或查询张量信息时才会初始化资源。
         */
        static std::shared_ptr<TRTInfer> create(const std::string &engine_path, int num_thread = 1);

        /**
         * @brief 初始化 TensorRT 引擎和相关资源
         *
         * 加载引擎文件、获取张量信息、分配 CUDA 内存、创建工作线程。
         * 通常由 create() 自动调用，也可手动调用以预热引擎。
         */
        void Init();

        /**
         * @brief 析构函数
         *
         * 释放所有 CUDA 资源和工作线程。
         */
        ~TRTInfer();

        /**
         * @brief 同步推理操作符
         *
         * @param  input_blob     输入张量 map，键为张量名称，值为 cv::Mat 格式数据
         * @return BlobType      输出张量 map，包含所有输出张量的结果
         *
         * 该方法内部调用 PostQueue 将任务加入队列，等待完成并返回结果。
         * 适用于单次推理或需要阻塞等待结果的场景。
         *
         * 示例:
         * @code
         * BlobType input;
         * input["input"] = cv::Mat(...);
         * BlobType output = model(input);
         * @endcode
         */
        BlobType operator()(const BlobType &input_blob);

        /**
         * @brief 异步推理，将任务推送到工作队列
         *
         * @param  input_blob     输入张量 map，键为张量名称，值为 cv::Mat 格式数据
         * @return std::future<BlobType> 返回异步结果Future，可在后续获取输出
         *
         * 该方法立即返回，不阻塞调用线程。推理任务被添加到内部队列，
         * 由工作线程异步执行。适用于需要并行提交多个推理任务的场景。
         *
         * 示例:
         * @code
         * std::vector<std::future<BlobType>> results;
         * for (int i = 0; i < batch_size; i++) {
         *     results.push_back(model->PostQueue(input_blobs[i]));
         * }
         * for (auto& f : results) {
         *     auto output = f.get();
         * }
         * @endcode
         */
        std::future<BlobType> PostQueue(const BlobType &input_blob);

    private:
        /**
         * @brief 私有构造函数
         *
         * @param  engine_path   TensorRT 引擎文件路径
         * @param  num_thread    工作线程数量，默认为 1
         *
         * 构造函数为私有，需要通过 create() 工厂方法创建实例。
         */
        TRTInfer(const std::string &engine_path, int num_thread = 1);

    public:
        /**
         * @brief 获取所有输入张量的名称
         *
         * @return std::vector<std::string> 输入张量名称列表
         *
         * 返回引擎中所有输入张量的名称，可用于遍历或验证输入数据。
         * 如果引擎未初始化，会触发延迟初始化。
         */
        std::vector<std::string> getInputNames() const;

        /**
         * @brief 获取所有输出张量的名称
         *
         * @return std::vector<std::string> 输出张量名称列表
         *
         * 返回引擎中所有输出张量的名称，可用于遍历输出数据。
         * 如果引擎未初始化，会触发延迟初始化。
         */
        std::vector<std::string> getOutputNames() const;

        /**
         * @brief 获取指定输入张量的形状
         *
         * @param  name    输入张量名称
         * @return TensorShape 形状信息，包含 n(批大小)、c(通道)、d(深度)、h(高)、w(宽)
         *
         * 如果指定的张量名称不存在，会返回默认构造的 TensorShape。
         * 如果引擎未初始化，会触发延迟初始化。
         */
        TensorShape getInputShape(const std::string &name) const;

        /**
         * @brief 获取指定输出张量的形状
         *
         * @param  name    输出张量名称
         * @return TensorShape 形状信息，包含 n(批大小)、c(通道)、d(深度)、h(高)、w(宽)
         *
         * 如果指定的张量名称不存在，会返回默认构造的 TensorShape。
         * 如果引擎未初始化，会触发延迟初始化。
         */
        TensorShape getOutputShape(const std::string &name) const;

    private:
        class Impl;                    /**< @brief 实现类前向声明 */
        std::unique_ptr<Impl> pImpl;   /**< @brief 智能指针管理实现类 */

        // 禁止拷贝
        TRTInfer(const TRTInfer &) = delete;
        TRTInfer &operator=(const TRTInfer &) = delete;

        // 禁止移动
        TRTInfer(TRTInfer &&) = delete;
        TRTInfer &operator=(TRTInfer &&) = delete;
    };
}

#endif
