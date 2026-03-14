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

/**
 * @brief TensorShape 结构体 - 封装张量形状信息
 *
 * 用于存储 TensorRT 张量的维度信息，对应标准 TensorRT Dims 结构。
 * 包含批大小(Batch)、通道数(Channel)、深度(Depth)、高度(Height)、宽度(Width)。
 *
 * 示例:
 * @code
 * TensorShape shape = model.getInputShape("input");
 * std::cout << "Batch: " << shape.n << ", Channel: " << shape.c
 *           << ", Height: " << shape.h << ", Width: " << shape.w << std::endl;
 * @endcode
 */
struct TensorShape
{
    int n = 0; /**< @brief 批大小 (Batch Size) */
    int c = 0; /**< @brief 通道数 (Channel) */
    int d = 0; /**< @brief 深度 (Depth) */
    int h = 0; /**< @brief 高度 (Height) */
    int w = 0; /**< @brief 宽度 (Width) */
};

/**
 * @brief TRTInfer 类 - TensorRT 推理引擎封装
 *
 * 提供简洁的 C++ 接口用于 TensorRT 模型推理，支持同步和异步推理模式。
 * 使用 Pimpl 模式隐藏 TensorRT 内部实现细节。
 *
 * @section 功能特性
 * - 支持动态批处理推理
 * - 同步/异步推理接口
 * - 自动内存管理
 * - 多线程安全
 *
 * @section 使用示例
 * @code
 * // 创建推理实例
 * auto model = TRTInfer::create("model.engine");
 *
 * // 准备输入数据 (使用 cv::Mat)
 * std::unordered_map<std::string, cv::Mat> input;
 * input["input"] = cv::imread("image.jpg");
 *
 * // 同步推理
 * auto output = model(input);
 *
 * // 获取输出
 * cv::Mat result = output["output"];
 * cv::imwrite("result.jpg", result);
 * @endcode
 */
class TRTInfer_API TRTInfer
{
public:
    /**
     * @brief 工厂方法，创建 TRTInfer 实例
     *
     * @param  engine_path   TensorRT 引擎文件路径 (.engine 二进制文件)
     * @return std::shared_ptr<TRTInfer> 返回智能指针管理的实例
     *
     * @note 该方法采用延迟初始化策略，构造对象时不会立即加载引擎，
     *       只有在实际调用推理或查询张量信息时才会初始化资源。
     *       这样做可以:
     *       - 加快对象构造速度
     *       - 支持在构造时只保存路径，后续按需初始化
     *       - 允许多次创建不同引擎的对象而不占用额外资源
     *
     * @throw std::runtime_error 当引擎文件不存在或格式错误时抛出异常
     *
     * 示例:
     * @code
     * try {
     *     auto model = TRTInfer::create("yolov8.engine");
     *     std::cout << "模型创建成功" << std::endl;
     * } catch (const std::exception& e) {
     *     std::cerr << "模型创建失败: " << e.what() << std::endl;
     * }
     * @endcode
     */
    static std::shared_ptr<TRTInfer> create(const std::string &engine_path);

    /**
     * @brief 初始化 TensorRT 引擎和相关资源
     *
     * 手动触发初始化，加载引擎文件、获取张量信息、分配 CUDA 内存。
     * 通常由推理接口自动调用，也可手动调用以预热引擎或提前验证引擎有效性。
     *
     * @note 如果引擎已初始化，此方法不会重复初始化。
     *
     * @throw std::runtime_error 当引擎加载失败时抛出异常
     *
     * 示例:
     * @code
     * auto model = TRTInfer::create("model.engine");
     * model->Init();  // 手动初始化（可选）
     * // 或用于预热
     * model->Init();
     * @endcode
     */
    void Init();

    /**
     * @brief 析构函数
     *
     * 释放所有 CUDA 资源、工作线程和内存。
     * 由智能指针自动调用，无需手动管理。
     */
    ~TRTInfer();

    // ===================== 推理接口 =====================

    /**
     * @brief 同步推理 (使用原始指针)
     *
     * @param  input_blob    输入张量 map，键为张量名称，值为 GPU 指针
     * @return std::unordered_map<std::string, std::shared_ptr<char[]>> 输出张量 map
     *
     * @note 适用于已手动分配 GPU 内存的场景，输入输出均为原始指针。
     *       内部会处理内存拷贝和同步。
     *
     * @warning 调用者需要确保:
     *       - 输入指针指向有效的 GPU 内存
     *       - 内存大小与张量形状匹配
     *       - 使用完毕后自行释放 GPU 内存（或使用返回的智能指针）
     *
     * 示例:
     * @code
     * void* input_ptr;
     * cudaMalloc(&input_ptr, input_size);
     * cudaMemcpy(input_ptr, cpu_data, input_size, cudaMemcpyHostToDevice);
     *
     * std::unordered_map<std::string, void*> input;
     * input["input"] = input_ptr;
     *
     * auto output = model(input);
     * char* output_ptr = output["output"].get();
     * @endcode
     */
    std::unordered_map<std::string, std::shared_ptr<char[]>> operator()(const std::unordered_map<std::string, void *> &input_blob);

    /**
     * @brief 同步推理 (使用 cv::Mat)
     *
     * @param  input_blob    输入张量 map，键为张量名称，值为 cv::Mat 格式数据
     * @return std::unordered_map<std::string, cv::Mat> 输出张量 map
     *
     * @note 这是最常用的推理接口，自动处理:
     *       - CPU 到 GPU 的数据拷贝
     *       - 内存分配
     *       - CUDA 同步
     *       - GPU 到 CPU 的结果回传
     *
     * @note 输入 cv::Mat 会被自动传输到 GPU，输出 cv::Mat 存储在 CPU 内存中。
     *
     * 示例:
     * @code
     * std::unordered_map<std::string, cv::Mat> input;
     * input["images"] = cv::imread("test.jpg");
     *
     * auto output = model(input);
     * cv::Mat result = output["output"];
     *
     * // 直接使用结果
     * cv::imshow("result", result);
     * cv::waitKey(0);
     * @endcode
     */
    std::unordered_map<std::string, cv::Mat> operator()(const std::unordered_map<std::string, cv::Mat> &input_blob);

    // ===================== 动态形状接口 =====================

    /**
     * @brief 设置输入张量形状 (用于动态批处理推理)
     *
     * @param  input_name    输入张量名称
     * @param  shape         形状向量，格式为 {batch_size, channels, height, width} 或 {batch_size, channels, depth, height, width}
     *
     * @note 仅当引擎支持动态形状时才需要调用此方法。
     *       对于固定形状的引擎，此方法不会生效。
     *
     * @note 必须在推理前调用，且在每次推理前需要重新设置（如果批大小变化）。
     *
     * 示例:
     * @code
     * auto model = TRTInfer::create("dynamic_batch.engine");
     *
     * // 设置不同批大小
     * model->setInputShape("input", {1, 3, 640, 640});   // 单张
     * auto output1 = model(input1);
     *
     * model->setInputShape("input", {4, 3, 640, 640});   // 批大小为4
     * auto output2 = model(input_batch);
     * @endcode
     */
    void setInputShape(const std::string &input_name, const std::vector<int> &shape);

    // ===================== 查询接口 =====================

    /**
     * @brief 获取所有输入张量的名称
     *
     * @return std::vector<std::string> 输入张量名称列表
     *
     * @note 返回引擎定义的所有输入张量名称，可用于遍历验证输入数据。
     *       如果引擎未初始化，会触发延迟初始化。
     *
     * 示例:
     * @code
     * auto model = TRTInfer::create("model.engine");
     * auto input_names = model->getInputNames();
     * for (const auto& name : input_names) {
     *     std::cout << "输入张量: " << name << std::endl;
     *     auto shape = model->getInputShape(name);
     *     std::cout << "  形状: [" << shape.n << ", " << shape.c << ", "
     *               << shape.h << ", " << shape.w << "]" << std::endl;
     * }
     * @endcode
     */
    std::vector<std::string> getInputNames() const;

    /**
     * @brief 获取所有输出张量的名称
     *
     * @return std::vector<std::string> 输出张量名称列表
     *
     * @note 返回引擎定义的所有输出张量名称，可用于遍历输出数据。
     *       如果引擎未初始化，会触发延迟初始化。
     *
     * 示例:
     * @code
     * auto output_names = model->getOutputNames();
     * for (const auto& name : output_names) {
     *     std::cout << "输出张量: " << name << std::endl;
     * }
     * @endcode
     */
    std::vector<std::string> getOutputNames() const;

    /**
     * @brief 获取指定输入张量的形状
     *
     * @param  name    输入张量名称
     * @return TensorShape 形状信息，包含 n(批大小)、c(通道)、d(深度)、h(高)、w(宽)
     *
     * @note 如果指定的张量名称不存在，会返回默认构造的 TensorShape（所有维度为0）。
     *       如果引擎未初始化，会触发延迟初始化。
     *
     * 示例:
     * @code
     * TensorShape shape = model->getInputShape("images");
     * if (shape.n > 0) {
     *     std::cout << "输入形状: Batch=" << shape.n
     *               << ", Channel=" << shape.c
     *               << ", H=" << shape.h
     *               << ", W=" << shape.w << std::endl;
     * }
     * @endcode
     */
    TensorShape getInputShape(const std::string &name) const;

    /**
     * @brief 获取指定输出张量的形状
     *
     * @param  name    输出张量名称
     * @return TensorShape 形状信息，包含 n(批大小)、c(通道)、d(深度)、h(高)、w(宽)
     *
     * @note 如果指定的张量名称不存在，会返回默认构造的 TensorShape（所有维度为0）。
     *       如果引擎未初始化，会触发延迟初始化。
     *
     * 示例:
     * @code
     * TensorShape shape = model->getOutputShape("output0");
     * std::cout << "输出形状: [" << shape.n << ", " << shape.c << ", "
     *           << shape.h << ", " << shape.w << "]" << std::endl;
     * @endcode
     */
    TensorShape getOutputShape(const std::string &name) const;

private:
    /**
     * @brief 私有构造函数
     *
     * @param  engine_path   TensorRT 引擎文件路径
     *
     * @note 构造函数为私有，需要通过 create() 工厂方法创建实例。
     *       构造时仅保存路径，不加载引擎，实现延迟初始化。
     */
    TRTInfer(const std::string &engine_path);

    /**
     * @brief 默认构造函数被删除
     *
     * 不允许使用默认构造函数创建实例，必须通过 create() 工厂方法。
     */
    TRTInfer() = delete;

    // 禁止拷贝和移动
    TRTInfer(const TRTInfer &) = delete;
    TRTInfer &operator=(const TRTInfer &) = delete;
    TRTInfer(TRTInfer &&) = delete;
    TRTInfer &operator=(TRTInfer &&) = delete;

private:
    class Impl;                  /**< @brief 实现类前向声明 */
    std::unique_ptr<Impl> pImpl; /**< @brief 智能指针管理实现类 */
};

#endif
