#ifndef UTILITY_H
#define UTILITY_H

#include <cuda_runtime_api.h>
#include <iostream>
#include <NvInfer.h>
#include <opencv2/core.hpp>
#include "config.h"

/**
 * @brief TensorShape 结构体
 *
 * 封装张量形状信息，包含标准 TensorRT 维度的所有属性。
 * 对应 TensorRT 的 Dims 结构，但使用更易读的字段名。
 */
struct TensorShape
{
    int n = 0; /**< @brief 批大小 (Batch) */
    int c = 0; /**< @brief 通道数 (Channel) */
    int d = 0; /**< @brief 深度 (Depth) */
    int h = 0; /**< @brief 高度 (Height) */
    int w = 0; /**< @brief 宽度 (Width) */
};

namespace utility
{
    /**
     * @brief 将 vector 转换为 TensorShape
     *
     * @param  vec     整数向量，包含维度信息
     * @return TensorShape 转换后的形状结构体
     *
     * 将标准 vector 格式的维度转换为 TensorShape 结构体。
     * vector 元素依次对应 n, c, d, h, w。
     */
    TensorShape vectorToShape(const std::vector<int> &vec);

    /**
     * @brief Defer 工具类
     *
     * 用于实现延迟执行，类似 Go 语言的 defer 或 C++ 的 atexit。
     * 在作用域结束时自动执行指定的清理函数。
     *
     * 使用示例:
     * @code
     * {
     *     auto resource = acquireResource();
     *     utility::Defer defer([&resource]() {
     *         releaseResource(resource);
     *     });
     *     // 使用 resource...
     * } // 作用域结束时自动释放
     * @endcode
     */
    struct Defer
    {
        std::function<void()> func_; /**< @brief 延迟执行的函数 */

        /**
         * @brief 构造函数
         *
         * @param  func    延迟执行的函数对象
         */
        Defer(std::function<void()> func)
        {
            func_ = func;
        }

        /**
         * @brief 析构函数
         *
         * 在对象销毁时执行延迟的函数
         */
        ~Defer()
        {
            func_();
        }
    };

    /**
     * @brief 安全分配 GPU 显存
     *
     * @param  memSize     需要分配的字节数
     * @return void*       分配得到的 GPU 内存指针，失败返回 nullptr
     *
     * 封装 cudaMalloc，添加错误检查和日志输出。
     * 注意: 分配失败会打印错误信息但不会抛异常。
     */
    TRTInfer_API void *safeCudaMalloc(size_t memSize);

    /**
     * @brief 安全释放 GPU 显存
     *
     * @param  ptr         GPU 内存指针的引用
     * @return bool        释放成功返回 true，失败返回 false
     *
     * 封装 cudaFree，释放后将指针置为 nullptr。
     */
    TRTInfer_API bool safeCudaFree(void *&ptr);

    /**
     * @brief 获取 TensorRT 基本类型的字节大小
     *
     * @param  type    TensorRT 数据类型
     * @return size_t  该类型对应的字节数
     *
     * 支持的类型包括: kFLOAT, kHALF, kINT32, kINT8, kBOOL 等。
     */
    TRTInfer_API size_t getTypebytes(const nvinfer1::DataType &type);

    /**
     * @brief 计算张量的总字节大小
     *
     * @param  dim     张量维度 (Dims)
     * @param  type    TensorRT 数据类型
     * @return size_t  存储该张量所需的字节数
     *
     * 根据维度信息计算连续内存存储所需的字节数。
     * 计算公式: bytes = n * c * d * h * w * type_size
     */
    TRTInfer_API size_t getTensorbytes(const nvinfer1::Dims &dim, const nvinfer1::DataType &type);

    /**
     * @brief OpenCV 类型转换为 TensorRT 类型
     *
     * @param  cv_type     OpenCV 数据类型 (如 CV_32F, CV_8U)
     * @return nvinfer1::DataType  对应的 TensorRT 数据类型
     *
     * 如果无法转换，默认返回 kFLOAT。
     * 常见映射:
     * - CV_32F -> kFLOAT
     * - CV_8U  -> kINT8
     * - CV_32FC3 -> kFLOAT
     */
    TRTInfer_API nvinfer1::DataType typeCv2Rt(const int &cv_type);

    /**
     * @brief TensorRT 类型转换为 OpenCV 类型
     *
     * @param  rt_type     TensorRT 数据类型
     * @return int         对应的 OpenCV 数据类型
     *
     * 如果无法转换，默认返回 CV_32F。
     * 常见映射:
     * - kFLOAT -> CV_32F
     * - kINT8  -> CV_8U
     * - kHALF  -> CV_16F
     */
    TRTInfer_API int typeRt2Cv(const nvinfer1::DataType &rt_type);
}

#endif
