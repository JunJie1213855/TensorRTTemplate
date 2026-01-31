#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>

namespace S2M2Async
{
    std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &left, cv::Mat &right)
    {
        std::unordered_map<std::string, cv::Mat> input_blob;

        cv::Mat left_rgb, right_rgb;
        cv::cvtColor(left, left_rgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(right, right_rgb, cv::COLOR_BGR2RGB);

        int img_height = 736;
        int img_width = 1280;

        cv::Mat left_cropped = cv::Mat::zeros(cv::Size(img_width, img_height), CV_32FC3);
        cv::Mat right_cropped = cv::Mat::zeros(cv::Size(img_width, img_height), CV_32FC3);
        left_rgb(cv::Rect(0, 0, left_rgb.cols, left_rgb.rows)).copyTo(left_cropped(cv::Rect(0, 0, left_rgb.cols, left_rgb.rows)));
        right_rgb(cv::Rect(0, 0, right_rgb.cols, right_rgb.rows)).copyTo(right_cropped(cv::Rect(0, 0, right_rgb.cols, right_rgb.rows)));

        std::cout << "原始尺寸: " << left_rgb.rows << " x " << left_rgb.cols << std::endl;
        std::cout << "裁剪尺寸: " << img_height << " x " << img_width << std::endl;

        input_blob["input_left"] = cv::dnn::blobFromImage(left_cropped, 1.0, cv::Size(img_width, img_height),
                                                          cv::Scalar(), true, false);
        input_blob["input_right"] = cv::dnn::blobFromImage(right_cropped, 1.0, cv::Size(img_width, img_height),
                                                         cv::Scalar(), true, false);
        return input_blob;
    }

    void postprocess(const cv::Mat &disp, cv::Mat &disp_vis)
    {
        cv::Mat disp_c = disp.clone();
        double min, max;
        cv::minMaxLoc(disp_c, &min, &max);
        cv::Mat disp_norm = ((disp_c - min) / (max - min)) * 255;
        disp_norm.convertTo(disp_vis, CV_8U);
        cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_INFERNO);
    }
}

int main(int argc, char *argv[])
{
    std::cout << "=== TensorRT S2M2 异步推理示例 ===" << std::endl;
    
    cv::Mat left = cv::imread("/root/code/C++/TensorRTTemplate/rect_left.png");
    cv::Mat right = cv::imread("/root/code/C++/TensorRTTemplate/rect_right.png");

    if (left.empty() || right.empty())
    {
        std::cerr << "无法加载图像文件" << std::endl;
        return -1;
    }

    std::cout << "使用 4 个 CUDA 流初始化模型..." << std::endl;
    TRTInfer model("/root/code/C++/TensorRTTemplate/model.engine", 4, true);

    auto input_blob = S2M2Async::preprocess(left, right);

    const int warmup_iterations = 3;
    std::cout << "\n进行 " << warmup_iterations << " 次 warmup 推理..." << std::endl;
    for (int i = 0; i < warmup_iterations; ++i)
    {
        model.infer_async(input_blob).get();
    }
    std::cout << "Warmup 完成" << std::endl;

    const int num_iterations = 10;
    
    std::cout << "\n1. 测试同步推理..." << std::endl;
    std::vector<long long> sync_times;
    cv::Mat sync_disp, sync_disp_conf;
    
    for (int i = 0; i < num_iterations; ++i)
    {
        auto start_sync = std::chrono::high_resolution_clock::now();
        auto output_sync = model(input_blob);
        auto end_sync = std::chrono::high_resolution_clock::now();
        auto duration_sync = std::chrono::duration_cast<std::chrono::milliseconds>(end_sync - start_sync).count();
        sync_times.push_back(duration_sync);
    }
    
    double avg_sync = 0;
    for (auto t : sync_times) avg_sync += t;
    avg_sync /= sync_times.size();
    
    std::cout << "同步推理平均时间: " << avg_sync << " ms" << std::endl;
    std::cout << "同步推理最小时间: " << *std::min_element(sync_times.begin(), sync_times.end()) << " ms" << std::endl;
    std::cout << "同步推理最大时间: " << *std::max_element(sync_times.begin(), sync_times.end()) << " ms" << std::endl;

    auto output_sync_final = model(input_blob);
    S2M2Async::postprocess(output_sync_final["output_disp"].reshape(1, 736), sync_disp);
    S2M2Async::postprocess(output_sync_final["output_conf"].reshape(1, 736), sync_disp_conf);
    cv::imwrite("s2m2_result_sync_disp.png", sync_disp);
    cv::imwrite("s2m2_result_sync_conf.png", sync_disp_conf);
    std::cout << "已保存同步结果: s2m2_result_sync_disp.png, s2m2_result_sync_conf.png" << std::endl;

    std::cout << "\n2. 测试异步推理 (Future 方式)..." << std::endl;
    std::vector<long long> async_times;
    cv::Mat async_disp, async_disp_conf;
    
    for (int i = 0; i < num_iterations; ++i)
    {
        auto start_async = std::chrono::high_resolution_clock::now();
        auto future = model.infer_async(input_blob);
        auto output_async = future.get();
        auto end_async = std::chrono::high_resolution_clock::now();
        auto duration_async = std::chrono::duration_cast<std::chrono::milliseconds>(end_async - start_async).count();
        async_times.push_back(duration_async);
    }
    
    double avg_async = 0;
    for (auto t : async_times) avg_async += t;
    avg_async /= async_times.size();
    
    std::cout << "异步推理平均时间: " << avg_async << " ms" << std::endl;
    std::cout << "异步推理最小时间: " << *std::min_element(async_times.begin(), async_times.end()) << " ms" << std::endl;
    std::cout << "异步推理最大时间: " << *std::max_element(async_times.begin(), async_times.end()) << " ms" << std::endl;

    auto output_async_final = model.infer_async(input_blob).get();
    S2M2Async::postprocess(output_async_final["output_disp"].reshape(1, 736), async_disp);
    S2M2Async::postprocess(output_async_final["output_conf"].reshape(1, 736), async_disp_conf);
    cv::imwrite("s2m2_result_async_disp.png", async_disp);
    cv::imwrite("s2m2_result_async_conf.png", async_disp_conf);
    std::cout << "已保存异步结果: s2m2_result_async_disp.png, s2m2_result_async_conf.png" << std::endl;

    std::cout << "\n3. 测试并发推理..." << std::endl;
    const int num_concurrent = 8;
    std::vector<long long> concurrent_times;
    
    for (int iter = 0; iter < 5; ++iter)
    {
        std::vector<std::future<std::unordered_map<std::string, cv::Mat>>> futures;
        
        auto start_concurrent = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_concurrent; ++i)
        {
            futures.push_back(model.infer_async(input_blob));
        }
        
        for (auto &f : futures)
        {
            f.get();
        }
        
        auto end_concurrent = std::chrono::high_resolution_clock::now();
        auto duration_concurrent = std::chrono::duration_cast<std::chrono::milliseconds>(end_concurrent - start_concurrent).count();
        concurrent_times.push_back(duration_concurrent);
    }
    
    double avg_concurrent = 0;
    for (auto t : concurrent_times) avg_concurrent += t;
    avg_concurrent /= concurrent_times.size();
    
    std::cout << "完成 " << num_concurrent << " 次并发推理，平均总时间: " << avg_concurrent << " ms" << std::endl;
    std::cout << "每次推理平均时间: " << (avg_concurrent / num_concurrent) << " ms" << std::endl;

    std::cout << "\n=== 性能总结 ===" << std::endl;
    std::cout << "同步推理平均: " << avg_sync << " ms" << std::endl;
    std::cout << "异步推理平均: " << avg_async << " ms" << std::endl;
    std::cout << "并发推理平均 (总时间): " << avg_concurrent << " ms (" << num_concurrent << " 次)" << std::endl;
    std::cout << "并发推理单次平均: " << (avg_concurrent / num_concurrent) << " ms" << std::endl;
    std::cout << "性能提升 (并发 vs 同步): " << (avg_sync / (avg_concurrent / num_concurrent)) << "x" << std::endl;
    std::cout << "模型拥有 " << model.num_streams() << " 个 CUDA 流" << std::endl;

    bool has_display = (getenv("DISPLAY") != nullptr) || (getenv("WAYLAND_DISPLAY") != nullptr);
    if (has_display)
    {
        cv::imshow("同步结果 - Disparity", sync_disp);
        cv::imshow("同步结果 - Confidence", sync_disp_conf);
        cv::imshow("异步结果 - Disparity", async_disp);
        cv::imshow("异步结果 - Confidence", async_disp_conf);
        cv::waitKey(0);
    }
    else
    {
        std::cout << "\n无显示环境，跳过显示窗口" << std::endl;
    }

    return 0;
}
