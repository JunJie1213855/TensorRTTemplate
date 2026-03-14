#include "TRTinfer.h"
#include "benchmark.h"
#include <opencv2/opencv.hpp>
#include <chrono>
using namespace TRT;
std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &left, cv::Mat &right)
{
    std::unordered_map<std::string, cv::Mat> input_blob;

    // Step 1: BGR to RGB conversion (matches cv2.cvtColor(..., cv2.COLOR_BGR2RGB))
    cv::Mat left_rgb, right_rgb;
    cv::cvtColor(left, left_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(right, right_rgb, cv::COLOR_BGR2RGB);

    // Step 2: Resize to multiple of 32 by cropping (not scaling)
    // Python: img_height = (img_height // 32) * 32
    int img_height = 736; // fixed height
    int img_width = 1280; // fixed width

    // Crop images to multiple of 32
    cv::Mat left_cropped = cv::Mat::zeros(cv::Size(img_width, img_height), CV_32FC3);
    cv::Mat right_cropped = cv::Mat::zeros(cv::Size(img_width, img_height), CV_32FC3);
    left_rgb(cv::Rect(0, 0, left_rgb.cols, left_rgb.rows)).copyTo(left_cropped(cv::Rect(0, 0, left_rgb.cols, left_rgb.rows)));
    right_rgb(cv::Rect(0, 0, right_rgb.cols, right_rgb.rows)).copyTo(right_cropped(cv::Rect(0, 0, right_rgb.cols, right_rgb.rows)));

    std::cout << "Original size: " << left_rgb.rows << " x " << left_rgb.cols << std::endl;
    std::cout << "Cropped size: " << img_height << " x " << img_width << std::endl;

    // Step 3: Convert to CHW format and add batch dimension
    // Python: torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)
    // No normalization, no scaling, just dimension permutation

    // Convert to blob (automatically does: BGR->RGB, HWC->CHW, adds batch dim)
    // scalefactor=1.0 (no scaling), size (original size, no resize), swapRB=true (BGR->RGB)
    // mean (no mean subtraction), crop=false
    input_blob["input_left"] = cv::dnn::blobFromImage(left_cropped, 1.0, cv::Size(img_width, img_height),
                                                      cv::Scalar(), true, false);
    input_blob["input_right"] = cv::dnn::blobFromImage(right_cropped, 1.0, cv::Size(img_width, img_height),
                                                       cv::Scalar(), true, false);
    return input_blob;
}

void postprocess(const cv::Mat &disp, cv::Mat &disp_vis)
{
    // to visualization
    cv::Mat disp_c = disp.clone();
    double min, max;
    cv::minMaxLoc(disp_c, &min, &max);
    cv::Mat disp_norm = ((disp_c - min) / (max - min)) * 255;
    disp_norm.convertTo(disp_vis, CV_8U);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_INFERNO);
}

int main(int argc, char *argv[])
{
    // 路径配置
    std::string left_path = "rect_left.png";
    std::string right_path = "rect_right.png";
    std::string engine_path = "S2M2.engine";
    int warmup_times = 10;
    int test_times = 100;

    // 加载模型
    auto model = TRT::TRTInfer::create(engine_path, 4);

    cv::Mat left = cv::imread(left_path);
    cv::Mat right = cv::imread(right_path);

    if (left.empty() || right.empty())
    {
        std::cerr << "Error: Could not load input images!" << std::endl;
        return -1;
    }

    // 预处理
    std::vector<std::unordered_map<std::string, cv::Mat>> warmup_blobs;
    std::vector<std::unordered_map<std::string, cv::Mat>> test_blobs;
    for (int i = 0; i < warmup_times; i++) {
        warmup_blobs.emplace_back(preprocess(left, right));
    }
    for (int i = 0; i < test_times; i++) {
        test_blobs.emplace_back(preprocess(left, right));
    }

    // 预热
    std::cout << "\n=== Warmup ===" << std::endl;
    std::vector<std::future<std::unordered_map<std::string, cv::Mat>>> results;
    for (auto& blob : warmup_blobs) {
        results.emplace_back(model->PostQueue(blob));
    }
    for (auto& result : results) {
        result.get();
    }
    results.clear();

    // 推理测试
    std::cout << "\n=== Running inference ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    for (auto& blob : test_blobs) {
        results.emplace_back(model->PostQueue(blob));
    }
    cv::Mat output;
    for (auto& result : results) {
        output = result.get()["output_disp"];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() / test_times << " ms" << std::endl;

    cv::Mat dst, dst_conf;
    // post process
    postprocess(output.reshape(1, 736), dst);
    // 保存结果
    cv::imwrite("./demo/s2m2_disp.png", dst);
    std::cout << "Saved disparity to s2m2_disp.png" << std::endl;

    cv::imshow("disp", dst);
    cv::waitKey();
    return 0;
}
