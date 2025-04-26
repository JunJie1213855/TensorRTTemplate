#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>
#include "TRTinfer.h"
#include <chrono>

// ImageNet mean and std for normalization
std::vector<double> mean = {0.406, 0.456, 0.485};
std::vector<double> std_ = {0.225, 0.224, 0.229};

// Function to load and preprocess stereo images
std::pair<cv::Mat, cv::Mat> preprocessStereoImages(const std::string &left_path, const std::string &right_path, int maxdisp)
{
    // Load images
    cv::Mat left_img = cv::imread(left_path);
    cv::Mat right_img = cv::imread(right_path);

    if (left_img.empty() || right_img.empty())
    {
        std::cerr << "Error: Could not read image files" << std::endl;
        return {cv::Mat(), cv::Mat()};
    }

    // Get original dimensions
    int w = left_img.cols;
    int h = left_img.rows;

    // Calculate padded dimensions (multiple of 32)
    int wi = ((w / 32) + 1) * 32;
    int hi = ((h / 32) + 1) * 32;

    // Create padded images
    cv::Mat left_padded = cv::Mat::zeros(hi, wi, left_img.type());
    cv::Mat right_padded = cv::Mat::zeros(hi, wi, right_img.type());

    // Copy original images to bottom-right of padded images
    // This mimics the PIL crop operation in the Python code
    cv::Rect roi(wi - w, hi - h, w, h);
    left_img.copyTo(left_padded(roi));
    right_img.copyTo(right_padded(roi));

    // Convert to float32 and normalize to 0-1
    left_padded.convertTo(left_padded, CV_32FC3, 1.0 / 255.0);
    right_padded.convertTo(right_padded, CV_32FC3, 1.0 / 255.0);

    // Normalize with ImageNet mean and std
    std::vector<cv::Mat> left_channels, right_channels;
    cv::split(left_padded, left_channels);
    cv::split(right_padded, right_channels);

    for (int c = 0; c < 3; c++)
    {
        // Normalization formula: (pixel - mean) / std
        left_channels[c] = (left_channels[c] - mean[c]) / std_[c];
        right_channels[c] = (right_channels[c] - mean[c]) / std_[c];
    }

    cv::merge(left_channels, left_padded);
    cv::merge(right_channels, right_padded);

    return {left_padded, right_padded};
}

// Function to save disparity as a colored visualization
void saveDisparityVisualization(const cv::Mat &disparity, const std::string &output_path)
{
    // Normalize disparity for visualization
    cv::Mat disp_normalized;
    cv::normalize(disparity, disp_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Apply colormap (jet)
    cv::Mat disp_color;
    cv::applyColorMap(disp_normalized, disp_color, cv::COLORMAP_JET);

    // Save to file
    cv::imwrite(output_path, disp_color);
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
std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &left, cv::Mat &right)
{
    std::unordered_map<std::string, cv::Mat> input_blob;
    // convert the H X W X 3 to 3 X H X W , and bgr to rgb
    input_blob["left_image"] = cv::dnn::blobFromImage(left, 1.0, cv::Size(640, 480), cv::Scalar(), true, false);
    input_blob["right_image"] = cv::dnn::blobFromImage(right, 1.0, cv::Size(640, 480), cv::Scalar(), true, false);
    return input_blob;
}

// Example of how to use these functions in your main code
int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cout << "Usage : [Exec] [model_path] [left_path] [right_path] [output_dir](default .)" << std::endl;
        return 0;
    }
    std::string left_img_path(argv[2]);
    std::string right_img_path(argv[3]);
    std::string output_dir = ".";

    // Preprocess images
    auto [left_processed, right_processed] = preprocessStereoImages(left_img_path, right_img_path, 256);

    // preprocess
    auto input_blob = preprocess(left_processed, right_processed);
    // model
    TRTInfer model(argv[1]);
    auto start = std::chrono::high_resolution_clock::now();
    // inference
    for (size_t i = 0; i < 25; i++)
        auto output_blob = model(input_blob);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << " the cost time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 25 << " ms" << std::endl;
    auto output_blob = model(input_blob);
    cv::Mat dst;
    // post process
    postprocess(output_blob["disparity"].reshape(1, 480), dst);
    cv::imwrite("disp.png",dst);
    cv::imshow("disp", dst);
    cv::waitKey();

    return 0;
}