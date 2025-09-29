#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
#include <random>
namespace FCN
{
    std::unordered_map<std::string, cv::Mat> preprocess(const cv::Mat &img)
    {
        cv::Mat imgc = img.clone();
        
        // Resize to target size (256, 256) as specified in config
        if (imgc.size() != cv::Size(256, 256))
        {
            cv::resize(imgc, imgc, cv::Size(256, 256));
        }
        
        // Convert BGR to RGB as specified in config (bgr_to_rgb=True)
        cv::Mat rgb_img;
        cv::cvtColor(imgc, rgb_img, cv::COLOR_BGR2RGB);
        
        // Convert to float32
        cv::Mat float_img;
        rgb_img.convertTo(float_img, CV_32F);
        
        // Normalize with mean and std from config
        // mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        cv::Scalar mean(123.675, 116.28, 103.53);
        cv::Scalar std_dev(58.395, 57.12, 57.375);
        
        // Apply normalization: (pixel - mean) / std
        cv::Mat normalized_img;
        cv::subtract(float_img, mean, normalized_img);
        cv::divide(normalized_img, std_dev, normalized_img);
        
        // Create blob (NCHW format: batch_size=1, channels=3, height=256, width=256)
        cv::Mat blob = cv::dnn::blobFromImage(normalized_img, 1.0, cv::Size(512, 512), cv::Scalar(), false, false);
        
        std::unordered_map<std::string, cv::Mat> input_blob;
        input_blob["input"] = blob;  // 根据你的模型输入名称调整
        return input_blob;
    }
}



int main(int argc, char *argv[])
{
    // model
    TRTInfer model("fcn.engine");
    // image
    cv::Mat image = cv::imread("demo/bus.jpg");

    // preprocess
    auto input_blob = FCN::preprocess(image);

    // inference
    auto output_blob = model(input_blob);
    // reshape 1x1x512x512 to 512 x 512
    cv::Mat mat2d = output_blob["output"].reshape(0, 512).clone();

    // norm 0 ~ 255
    double maxval, minval;
    cv::minMaxIdx(mat2d, &minval, &maxval);
    mat2d = (mat2d - minval)/ (maxval - minval) * 255;

    // convert unsigned char type
    mat2d.convertTo(mat2d, CV_8U);
    cv::resize(mat2d, mat2d, image.size(), 0, 0, cv::INTER_NEAREST);

    cv::Mat viz, mask;
    // apply color map
    cv::applyColorMap(mat2d, viz, cv::COLORMAP_INFERNO);

    // for mask
    cv::addWeighted(image, 0.7, viz, 0.3, 0, mask);

    // show result
    cv::imshow("output", viz);
    cv::imshow("mask", mask);
    cv::waitKey();


    return 1;
}