#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
#include <random>
namespace SegFormer
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
        input_blob["input"] = blob; // 根据你的模型输入名称调整
        return input_blob;
    }
    cv::Mat visualizeSegmentation(const cv::Mat &segMap)
    {
        // 定义颜色调色板 (BGR格式)
        std::vector<cv::Vec3b> colorMap = {
            {0, 0, 0},       // 0: 黑色 - 背景
            {0, 0, 255},     // 1: 红色
            {0, 255, 0},     // 2: 绿色
            {255, 0, 0},     // 3: 蓝色
            {0, 255, 255},   // 4: 黄色
            {255, 0, 255},   // 5: 品红
            {255, 255, 0},   // 6: 青色
            {128, 0, 128},   // 7: 紫色
            {255, 128, 0},   // 8: 橙色
            {0, 128, 255},   // 9: 天蓝色
            {128, 255, 0},   // 10: 青绿色
            {255, 0, 128},   // 11: 玫红色
            {128, 128, 0},   // 12: 橄榄绿
            {0, 128, 128},   // 13: 青绿色
            {128, 0, 0},     // 14: 深红色
            {0, 0, 128},     // 15: 深蓝色
            {255, 255, 255}, // 16: 白色
            {192, 192, 192}, // 17: 银灰色
            {128, 128, 128}, // 18: 灰色
            {255, 165, 0}    // 19: 橙红色
        };

        cv::Mat coloredMap(segMap.rows, segMap.cols, CV_8UC3);

        for (int i = 0; i < segMap.rows; i++)
        {
            for (int j = 0; j < segMap.cols; j++)
            {
                int label = segMap.at<uchar>(i, j);

                // 如果类别超过调色板大小，使用循环
                label = label % colorMap.size();
                coloredMap.at<cv::Vec3b>(i, j) = colorMap[label];
            }
        }

        return coloredMap;
    }
}

int main(int argc, char *argv[])
{
    // model
    TRTInfer model("segformer.engine");
    // image
    cv::Mat image = cv::imread("/root/code/python/ImageSegment/mmsegmentation/demo/demo.png");

    // preprocess
    auto input_blob = SegFormer::preprocess(image);

    // inference
    auto output_blob = model(input_blob);
    // reshape 1x1x512x512 to 512 x 512
    cv::Mat mat2d = output_blob["output"].reshape(0, 512).clone();

    // convert unsigned char type
    mat2d.convertTo(mat2d, CV_8U);
    // cv::resize(mat2d, mat2d, image.size(), 0, 0, cv::INTER_NEAREST);

    cv::Mat viz, mask, image_512;
    // apply color map
    viz = SegFormer::visualizeSegmentation(mat2d);
    // for mask

    cv::resize(image, image_512, cv::Size(512, 512));
    cv::addWeighted(image_512, 0.7, viz, 0.3, 0, mask);

    // show result
    cv::imshow("output", viz);
    cv::imshow("mask", mask);
    cv::waitKey();

    return 1;
}