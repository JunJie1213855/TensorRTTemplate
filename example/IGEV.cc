#include "TRTinfer.h"
#include "benchmark.h"
#include <opencv2/opencv.hpp>

// 保持原始图像尺寸用于后处理还原
struct OriginalSize
{
    int height;
    int width;
};

/**
 * @brief 预处理: 加载并预处理图像
 * @param imfile 图像路径
 * @param target_size 目标尺寸 (width, height)
 * @return 预处理后的图像 (1x3xHxW)
 */
cv::Mat loadImage(const std::string &imfile)
{
    // 读取图像并转为 uint8
    cv::Mat img = cv::imread(imfile, cv::IMREAD_COLOR);
    if (img.empty())
    {
        throw std::runtime_error("Failed to load image: " + imfile);
    }
    return img;
}

/**
 * @brief 填充图像到 32 的倍数
 * @param img 输入图像 (1x3xHxW)
 * @param orig_size 输出原始尺寸
 * @return 填充后的图像
 */
cv::Mat padImage(const cv::Mat &img, OriginalSize &orig_size)
{
    // 获取当前尺寸
    int h = img.rows;
    int w = img.cols;

    // 保存原始尺寸
    orig_size.height = h;
    orig_size.width = w;

    // 计算需要填充的尺寸 (32 的倍数)
    int pad_h = (32 - h % 32) % 32;
    int pad_w = (32 - w % 32) % 32;

    if (pad_h == 0 && pad_w == 0)
    {
        return img.clone();
    }

    // 填充图像 (右侧和底部)
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0));

    return padded;
}

/**
 * @brief 移除填充，恢复原始尺寸
 * @param img 带填充的图像
 * @param orig_size 原始尺寸
 * @return 恢复后的图像
 */
cv::Mat unpadImage(const cv::Mat &img, const OriginalSize &orig_size)
{
    return img(cv::Rect(0, 0, orig_size.width, orig_size.height)).clone();
}

/**
 * @brief 预处理: 加载、resize、填充左右图像
 * @param left_path 左图像路径
 * @param right_path 右图像路径
 * @param orig_left 输出左图像原始尺寸
 * @param orig_right 输出右图像原始尺寸
 * @return 包含 left 和 right 的 map
 */
std::unordered_map<std::string, cv::Mat> preprocess(const std::string &left_path,
                                                    const std::string &right_path,
                                                    OriginalSize &orig_left,
                                                    OriginalSize &orig_right)
{
    // 目标尺寸 (与 Python 代码一致: 480x752)
    // cv::Size target_size(target_shape.w, target_shape.h);

    // 加载并预处理图像
    cv::Mat left_blob = loadImage(left_path);
    cv::Mat right_blob = loadImage(right_path);

    // 填充到 32 的倍数
    cv::Mat left_padded = padImage(left_blob, orig_left);
    cv::Mat right_padded = padImage(right_blob, orig_right);

    // HWC 转 CHW: 转换为 blob (1x3xHxW);BGR转RGB
    cv::Mat left_b = cv::dnn::blobFromImage(left_padded, 1.0, cv::Size(), cv::Scalar(), true, false);
    cv::Mat right_b = cv::dnn::blobFromImage(right_padded, 1.0, cv::Size(), cv::Scalar(), true, false);

    std::unordered_map<std::string, cv::Mat> input_blob;
    input_blob["left"] = left_b;
    input_blob["right"] = right_b;

    return input_blob;
}

/**
 * @brief 后处理: 移除 padding 并可视化视差图
 * @param disp 视差图 (1x1xHxW)
 * @param orig_size 原始尺寸
 * @param disp_vis 输出可视化图像
 */
void postprocess(const cv::Mat &disp, const OriginalSize &orig_size, cv::Mat &disp_vis)
{
    // 移除 padding，恢复原始尺寸
    cv::Mat disp_unpadded = unpadImage(disp, orig_size);

    // 去除 batch 和 channel 维度: HxW
    disp_unpadded = disp_unpadded.reshape(1, orig_size.height);

    // 归一化到 0-255 用于可视化
    double min_val, max_val;
    cv::minMaxLoc(disp_unpadded, &min_val, &max_val);

    cv::Mat disp_norm;
    if (max_val - min_val > 0)
    {
        disp_norm = ((disp_unpadded - min_val) / (max_val - min_val)) * 255;
    }
    else
    {
        disp_norm = cv::Mat::zeros(disp_unpadded.size(), CV_8U);
    }

    disp_norm.convertTo(disp_vis, CV_8U);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_JET);
}

int main(int argc, char *argv[])
{
    // 图像路径
    std::string left_path = "/root/code/C++/TensorRTTemplate/rect_left.png";
    std::string right_path = "/root/code/C++/TensorRTTemplate/rect_right.png";
    std::string engine_path = "/root/code/python/StereoMatch/StereoAlgorithms/IGEV-Stereo/igev_720_1280.engine";

    // 加载图像
    cv::Mat left = cv::imread(left_path);
    cv::Mat right = cv::imread(right_path);

    if (left.empty() || right.empty())
    {
        std::cerr << "Error: Could not load images" << std::endl;
        return -1;
    }

    OriginalSize orig_left, orig_right;

    // 加载模型 (使用工厂方法创建实例)
    auto model = TRTInfer::create(engine_path);

    // 输出尺寸预备
    std::vector<std::string> output_names = model->getOutputNames();
    TensorShape outputshape = model->getOutputShape(output_names[0]);

    // 预处理
    auto input_blob = preprocess(left_path, right_path, orig_left, orig_right);

    // 打印输入信息
    std::cout << "Input shape - left: " << input_blob["left"].size << " (d=" << input_blob["left"].size.p[0]
              << ", c=" << input_blob["left"].size.p[1]
              << ", h=" << input_blob["left"].size.p[2]
              << ", w=" << input_blob["left"].size.p[3] << ")" << std::endl;
    // 预热
    std::cout << "\n=== Warmup (10 iterations) ===" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        (*model)(input_blob);
    }

    // 推理
    std::cout << "\n=== Running inference ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto output_blob = (*model)(input_blob);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

    // 后处理
    cv::Mat disp_vis;
    postprocess(output_blob["disparity"].reshape(1, outputshape.h), orig_left, disp_vis);

    // 保存结果
    cv::imwrite("/root/code/C++/TensorRTTemplate/disp_output.png", disp_vis);
    std::cout << "Saved disparity visualization to disp_output.png" << std::endl;

    // 显示
    cv::imshow("Disparity", disp_vis);
    cv::waitKey(0);

    return 0;
}
