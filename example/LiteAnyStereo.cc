#include "TRTinfer.h"
#include "benchmark.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
using namespace TRT;
namespace LiteAnyStereo
{
    // Padding information structure
    struct PadInfo
    {
        int top, bottom, left, right;
    };

    /**
     * @brief Pad image to multiple of specified value
     */
    std::pair<cv::Mat, PadInfo> padToMultiple(const cv::Mat &img, int multiple = 32)
    {
        int H = img.rows;
        int W = img.cols;

        int pad_h = (multiple - H % multiple) % multiple;
        int pad_w = (multiple - W % multiple) % multiple;

        PadInfo pad_info;
        pad_info.top = pad_h / 2;
        pad_info.bottom = pad_h - pad_info.top;
        pad_info.left = pad_w / 2;
        pad_info.right = pad_w - pad_info.left;

        cv::Mat img_padded;
        cv::copyMakeBorder(img, img_padded, pad_info.top, pad_info.bottom,
                           pad_info.left, pad_info.right, cv::BORDER_REFLECT_101);

        return {img_padded, pad_info};
    }

    /**
     * @brief Remove padding from image
     */
    cv::Mat unpad(const cv::Mat &img, const PadInfo &pad_info, const cv::Size &original_size)
    {
        if (img.rows == original_size.height && img.cols == original_size.width)
        {
            return img;
        }

        if (img.rows >= original_size.height && img.cols >= original_size.width)
        {
            int H = img.rows;
            int W = img.cols;
            return img(cv::Rect(pad_info.left, pad_info.top,
                                W - pad_info.left - pad_info.right,
                                H - pad_info.top - pad_info.bottom));
        }

        cv::Mat result;
        cv::resize(img, result, original_size);
        return result;
    }

    /**
     * @brief Load and preprocess image
     */
    std::tuple<cv::Mat, cv::Mat, cv::Size> loadImage(const std::string &img_path,
                                                     const cv::Size &target_size = cv::Size())
    {
        cv::Mat img = cv::imread(img_path);
        if (img.empty())
        {
            throw std::runtime_error("Failed to load image: " + img_path);
        }

        cv::Size original_size(img.cols, img.rows);
        cv::Mat img_bgr = img.clone();
        cv::Mat img_rgb;

        if (target_size.width > 0 && target_size.height > 0)
        {
            cv::resize(img, img, target_size);
        }

        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

        return {img_bgr, img_rgb, original_size};
    }

    /**
     * @brief Preprocess stereo images for model input
     */
    std::unordered_map<std::string, cv::Mat> preprocess(const cv::Mat &left_rgb,
                                                        const cv::Mat &right_rgb,
                                                        bool normalize = false)
    {
        std::unordered_map<std::string, cv::Mat> input_blob;

        input_blob["left_image"] = cv::dnn::blobFromImage(left_rgb, 1.0, cv::Size(), cv::Scalar(), false, false);
        input_blob["right_image"] = cv::dnn::blobFromImage(right_rgb, 1.0, cv::Size(), cv::Scalar(), false, false);

        return input_blob;
    }

    /**
     * @brief Visualize disparity map with colormap
     */
    cv::Mat visualizeDisparity(const cv::Mat &disp)
    {
        double min_val, max_val;
        cv::minMaxLoc(disp, &min_val, &max_val);

        cv::Mat vis;
        if (max_val > min_val)
        {
            vis = ((disp - min_val) / (max_val - min_val));
        }
        else
        {
            vis = cv::Mat::zeros(disp.size(), CV_32F);
        }

        vis.convertTo(vis, CV_8U, 255.0);

        cv::Mat vis_color;
        cv::applyColorMap(vis, vis_color, cv::COLORMAP_TURBO);

        return vis_color;
    }

    /**
     * @brief Concatenate images horizontally for visualization
     */
    cv::Mat hconcat(const std::vector<cv::Mat> &images)
    {
        if (images.empty())
            return cv::Mat();

        std::vector<cv::Mat> valid_images;
        for (const auto &img : images)
        {
            if (!img.empty())
                valid_images.push_back(img);
        }

        if (valid_images.empty())
            return cv::Mat();

        if (valid_images.size() == 1)
            return valid_images[0];

        cv::Mat result;
        cv::hconcat(valid_images, result);
        return result;
    }

} // namespace LiteAnyStereo

void printUsage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "\nOptions:\n"
              << "  --left_img <path>     Path to left rectified image (default: rect_left.png)\n"
              << "  --right_img <path>    Path to right rectified image (default: rect_right.png)\n"
              << "  --engine <path>       Path to TensorRT engine file (default: liteanystereo.engine)\n"
              << "  --output_dir <path>   Output directory (default: ./output)\n"
              << "  --target_size <h,w>   Target size for model input (default: original size)\n"
              << "  --benchmark           Enable benchmark mode\n"
              << "  --warmup <n>          Warmup iterations (default: 5)\n"
              << "  --runs <n>            Benchmark runs (default: 50)\n"
              << "  --no_display          Disable display window\n"
              << "  --debug, -d           Enable debug mode (save intermediate outputs)\n"
              << "  --normalize           Normalize input to [0, 1] range (default: [0, 255])\n"
              << "  --help                Show this help message\n"
              << "\nExample:\n"
              << "  " << program_name << " --left_img left.png --right_img right.png --engine model.engine\n"
              << std::endl;
}

int main(int argc, char *argv[])
{
    // Default parameters
    std::string left_img_path = "rect_left.png";
    std::string right_img_path = "rect_right.png";
    std::string engine_file = "liteanystereo.engine";
    std::string output_dir = "./output";
    cv::Size target_size;
    bool benchmark_mode = false;
    int warmup_runs = 5;
    int benchmark_runs = 50;
    bool enable_display = true;
    bool debug_mode = false;
    bool normalize_input = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            return 0;
        }
        else if (arg == "--left_img" && i + 1 < argc)
        {
            left_img_path = argv[++i];
        }
        else if (arg == "--right_img" && i + 1 < argc)
        {
            right_img_path = argv[++i];
        }
        else if (arg == "--engine" && i + 1 < argc)
        {
            engine_file = argv[++i];
        }
        else if (arg == "--output_dir" && i + 1 < argc)
        {
            output_dir = argv[++i];
        }
        else if (arg == "--target_size" && i + 1 < argc)
        {
            std::string size_str = argv[++i];
            size_t comma_pos = size_str.find(',');
            if (comma_pos != std::string::npos)
            {
                int h = std::stoi(size_str.substr(0, comma_pos));
                int w = std::stoi(size_str.substr(comma_pos + 1));
                target_size = cv::Size(w, h);
            }
        }
        else if (arg == "--benchmark" || arg == "-b")
        {
            benchmark_mode = true;
        }
        else if (arg == "--warmup" && i + 1 < argc)
        {
            warmup_runs = std::stoi(argv[++i]);
        }
        else if (arg == "--runs" && i + 1 < argc)
        {
            benchmark_runs = std::stoi(argv[++i]);
        }
        else if (arg == "--no_display")
        {
            enable_display = false;
        }
        else if (arg == "--debug" || arg == "-d")
        {
            debug_mode = true;
        }
        else if (arg == "--normalize")
        {
            normalize_input = true;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    // Create output directory
    std::string mkdir_cmd = "mkdir -p " + output_dir;
    system(mkdir_cmd.c_str());

    // Print configuration
    std::cout << "========================================" << std::endl;
    std::cout << "LiteAnyStereo TensorRT Inference" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Left Image:   " << left_img_path << std::endl;
    std::cout << "  Right Image:  " << right_img_path << std::endl;
    std::cout << "  Engine:       " << engine_file << std::endl;
    std::cout << "  Output Dir:   " << output_dir << std::endl;
    std::cout << "  Benchmark:    " << (benchmark_mode ? "Yes" : "No") << std::endl;
    if (target_size.width > 0)
    {
        std::cout << "  Target Size:  " << target_size.height << "x" << target_size.width << std::endl;
    }
    std::cout << "========================================" << std::endl;

    try
    {
        // Load images
        std::cout << "\nLoading images..." << std::endl;
        cv::Mat left_bgr, left_rgb, right_bgr, right_rgb;
        cv::Size left_original_size, right_original_size;

        std::tie(left_bgr, left_rgb, left_original_size) = LiteAnyStereo::loadImage(left_img_path, target_size);
        std::tie(right_bgr, right_rgb, right_original_size) = LiteAnyStereo::loadImage(right_img_path, target_size);

        std::cout << "  Left image size:  " << left_rgb.cols << "x" << left_rgb.rows << std::endl;
        std::cout << "  Right image size: " << right_rgb.cols << "x" << right_rgb.rows << std::endl;

        // Pad to multiple of 32
        std::cout << "\nPadding images to multiple of 32..." << std::endl;
        cv::Mat left_rgb_pad, right_rgb_pad;
        LiteAnyStereo::PadInfo left_pad_info;

        std::tie(left_rgb_pad, left_pad_info) = LiteAnyStereo::padToMultiple(left_rgb, 32);
        std::tie(right_rgb_pad, std::ignore) = LiteAnyStereo::padToMultiple(right_rgb, 32);

        std::cout << "  Padded size: " << left_rgb_pad.cols << "x" << left_rgb_pad.rows << std::endl;

        // Preprocess
        auto input_blob = LiteAnyStereo::preprocess(left_rgb_pad, right_rgb_pad, normalize_input);

        // Load model
        std::cout << "\nLoading TensorRT engine..." << std::endl;
        auto model = TRT::TRTInfer::create(engine_file, 4);
        std::cout << "Model loaded successfully!" << std::endl;

        // Benchmark mode
        if (benchmark_mode)
        {
            // 预热
            std::cout << "\n=== Warmup ===" << std::endl;
            std::vector<std::future<std::unordered_map<std::string, cv::Mat>>> warmup_results;
            for (int i = 0; i < warmup_runs; i++)
            {
                warmup_results.emplace_back(model->PostQueue(input_blob));
            }
            for (auto &r : warmup_results)
            {
                r.get();
            }

            // 推理测试
            std::cout << "\n=== Running inference ===" << std::endl;
            std::vector<std::future<std::unordered_map<std::string, cv::Mat>>> results;
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < benchmark_runs; i++)
            {
                results.emplace_back(model->PostQueue(input_blob));
            }
            cv::Mat output;
            for (auto &r : results)
            {
                output = r.get().begin()->second;
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Inference time: " << duration.count() / benchmark_runs << " ms" << std::endl;
        }

        // Inference
        auto output_blob = (*model)(input_blob);

        // Get disparity output
        cv::Mat disparity;
        if (output_blob.find("disparity") != output_blob.end())
            disparity = output_blob["disparity"];
        else if (output_blob.find("output") != output_blob.end())
            disparity = output_blob["output"];
        else
            disparity = output_blob.begin()->second;

        // Reshape to 2D
        int model_h = disparity.size[2];
        int model_w = disparity.size[3];
        cv::Mat disp_2d(model_h, model_w, CV_32F, disparity.ptr<float>());

        // Unpad to original size
        cv::Size original_size_before_padding(left_rgb.cols, left_rgb.rows);
        cv::Mat disp_unpadded = LiteAnyStereo::unpad(disp_2d, left_pad_info, original_size_before_padding);

        // Print disparity range
        double min_disp, max_disp;
        cv::minMaxLoc(disp_unpadded, &min_disp, &max_disp);
        std::cout << "  Disparity range: [" << min_disp << ", " << max_disp << "]" << std::endl;

        // Visualize disparity
        cv::Mat disp_vis = LiteAnyStereo::visualizeDisparity(disp_unpadded);

        // Save outputs
        std::cout << "\nSaving outputs..." << std::endl;
        cv::imwrite(output_dir + "/disparity_color.png", disp_vis);
        std::cout << "  Saved colored disparity to: " << output_dir << "/disparity_color.png" << std::endl;

        // Display
        if (enable_display)
        {
            cv::Mat stereo_vis = LiteAnyStereo::hconcat({left_bgr, right_bgr});

            int display_width = 1280;
            if (stereo_vis.cols > display_width)
            {
                float scale = static_cast<float>(display_width) / stereo_vis.cols;
                cv::resize(stereo_vis, stereo_vis, cv::Size(), scale, scale);
                cv::resize(disp_vis, disp_vis, cv::Size(), scale, scale);
            }

            cv::imshow("Stereo Images (Left | Right)", stereo_vis);
            cv::imshow("Disparity", disp_vis);

            std::cout << "\nPress any key to exit..." << std::endl;
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

        std::cout << "\nDone!" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}