#include "TRTinfer.h"
#include "benchmark.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <chrono>

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
     * @brief Preprocess stereo images for model input
     */
    std::unordered_map<std::string, cv::Mat> preprocess(const cv::Mat &left_rgb,
                                                        const cv::Mat &right_rgb,
                                                        bool normalize = false)
    {
        std::unordered_map<std::string, cv::Mat> input_blob;

        if (normalize)
        {
            input_blob["left_image"] = cv::dnn::blobFromImage(left_rgb, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false);
            input_blob["right_image"] = cv::dnn::blobFromImage(right_rgb, 1.0 / 255.0, cv::Size(), cv::Scalar(), false, false);
        }
        else
        {
            input_blob["left_image"] = cv::dnn::blobFromImage(left_rgb, 1.0, cv::Size(), cv::Scalar(), false, false);
            input_blob["right_image"] = cv::dnn::blobFromImage(right_rgb, 1.0, cv::Size(), cv::Scalar(), false, false);
        }

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

} // namespace LiteAnyStereo


void printUsage(const char *program_name)
{
    std::cout << "Usage: " << program_name << " [options]\n"
              << "\nOptions:\n"
              << "  --left_video <path>       Path to left video (default: left.mp4)\n"
              << "  --right_video <path>      Path to right video (default: right.mp4)\n"
              << "  --engine <path>           Path to TensorRT engine file (default: liteanystereo.engine)\n"
              << "  --output <path>           Output video path (default: disparity.mp4)\n"
              << "  --output_dir <path>       Output directory for frame-by-frame (default: ./output_frames)\n"
              << "  --fps <fps>               Output video fps (default: same as input)\n"
              << "  --codec <fourcc>          Video codec (default: mp4v)\n"
              << "  --start_frame <n>         Start frame index (default: 0)\n"
              << "  --num_frames <n>          Number of frames to process (default: all)\n"
              << "  --skip_frames <n>         Process every Nth frame (default: 1)\n"
              << "  --show_video              Show video preview during processing\n"
              << "  --save_frames             Save individual frames to output_dir\n"
              << "  --normalize               Normalize input to [0, 1] range\n"
              << "  --help, -h                Show this help message\n"
              << "\nExamples:\n"
              << "  " << program_name << " --left_video left.mp4 --right_video right.mp4\n"
              << "  " << program_name << " --left_video left.mp4 --right_video right.mp4 --output disparity.mp4 --show_video\n"
              << "  " << program_name << " --left_video left.mp4 --right_video right.mp4 --num_frames 100 --save_frames\n"
              << std::endl;
}


int main(int argc, char *argv[])
{
    // Default parameters
    std::string left_video_path = "left.mp4";
    std::string right_video_path = "right.mp4";
    std::string engine_file = "liteanystereo.engine";
    std::string output_video_path = "disparity.mp4";
    std::string output_dir = "./output_frames";
    double output_fps = 0;  // 0 means use input fps
    std::string codec = "mp4v";
    int start_frame = 0;
    int num_frames = -1;  // -1 means all frames
    int skip_frames = 1;   // Process every frame
    bool show_video = false;
    bool save_frames = false;
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
        else if (arg == "--left_video" && i + 1 < argc)
        {
            left_video_path = argv[++i];
        }
        else if (arg == "--right_video" && i + 1 < argc)
        {
            right_video_path = argv[++i];
        }
        else if (arg == "--engine" && i + 1 < argc)
        {
            engine_file = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc)
        {
            output_video_path = argv[++i];
        }
        else if (arg == "--output_dir" && i + 1 < argc)
        {
            output_dir = argv[++i];
        }
        else if (arg == "--fps" && i + 1 < argc)
        {
            output_fps = std::stod(argv[++i]);
        }
        else if (arg == "--codec" && i + 1 < argc)
        {
            codec = argv[++i];
        }
        else if (arg == "--start_frame" && i + 1 < argc)
        {
            start_frame = std::stoi(argv[++i]);
        }
        else if (arg == "--num_frames" && i + 1 < argc)
        {
            num_frames = std::stoi(argv[++i]);
        }
        else if (arg == "--skip_frames" && i + 1 < argc)
        {
            skip_frames = std::stoi(argv[++i]);
        }
        else if (arg == "--show_video")
        {
            show_video = true;
        }
        else if (arg == "--save_frames")
        {
            save_frames = true;
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

    // Print configuration
    std::cout << "========================================" << std::endl;
    std::cout << "LiteAnyStereo Video TensorRT Inference" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Left Video:   " << left_video_path << std::endl;
    std::cout << "  Right Video:  " << right_video_path << std::endl;
    std::cout << "  Engine:       " << engine_file << std::endl;
    std::cout << "  Output:       " << output_video_path << std::endl;
    std::cout << "  Start Frame:  " << start_frame << std::endl;
    if (num_frames > 0)
        std::cout << "  Num Frames:   " << num_frames << std::endl;
    if (skip_frames > 1)
        std::cout << "  Skip Frames:  " << skip_frames << std::endl;
    std::cout << "========================================" << std::endl;

    try
    {
        // Open video files
        cv::VideoCapture left_cap(left_video_path);
        cv::VideoCapture right_cap(right_video_path);

        if (!left_cap.isOpened())
        {
            throw std::runtime_error("Failed to open left video: " + left_video_path);
        }
        if (!right_cap.isOpened())
        {
            throw std::runtime_error("Failed to open right video: " + right_video_path);
        }

        // Get video properties
        int input_width = static_cast<int>(left_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int input_height = static_cast<int>(left_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double input_fps = left_cap.get(cv::CAP_PROP_FPS);
        int total_frames = static_cast<int>(left_cap.get(cv::CAP_PROP_FRAME_COUNT));

        // Verify right video has same properties
        int right_width = static_cast<int>(right_cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int right_height = static_cast<int>(right_cap.get(cv::CAP_PROP_FRAME_HEIGHT));

        if (input_width != right_width || input_height != right_height)
        {
            throw std::runtime_error("Left and right videos have different dimensions!");
        }

        // Set output fps
        if (output_fps <= 0)
        {
            output_fps = input_fps;
        }

        std::cout << "\nVideo Info:" << std::endl;
        std::cout << "  Resolution: " << input_width << "x" << input_height << std::endl;
        std::cout << "  FPS: " << input_fps << std::endl;
        std::cout << "  Total Frames: " << total_frames << std::endl;

        // Create output directory if saving frames
        if (save_frames)
        {
            std::string mkdir_cmd = "mkdir -p " + output_dir;
            system(mkdir_cmd.c_str());
        }

        // Load model
        std::cout << "\nLoading TensorRT engine..." << std::endl;
        auto model = TRTInfer::create(engine_file);
        std::cout << "Model loaded successfully!" << std::endl;

        // Setup video writer
        cv::VideoWriter writer;
        cv::Size frame_size(input_width, input_height);
        auto fourcc = cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]);
        writer.open(output_video_path, fourcc, output_fps, frame_size, true);

        if (!writer.isOpened())
        {
            throw std::runtime_error("Failed to create video writer for: " + output_video_path);
        }

        // Seek to start frame
        if (start_frame > 0)
        {
            left_cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
            right_cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
        }

        // Calculate number of frames to process
        int frames_to_process = (num_frames > 0) ? num_frames : (total_frames - start_frame);
        int processed_count = 0;
        int saved_count = 0;

        std::cout << "\nProcessing video..." << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Process frames
        for (int frame_idx = 0; frame_idx < frames_to_process; ++frame_idx)
        {
            // Read frames
            cv::Mat left_bgr, right_bgr;
            bool left_ok = left_cap.read(left_bgr);
            bool right_ok = right_cap.read(right_bgr);

            if (!left_ok || !right_ok)
            {
                std::cout << "\nEnd of video reached." << std::endl;
                break;
            }

            // Skip frames if needed
            if (frame_idx % skip_frames != 0)
            {
                continue;
            }

            // Convert BGR to RGB
            cv::Mat left_rgb, right_rgb;
            cv::cvtColor(left_bgr, left_rgb, cv::COLOR_BGR2RGB);
            cv::cvtColor(right_bgr, right_rgb, cv::COLOR_BGR2RGB);

            // Pad to multiple of 32
            cv::Mat left_rgb_pad, right_rgb_pad;
            LiteAnyStereo::PadInfo left_pad_info;
            std::tie(left_rgb_pad, left_pad_info) = LiteAnyStereo::padToMultiple(left_rgb, 32);
            std::tie(right_rgb_pad, std::ignore) = LiteAnyStereo::padToMultiple(right_rgb, 32);

            // Preprocess
            auto input_blob = LiteAnyStereo::preprocess(left_rgb_pad, right_rgb_pad, normalize_input);

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
            cv::Size original_size(left_rgb.cols, left_rgb.rows);
            cv::Mat disp_unpadded = LiteAnyStereo::unpad(disp_2d, left_pad_info, original_size);

            // Visualize disparity
            cv::Mat disp_vis = LiteAnyStereo::visualizeDisparity(disp_unpadded);

            // Write to output video
            writer.write(disp_vis);

            // Save individual frame if requested
            if (save_frames)
            {
                char filename[256];
                snprintf(filename, sizeof(filename), "%s/disparity_%06d.png", output_dir.c_str(), saved_count);
                cv::imwrite(filename, disp_vis);
                saved_count++;
            }

            processed_count++;

            // Print progress
            if (processed_count % 30 == 0)
            {
                float progress = 100.0f * processed_count / frames_to_process;
                std::cout << "  Progress: " << processed_count << "/" << frames_to_process
                          << " (" << progress << "%)" << std::endl;
            }

            // Show video preview
            if (show_video)
            {
                cv::Mat preview;
                cv::hconcat(left_bgr, disp_vis, preview);

                // Resize for display if too large
                if (preview.cols > 1280)
                {
                    float scale = 1280.0f / preview.cols;
                    cv::resize(preview, preview, cv::Size(), scale, scale);
                }

                cv::imshow("Left | Disparity", preview);

                // Exit on ESC key
                int key = cv::waitKey(1);
                if (key == 27)  // ESC
                {
                    std::cout << "\nProcessing interrupted by user." << std::endl;
                    break;
                }
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Cleanup
        left_cap.release();
        right_cap.release();
        writer.release();
        cv::destroyAllWindows();

        // Print statistics
        std::cout << "\n========================================" << std::endl;
        std::cout << "Processing Complete!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "  Frames Processed: " << processed_count << std::endl;
        if (save_frames)
            std::cout << "  Frames Saved:     " << saved_count << std::endl;
        std::cout << "  Total Time:       " << duration.count() / 1000.0 << " seconds" << std::endl;
        std::cout << "  Average FPS:      " << (processed_count * 1000.0 / duration.count()) << std::endl;
        std::cout << "  Output:           " << output_video_path << std::endl;
        std::cout << "========================================" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
