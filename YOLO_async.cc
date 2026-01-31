#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

namespace YOLOAsync
{
    struct Detection
    {
        int class_id{0};
        std::string className{};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
    };

    std::unordered_map<std::string, cv::Mat> preprocess(const cv::Mat &img)
    {
        cv::Mat imgc = img.clone();

        int target_w = 480;
        int target_h = 640;
        int img_w = imgc.cols;
        int img_h = imgc.rows;

        std::cout << "Preprocess: Original " << img_w << "x" << img_h << std::endl;

        // Letterbox: resize maintaining aspect ratio
        float scalew = target_w / img_w;
        float scaleh = target_h / img_h;
        // resize 
        cv::Mat blob = cv::dnn::blobFromImage(imgc, 1 / 255.f, cv::Size(target_w, target_h), cv::Scalar(), true, false);
        std::unordered_map<std::string, cv::Mat> input_blob;
        input_blob["images"] = blob;
        return input_blob;
    }

    cv::Mat postprocess(const cv::Mat &output_blob, const cv::Mat &img, const cv::Size2f &scale)
    {
        cv::Mat imgc = img.clone();
        cv::Mat output_blobc = output_blob.clone().reshape(1, 84);
        output_blobc.convertTo(output_blobc, CV_32F);
        cv::transpose(output_blobc, output_blobc);

        std::vector<cv::Rect> boxes;
        std::vector<float> scores_classs;
        std::vector<int> indices;

        float confidenceThreshold = 0.5; // 进一步降低置信度阈值
        float nmsThreshold = 0.5;        // 降低NMS阈值

        for (int i = 0; i < output_blobc.rows; i++)
        {
            float *classes_scores = (float *)output_blobc.row(i).data + 4;
            cv::Mat scores(cv::Size(80, 1), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > confidenceThreshold)
            {
                scores_classs.push_back(maxClassScore);
                indices.push_back(class_id.x);
                float x = output_blobc.at<float>(i, 0);
                float y = output_blobc.at<float>(i, 1);
                float w = output_blobc.at<float>(i, 2);
                float h = output_blobc.at<float>(i, 3);
                int left = int((x - 0.5 * w) * scale.width);
                int top = int((y - 0.5 * h) * scale.height);

                int width = int(w * scale.width);
                int height = int(h * scale.height);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, scores_classs, confidenceThreshold, nmsThreshold, nms_result);
        for (unsigned long i = 0; i < nms_result.size(); ++i)
        {
            int idx = nms_result[i];

            Detection result;
            result.class_id = indices[idx];
            result.confidence = scores_classs[idx];

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = cv::Scalar(dis(gen),
                                      dis(gen),
                                      dis(gen));

            result.className = std::to_string(indices[idx]);
            result.box = boxes[idx];
            cv::rectangle(imgc, boxes[idx], result.color, 4);
            cv::putText(imgc, result.className, cv::Point(boxes[idx].x, boxes[idx].y),
                        cv::FONT_HERSHEY_COMPLEX, 1.0, result.color);
        }
        return imgc;
    }
}

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    std::cout << "=== TensorRT Async Inference Example ===" << std::endl;
    std::cout << "Input Engine PAth : /root/code/C++/TensorRTTemplate/yolov8n.engine" << std::endl;
    std::cout << "Initializing model with 4 streams..." << std::endl;
    // Check for display environment
    bool has_display = (getenv("DISPLAY") != nullptr) || (getenv("WAYLAND_DISPLAY") != nullptr);
    if (!has_display)
    {
        std::cout << "No display environment detected. Running in headless mode." << std::endl;
    }

    TRTInfer model("/root/code/C++/TensorRTTemplate/data/yolov8n.engine", 4, true);

    cv::Mat image = cv::imread("/root/code/C++/TensorRTTemplate/demo/bus.jpg");
    if (image.empty())
    {
        std::cerr << "Failed to load image" << std::endl;
        return -1;
    }

    float scalew = static_cast<float>(image.size().width) / 480.f;
    float scaleh = static_cast<float>(image.size().height) / 640.f;
    cv::Size2f scale_factor(scalew, scaleh);

    auto input_blob = YOLOAsync::preprocess(image);

    std::cout << "\n1. Testing synchronous inference..." << std::endl;
    auto start_sync = std::chrono::high_resolution_clock::now();
    auto output_sync = model(input_blob);
    auto end_sync = std::chrono::high_resolution_clock::now();
    auto duration_sync = std::chrono::duration_cast<std::chrono::milliseconds>(end_sync - start_sync).count();
    std::cout << "Synchronous inference time: " << duration_sync << " ms" << std::endl;

    cv::Mat result_sync = YOLOAsync::postprocess(output_sync["output0"], image, scale_factor);
    if (has_display)
    {
        cv::imshow("Synchronous Result", result_sync);
        cv::waitKey(100);
    }
    else
    {
        std::cout << "Synchronous inference completed" << std::endl;
    }

    // Save result for verification
    std::cout << "Saving result_sync.png..." << std::endl;
    cv::imwrite("result_sync.png", result_sync);

    // Count detections
    int sync_detections = 0;
    for (int y = 0; y < result_sync.rows; ++y)
    {
        for (int x = 0; x < result_sync.cols; ++x)
        {
            cv::Vec3b pixel = result_sync.at<cv::Vec3b>(y, x);
            if (pixel != cv::Vec3b(0, 0, 0))
            {
                sync_detections++;
            }
        }
    }
    std::cout << "Sync mode: Found " << sync_detections << " non-black pixels (detections drawn)" << std::endl;

    std::cout << "\n2. Testing async inference with Future..." << std::endl;
    auto start_async = std::chrono::high_resolution_clock::now();
    auto future = model.infer_async(input_blob);
    auto output_async = future.get();
    auto end_async = std::chrono::high_resolution_clock::now();
    auto duration_async = std::chrono::duration_cast<std::chrono::milliseconds>(end_async - start_async).count();
    std::cout << "Async inference time: " << duration_async << " ms" << std::endl;

    cv::Mat result_async = YOLOAsync::postprocess(output_async["output0"], image, scale_factor);
    std::cout << "Saving result_async.png..." << std::endl;
    cv::imwrite("result_async.png", result_async);

    // Count detections
    int async_detections = 0;
    for (int y = 0; y < result_async.rows; ++y)
    {
        for (int x = 0; x < result_async.cols; ++x)
        {
            cv::Vec3b pixel = result_async.at<cv::Vec3b>(y, x);
            if (pixel != cv::Vec3b(0, 0, 0))
            {
                async_detections++;
            }
        }
    }
    std::cout << "Async mode: Found " << async_detections << " non-black pixels (detections drawn)" << std::endl;

    if (has_display)
    {
        cv::imshow("Async Result", result_async);
        cv::waitKey(100);
    }
    else
    {
        std::cout << "Async inference completed" << std::endl;
    }
    cv::destroyAllWindows();
    std::cout << "\n3. Testing concurrent inference..." << std::endl;
    const int num_concurrent = 8;
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
    std::cout << "Completed " << num_concurrent << " concurrent inferences in "
              << duration_concurrent << " ms" << std::endl;
    std::cout << "Average time per inference: " << (duration_concurrent / num_concurrent) << " ms" << std::endl;

    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Synchronous: " << duration_sync << " ms" << std::endl;
    std::cout << "Async (Future): " << duration_async << " ms" << std::endl;
    std::cout << "Concurrent (8 inferences): " << duration_concurrent << " ms" << std::endl;
    std::cout << "Average time per inference: " << (duration_concurrent / num_concurrent) << " ms" << std::endl;
    std::cout << "Performance improvement: " << (duration_sync / (duration_concurrent / num_concurrent)) << "x" << std::endl;
    std::cout << "Model has " << model.num_streams() << " CUDA streams" << std::endl;

    std::cout << "\nCleaning up..." << std::endl;

    return 0;
}
