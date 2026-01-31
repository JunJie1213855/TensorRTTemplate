#include "async_infer.h"
#include "utility.h"
#include <iostream>
#include <stdexcept>
#include <thread>

namespace inference
{

    template <typename OutputType>
    AsyncInfer<OutputType>::AsyncInfer(nvinfer1::IExecutionContext *context,
                                       nvinfer1::ICudaEngine *engine,
                                       std::shared_ptr<StreamPool> stream_pool,
                                       std::shared_ptr<MemoryPool> memory_pool,
                                       const std::vector<std::string> &input_names,
                                       const std::vector<std::string> &output_names,
                                       const std::unordered_map<std::string, size_t> &input_sizes,
                                       const std::unordered_map<std::string, size_t> &output_sizes,
                                       const std::unordered_map<std::string, std::vector<int>> &output_shapes)
        : context_(context),
          engine_(engine),
          stream_pool_(stream_pool),
          memory_pool_(memory_pool),
          input_names_(input_names),
          output_names_(output_names),
          input_sizes_(input_sizes),
          output_sizes_(output_sizes),
          output_shapes_(output_shapes),
          running_(true),
          task_counter_(0)
    {

        std::cout << "AsyncInfer initialized with " << stream_pool_->num_streams()
                  << " streams" << std::endl;
    }

    template <typename OutputType>
    AsyncInfer<OutputType>::~AsyncInfer()
    {
        shutdown();
    }

    template <typename OutputType>
    std::future<OutputType> AsyncInfer<OutputType>::infer_async(const InputType &input_blob)
    {
        auto stream = stream_pool_->acquire();
        int task_id = task_counter_++;

        auto task = std::make_shared<InferenceTask<OutputType>>(task_id, stream);
        task->set_status(TaskStatus::Running);

        auto future = task->get_future();

        try
        {
            auto output = infer_impl(input_blob, stream);
            task->set_output(output);
            task->set_status(TaskStatus::Completed);
        }
        catch (const std::exception &e)
        {
            task->set_exception(e);
        }

        stream_pool_->release(stream);

        return future;
    }

    template <typename OutputType>
    std::future<OutputType> AsyncInfer<OutputType>::infer_async(const InputMatType &input_blob)
    {
        auto stream = stream_pool_->acquire();
        int task_id = task_counter_++;

        auto task = std::make_shared<InferenceTask<OutputType>>(task_id, stream);
        task->set_status(TaskStatus::Running);

        auto future = task->get_future();

        try
        {
            auto output = infer_impl_mat(input_blob, stream);
            task->set_output(output);
            task->set_status(TaskStatus::Completed);
        }
        catch (const std::exception &e)
        {
            task->set_exception(e);
        }

        stream_pool_->release(stream);

        return future;
    }

    template <typename OutputType>
    void AsyncInfer<OutputType>::infer_with_callback(const InputType &input_blob, Callback callback)
    {
        auto future = infer_async(input_blob);

        std::thread([future = std::move(future), callback]() mutable
                    {
        try {
            auto output = future.get();
            if (callback) {
                callback(output);
            }
        } catch (const std::exception& e) {
            std::cerr << "Callback inference failed: " << e.what() << std::endl;
        } })
            .detach();
    }

    template <typename OutputType>
    void AsyncInfer<OutputType>::infer_with_callback(const InputMatType &input_blob, Callback callback)
    {
        auto future = infer_async(input_blob);

        std::thread([future = std::move(future), callback]() mutable
                    {
        try {
            auto output = future.get();
            if (callback) {
                callback(output);
            }
        } catch (const std::exception& e) {
            std::cerr << "Callback inference failed: " << e.what() << std::endl;
        } })
            .detach();
    }

    template <typename OutputType>
    size_t AsyncInfer<OutputType>::pending_count() const
    {
        return static_cast<size_t>(task_counter_.load());
    }

    template <typename OutputType>
    void AsyncInfer<OutputType>::wait_all()
    {
        stream_pool_->synchronize_all();
    }

    template <typename OutputType>
    void AsyncInfer<OutputType>::shutdown()
    {
        if (running_.load())
        {
            running_.store(false);
            wait_all();
        }
    }

    template <typename OutputType>
    OutputType AsyncInfer<OutputType>::infer_impl(const InputType &input_blob,
                                                  std::shared_ptr<StreamHandle> stream)
    {
        copy_input_to_device(input_blob, stream);

        int stream_id = stream->id();
        for (const auto &name : output_names_)
        {
            void *cuda_ptr = memory_pool_->get_output_binding(name, stream_id);
            if (!cuda_ptr)
            {
                std::cerr << "Failed to get output binding for " << name << std::endl;
                throw std::runtime_error("Failed to get output binding");
            }
            context_->setOutputTensorAddress(name.c_str(), cuda_ptr);
        }

        context_->enqueueV3(stream->get());

        return copy_output_from_device(stream);
    }

    template <typename OutputType>
    OutputType AsyncInfer<OutputType>::infer_impl_mat(const InputMatType &input_blob,
                                                      std::shared_ptr<StreamHandle> stream)
    {
        copy_input_to_device(input_blob, stream);

        int stream_id = stream->id();
        for (const auto &name : output_names_)
        {
            void *cuda_ptr = memory_pool_->get_output_binding(name, stream_id);
            if (!cuda_ptr)
            {
                std::cerr << "Failed to get output binding for " << name << std::endl;
                throw std::runtime_error("Failed to get output binding");
            }
            context_->setOutputTensorAddress(name.c_str(), cuda_ptr);
        }

        context_->enqueueV3(stream->get());

        return copy_output_from_device(stream);
    }

    template <typename OutputType>
    void AsyncInfer<OutputType>::copy_input_to_device(const InputType &input_blob,
                                                      std::shared_ptr<StreamHandle> stream)
    {
        int stream_id = stream->id();

        for (const auto &[name, cpu_ptr] : input_blob)
        {
            void *cuda_ptr = memory_pool_->get_input_binding(name, stream_id);
            if (!cuda_ptr)
            {
                std::cerr << "Failed to get input binding for " << name << std::endl;
                throw std::runtime_error("Failed to get input binding");
            }

            size_t data_size = input_sizes_.at(name);
            cudaError_t err = cudaMemcpyAsync(cuda_ptr, cpu_ptr, data_size,
                                              cudaMemcpyHostToDevice, stream->get());
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }

            context_->setInputTensorAddress(name.c_str(), cuda_ptr);
        }
    }

    template <typename OutputType>
    void AsyncInfer<OutputType>::copy_input_to_device(const InputMatType &input_blob,
                                                      std::shared_ptr<StreamHandle> stream)
    {
        int stream_id = stream->id();

        for (const auto &[name, mat] : input_blob)
        {
            void *cuda_ptr = memory_pool_->get_input_binding(name, stream_id);
            if (!cuda_ptr)
            {
                std::cerr << "Failed to get input binding for " << name << std::endl;
                throw std::runtime_error("Failed to get input binding");
            }

            cv::Mat cpu_mat = mat;
            if (utility::typeCv2Rt(cpu_mat.type()) != engine_->getTensorDataType(name.c_str()))
            {
                cpu_mat.convertTo(cpu_mat,
                                  utility::typeRt2Cv(engine_->getTensorDataType(name.c_str())));
            }

            size_t data_size = input_sizes_.at(name);
            cudaError_t err = cudaMemcpyAsync(cuda_ptr, static_cast<void *>(cpu_mat.data),
                                              data_size, cudaMemcpyHostToDevice, stream->get());
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }

            context_->setInputTensorAddress(name.c_str(), cuda_ptr);
        }
    }

    template <typename OutputType>
    OutputType AsyncInfer<OutputType>::copy_output_from_device(std::shared_ptr<StreamHandle> stream)
    {
        OutputType output_blob;
        int stream_id = stream->id();

        for (const auto &name : output_names_)
        {
            void *cuda_ptr = memory_pool_->get_output_binding(name, stream_id);
            if (!cuda_ptr)
            {
                std::cerr << "Failed to get output binding for " << name << std::endl;
                throw std::runtime_error("Failed to get output binding");
            }

            size_t data_size = output_sizes_.at(name);
            std::shared_ptr<char[]> cpu_ptr(new char[data_size]);

            cudaError_t err = cudaMemcpyAsync(static_cast<void *>(cpu_ptr.get()), cuda_ptr,
                                              data_size, cudaMemcpyDeviceToHost, stream->get());
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }

            output_blob[name] = cpu_ptr;
        }

        cudaStreamSynchronize(stream->get());

        return output_blob;
    }

    template <>
    std::unordered_map<std::string, cv::Mat>
    AsyncInfer<std::unordered_map<std::string, cv::Mat>>::copy_output_from_device(
        std::shared_ptr<StreamHandle> stream)
    {
        std::unordered_map<std::string, cv::Mat> output_blob;
        int stream_id = stream->id();

        for (const auto &name : output_names_)
        {
            void *cuda_ptr = memory_pool_->get_output_binding(name, stream_id);
            if (!cuda_ptr)
            {
                std::cerr << "Failed to get output binding for " << name << std::endl;
                throw std::runtime_error("Failed to get output binding");
            }

            size_t data_size = output_sizes_.at(name);
            std::shared_ptr<char[]> cpu_ptr(new char[data_size]);

            cudaError_t err = cudaMemcpyAsync(static_cast<void *>(cpu_ptr.get()), cuda_ptr,
                                              data_size, cudaMemcpyDeviceToHost, stream->get());
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                throw std::runtime_error(cudaGetErrorString(err));
            }

            nvinfer1::DataType dtype = engine_->getTensorDataType(name.c_str());
            const auto &shape = output_shapes_.at(name);

            cv::Mat temp(shape.size(), shape.data(), utility::typeRt2Cv(dtype), cpu_ptr.get());
            output_blob[name] = temp.clone();
        }

        cudaStreamSynchronize(stream->get());

        return output_blob;
    }

    template class inference::AsyncInfer<std::unordered_map<std::string, std::shared_ptr<char[]>>>;
    template class inference::AsyncInfer<std::unordered_map<std::string, cv::Mat>>;

}
