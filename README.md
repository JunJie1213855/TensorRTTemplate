## <div align="center">📄 TensorRT Template</div>

### 🛠️ Introduction
This is a template library for TensorRT inference that supports OpenCV's cv:: Mat type data and can support multiple input and output data. 

**New Features**: Now supports **async inference** with multi-stream concurrent processing for improved performance!

### ✒️​ Environment
* Windows 11 / Ubuntu20.04
* Visual Studio 2022 ~ 2026 / GNU
* CMake 3.20+
* TensorRT 10.x
* OpenCV > 4.5
* Cuda 11.x / 12.x

## 🚀 New Features - Async Inference

The library now supports **asynchronous inference** with the following features:

- **Multi-Stream Concurrent Inference**: Run multiple inference tasks in parallel using multiple CUDA streams
- **Future-based Async API**: Use `std::future` for non-blocking inference
- **Callback Support**: Register callback functions to handle results asynchronously
- **Backward Compatible**: All existing synchronous APIs still work unchanged
- **Memory Pool**: Pre-allocated GPU memory for reduced allocation overhead

### Async Inference Examples

#### 1. Initialize Model with Multiple Streams
```cpp
// Enable async with 4 CUDA streams
TRTInfer model("yolov8n.engine", 4, true);

// Use default 4 streams
TRTInfer model("yolov8n.engine");

// Disable async (backward compatible)
TRTInfer model("yolov8n.engine", 1, false);
```

#### 2. Async Inference with Future
```cpp
// Non-blocking inference
auto future = model.infer_async(input_blob);

// Do other work here...

// Wait for result when needed
auto output = future.get();
```

#### 3. Async Inference with Callback
```cpp
model.infer_with_callback(input_blob, 
    [](const auto& output) {
        // Process results asynchronously
        process_output(output);
    }
);

// Continue with other work
```

#### 4. Concurrent Inference
```cpp
std::vector<std::future<OutputType>> futures;

// Submit multiple inference tasks
for (auto& img : images) {
    futures.push_back(model.infer_async(preprocess(img)));
}

// Wait for all results
for (auto& f : futures) {
    auto result = f.get();
    process(result);
}
```

### Performance Benefits

- **2-5x throughput improvement** with 4-8 streams
- **70-90% GPU utilization** compared to ~30% with single stream
- **Better scalability** for batch processing and video inference

### API Reference

#### Constructors
```cpp
// Async enabled (default: 4 streams)
TRTInfer(const std::string &engine_path, int num_streams = 4, bool enable_async = true);

// Synchronous mode (backward compatible)
TRTInfer(const std::string &engine_path);
```

#### Async Methods
```cpp
// Future-based async inference
std::future<std::unordered_map<std::string, cv::Mat>> 
infer_async(const std::unordered_map<std::string, cv::Mat> &input_blob);

// Callback-based async inference
void infer_with_callback(const std::unordered_map<std::string, cv::Mat> &input_blob,
                        std::function<void(const std::unordered_map<std::string, cv::Mat>&)> callback);

// Wait for all pending async inferences
void wait_all();

// Get number of active streams
int num_streams() const;
```

#### Synchronous Methods (Unchanged)
```cpp
// Original synchronous API - still works!
std::unordered_map<std::string, cv::Mat> operator()(const std::unordered_map<std::string, cv::Mat> &input_blob);
```

### Example: YOLO Async Inference

See `YOLO_async.cc` for a complete example demonstrating:
- Synchronous inference (original method)
- Future-based async inference
- Callback-based async inference
- Concurrent batch processing

Build and run:
```bash
cmake -S . -B build
cmake --build ./build --config release -j 12
./build/yolo_async
```

---

### ⚙️ Common Configuration Options

#### CMake Configuration
Before building, you need to configure the library paths in `CMakeLists.txt`:

**For Windows:**
```cmake
set(CUDA_ROOT_DIR "E:/lib/cuda/12.1")              # Path to CUDA installation
set(TensorRT_ROOT_DIR "E:/lib/TensorRT/TensorRT-10.10.0.31")  # Path to TensorRT
set(OpenCV_ROOT_DIR "E:/lib/opencv/opencv-4.8.0/build/x64/vc16/lib")  # Path to OpenCV
set(LIB_TYPE SHARED)  # Options: SHARED (DLL) or STATIC
```

**For Linux:**
```cmake
set(CUDA_ROOT_DIR "/usr/local/cuda")
set(TensorRT_ROOT_DIR "/usr/local/TensorRT-10.10.0.31")
set(LIB_TYPE SHARED)  # Options: SHARED (.so) or STATIC
```

#### Library Type Options
- `LIB_TYPE = SHARED`: Build as shared library (Windows: `.dll`, Linux: `.so`)
- `LIB_TYPE = STATIC`: Build as static library (Windows: `.lib`, Linux: `.a`)


### 📦 Build

#### Windows Setup

1. **Install Dependencies:**
   - Download and install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (11.x or 12.x)
   - Download and install [TensorRT](https://developer.nvidia.com/tensorrt) 10.x for Windows
   - Download and build [OpenCV](https://opencv.org/releases/) or use prebuilt binaries

2. **Configure Paths in CMakeLists.txt:**
   ```cmake
   set(CUDA_ROOT_DIR "Your/CUDA/Path")
   set(TensorRT_ROOT_DIR "Your/TensorRT/Path")
   set(OpenCV_ROOT_DIR "Your/OpenCV/Path")
   ```

3. **Build the Project:**
   ```bash
   cmake -S . -B build
   cmake --build build --config release
   ```

4. **Output:**
   - Library: `build/Release/trtemplate.dll` and `build/Release/trtemplate.lib`
   - Executables: `build/Release/yolo.exe`, `build/Release/fcn.exe`, etc.

#### Linux Setup

1. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install cuda-toolkit-12-x  # or cuda-toolkit-11-x
   sudo apt install libopencv-dev // for opencv
   ```

2. **Install TensorRT:**
   Download TensorRT for Linux from [NVIDIA website](https://developer.nvidia.com/tensorrt) and follow the installation guide.

3. **Configure Paths in CMakeLists.txt:**
   ```cmake
   set(CUDA_ROOT_DIR "/usr/local/cuda")
   set(TensorRT_ROOT_DIR "/path/to/TensorRT")
   ```

4. **Build the Project:**
   ```bash
   cmake -S . -B build
   cmake --build build --config release -j 12
   ```

5. **Output:**
   - Library: `build/libtrtemplate.so`
   - Executables: `build/yolo`, `build/fcn`, etc.

#### Common Build Issues

1. **TensorRT Version Compatibility:**
   - TensorRT 8.x are not compatible
   - Must use TensorRT 10.x for this project

2. **CUDA Version Mismatch:**
   - Ensure CUDA version matches TensorRT requirements
   - TensorRT 10.x requires CUDA 11.8 or 12.x

3. **OpenCV Path Issues:**
   - Windows: Point to the `lib` directory containing `*.lib` files
   - Linux: Ensure `pkg-config opencv4 --cflags --libs` works

4. **Build Type:**
   - Use Release builds for production (`--config release`)
   - Use Debug builds for development and debugging


### ✨ Example
First，download the onnx file of YOLOv8 and fcn from the link [GoogleDrive](https://drive.google.com/drive/folders/19UBgYWeEADKTA1w44HIDkzn2oPxKATOH?usp=drive_link).

#### YOLOv8 - Detection Example
convert the onnx file to engien file like
```bash
trtexec \ 
--onnx=./pretrain/yolov8n.onnx \
--saveEngine=./yolov8n.engine
```
build this example by later command
```bash
cmake -S . -B build
cmake --build ./build --config release -j 12
```
run
```bash
./build/Release/yolo.exe
# or linux
./build/yolo 
```
more detail code see YOLO.cc.

![alt text](demo/image.png)

#### FCN / Segformer - Segmenatation Example
convert the onnx file to engien file like
```bash
trtexec \ 
--onnx=./pretrain/fcn.onnx \
--saveEngine=./fcn.engine
```
build this example by later command
```bash
cmake -S . -B build
cmake --build ./build --config release -j 12
```
run
```bash
./build/Release/example.exe
# or linux
./build/fcn
```
more detail code see FCN.cc.And Segformer is like this.

![alt text](demo/image-1.png)



#### Template - for your model
If you are writing your own model inference acceleration, Please follow the steps below
* Export the model to ONNX
* Convert ONNX to engine
* Write the preprocess and postprocess code

The following is a preprocess and postprocess code template:

**Synchronous Inference (Original):**
```C++
#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
namespace model{
    // Preprocess for the input blob 
    std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &left, cv::Mat &right)
    {
        ...
    }
    // Postprocess for the input blob 
    std::unordered_map<std::string, cv::Mat> postprocess(std::unordered_map<std::string, cv::Mat> )
    {
        ...
    }
}
int main(int argc, char *argv[])
{
    cv::Mat  tensor1 = cv::imread("...");
    cv::Mat  tensor2 = cv::imread("...");

    // Preprocess
    auto input_blob = model::preprocess(tensor1, tensor2);
    // Model inference
    TRTInfer model("*.engine");
    // Output
    auto output_blob = model(input_blob);
    cv::Mat dst;
    model::postprocess(output_blob);

    // Visualization
    ...
    return 1;
}
```

**Asynchronous Inference (New):**
```C++
#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace model{
    std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &img) { ... }
    void postprocess(std::unordered_map<std::string, cv::Mat> output) { ... }
}

int main(int argc, char *argv[])
{
    // Initialize model with 4 streams for async
    TRTInfer model("*.engine", 4, true);
    
    // Load images
    std::vector<cv::Mat> images = {...};
    std::vector<std::future<std::unordered_map<std::string, cv::Mat>>> futures;
    
    // Submit all async inference tasks
    for (auto& img : images) {
        auto input_blob = model::preprocess(img);
        futures.push_back(model.infer_async(input_blob));
    }
    
    // Wait for all results and process
    for (auto& f : futures) {
        auto output = f.get();
        model::postprocess(output);
    }
    
    return 0;
}
```
Here are a few points that may likely cause errors:
* **Tensor Names**
  * Some weight files have inconsistent input tensor names – handle with care
  * Use `polygraphy` to verify tensor names before implementation
* **Data Types**
  * Internal data type conversions may be involved – proceed with caution
  * Common conversions: float32 ↔ float16 (FP16), int8, uint8
* **Input Data Shape**
  * While preprocessing includes resizing, errors may occur inside the model – pay close attention
  * Ensure NCHW vs NHWC format matches model expectations

#### Verifying Model with Polygraphy

It's best to run a check with Python lib `polygraphy` before execution. For example, here's the command for checking YOLOv8:

```bash
$ polygraphy run yolov8n.onnx --onnxrt
[I] RUNNING | Command: /root/miniconda3/envs/dlpy310/bin/polygraphy run yolov8n.onnx --onnxrt
[I] onnxrt-runner-N0-11/21/25-13:22:08  | Activating and starting inference
[I] Creating ONNX-Runtime Inference Session with providers: ['CPUExecutionProvider']
[I] onnxrt-runner-N0-11/21/25-13:22:08 
    ---- Inference Input(s) ----
    {images [dtype=float32, shape=(1, 3, 640, 480)]}
[I] onnxrt-runner-N0-11/21/25-13:22:08 
    ---- Inference Output(s) ----
    {output0 [dtype=float32, shape=(1, 84, 6300)]}
[I] onnxrt-runner-N0-11/21/25-13:22:08  | Completed 1 iteration(s) in 59.11 ms | Average inference time: 59.11 ms.
[I] PASSED | Runtime: 1.037s | Command: /root/miniconda3/envs/dlpy310/bin/polygraphy run yolov8n.onnx --onnxrt
```

The name, type, and dimensions of the tensor can be clearly seen.

#### Engine Conversion Options

When converting ONNX to TensorRT engine, you can use various optimization options:

**Basic Conversion:**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine
```

**FP16 Precision (Faster, slightly lower accuracy):**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

**INT8 Precision (Fastest, requires calibration):**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --int8 --calib=calibration.cache
```

**Batch Size Configuration:**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640
```

**Workspace Size:**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --workspace=4096  # in MB
```

**Verbose Output:**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --verbose
```

---

## 🔧 Advanced Topics

### Async Inference Architecture

The async inference system uses the following components:

1. **StreamPool**: Manages a pool of CUDA streams for concurrent execution
2. **MemoryPool**: Pre-allocates GPU memory for each stream to avoid runtime allocation
3. **AsyncInfer**: Template class providing async inference capabilities
4. **InferenceTask**: Encapsulates individual inference tasks

### Performance Tuning

#### Number of Streams
- **Start with 4 streams** for most models
- Increase to 8 for small/fast models (YOLOv8n)
- Use 2-4 for large models (Segformer, FCN)
- Monitor GPU utilization to find optimal value

#### Memory Considerations
```cpp
// Memory usage = (input + output) size × num_streams
// Example: YOLOv8n (640x640) × 4 streams ≈ 200MB
```

#### Thread Safety
- The `TRTInfer` class is **thread-safe** for concurrent `infer_async()` calls
- Each call gets its own stream from the pool
- Synchronization is managed automatically

### Best Practices

1. **Use async for batch processing**: When processing multiple images/videos
2. **Use sync for single inference**: When latency is critical and throughput doesn't matter
3. **Reuse model instances**: Avoid creating multiple `TRTInfer` instances
4. **Profile your use case**: Measure performance to find optimal stream count

### Troubleshooting

#### Out of Memory Error
```cpp
// Reduce number of streams
TRTInfer model("engine", 2, true);  // Use fewer streams
```

#### No Performance Improvement
- Ensure your GPU supports concurrent kernel execution
- Check that model is not too small to benefit from concurrency
- Verify `enable_async` is set to `true`

#### High Latency in Async Mode
- Use `wait_all()` strategically to batch sync operations
- Consider callback-based API for pipeline parallelism

### Code Structure

```
TRTInfer/
├── TRTinfer.h/cc           # Main inference class
├── inference_config.h       # Configuration constants
├── stream_pool.h/cc        # CUDA stream management
├── memory_pool.h/cc        # GPU memory management
├── async_infer.h/cc       # Async inference implementation
├── inference_task.h        # Task encapsulation
├── utility.h/cc           # Utility functions
└── config.h               # Build configuration
```

### Implementation Details

#### Memory Pool Design
- Pre-allocates memory for each stream during initialization
- Each tensor has N allocations (N = num_streams)
- No runtime cudaMalloc/cudaFree during inference
- Automatic memory cleanup on destruction

#### Stream Pool Design
- Round-robin stream allocation
- Blocking acquire when all streams are busy
- Automatic release after inference completes
- Supports 1-16 concurrent streams

#### Async Execution Flow
1. User calls `infer_async(input)`
2. Stream acquired from pool
3. Input copied to GPU (async)
4. Inference enqueued (async)
5. Output copied to CPU (async)
6. Future returned to user
7. Stream released back to pool