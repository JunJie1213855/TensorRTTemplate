## <div align="center">📄 TensorRT Template</div>

### 🛠️ Introduction
This is a template library for TensorRT inference that supports OpenCV's cv:: Mat type data and can support multiple input and output data.

### ✒️​ Environment
* Windows 11 / Ubuntu20.04
* msvc / gnu
* TensorRT 10.x
* OpenCV > 4.5
* Cuda 11.x



### 📦 Build

Configure CUDA and TensorRT according to online guides. Please note that the APIs for TensorRT 8.x and TensorRT 10.x are not fully compatible; therefore, you must use the TensorRT 10.x version specifically for this project. Also, install OpenCV by the command
```bash
sudo apt install libopencv-dev
```
The, compiling with cmake
```bash
cmake -S . -B build
cmake --build build --config release
```


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
* export the model to onnx
* convert onnx to engine
* write the preprocess and postprocess code

the following of preprocess and postprocess code template to write the code
```C++
#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
namespace model{
    // preprocess for the input blob 
    std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &left, cv::Mat &right)
    {
        ...
    }
    // postprocess for the input blob 
    std::unordered_map<std::string, cv::Mat> postprocess(std::unordered_map<std::string, cv::Mat> )
    {
        ...
    }
}
int main(int argc, char *argv[])
{
    cv::Mat  tensor1 = cv::imread("...");
    cv::Mat  tensor2 = cv::imread("...");

    // 预处理
    auto input_blob = model::preprocess(tensor1, tensor2);
    //  模型
    TRTInfer model("*.engine");
    //  输出
    auto output_blob = model(input_blob);
    cv::Mat dst;
    model::postprocess(output_blob);

    // visualization
    ...
    return 1;
}
```
Here are a few points that may likely cause errors:
* Tensor Names
  * Some weight files have inconsistent input tensor names – handle with care
* Data Types
  * Internal data type conversions may be involved – proceed with caution
* Input Data Shape
  * While preprocessing includes resizing, errors may occur inside the model – pay close attention

It's best to run a check with Polygraphy before execution. For example, here's the command for checking YOLOv8:
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
