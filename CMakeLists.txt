cmake_minimum_required(VERSION 3.20)

project(trttemplate)

# set lib type
set(LIB_TYPE SHARED)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_TYPE STATIC)
    message("Window operating system build static library")
endif()

# set TensorRT and cuda library and include directory
set(TensorRT_INCLUDE_DIR "E:/lib/Tensorrt/TensorRT-10.0.1.6/include")
set(CUDA_INCLUDE_DIR "E:/lib/cudalib-11.8/development/include")
file(GLOB CUDA_LIBS "E:/lib/cudalib-11.8/development/lib/x64/*.lib")
file(GLOB TensorRT_LIBS "E:/lib/Tensorrt/TensorRT-10.0.1.6/lib/*.lib")


set(TensorRT_INCLUDE_DIR "/root/TensorRT/include")
set(CUDA_INCLUDE_DIR "/usr/local/cuda/include")
file(GLOB CUDA_LIBS "/usr/local/cuda/lib64/*.so")
file(GLOB TensorRT_LIBS "/root/TensorRT/lib/*.so")


# find dependencies library
find_package(OpenCV REQUIRED)

# include
include_directories(
    ${PROJECT_SOURCE_DIR}/TRTInfer
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
# generate library
add_library(trtemplate ${LIB_TYPE}
    TRTInfer/TRTinfer.cc
    TRTInfer/utility.cc
)

target_link_libraries(trtemplate 
PRIVATE
    ${CUDA_LIBS}
    ${TensorRT_LIBS}
PUBLIC
    ${OpenCV_LIBS}
)


add_executable(igev IGEV.cc)
target_link_libraries(igev
    trtemplate
)

add_executable(yolo YOLO.cc)
target_link_libraries(yolo
    trtemplate
)


add_executable(fcn FCN.cc)

target_link_libraries(fcn
    trtemplate
)
