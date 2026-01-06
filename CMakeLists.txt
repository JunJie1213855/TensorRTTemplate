cmake_minimum_required(VERSION 3.15)

project(trttemplate)

set(CMAKE_CXX_STANDARD 17)
set(LIB_TYPE SHARED)
set(CUDA_ROOT_DIR "/usr/local/cuda")
set(TensorRT_ROOT_DIR "/root/TensorRT")



# set lib type
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message("-- Window operating system")

    # for opencv cmake config
    set(OpenCV_ROOT_DIR "E:/lib/opencv/opencv-4.8.0/build/x64/vc16/lib")
    set(CMAKE_PREFIX_PATH "${OpenCV_ROOT_DIR};${CMAKE_PREFIX_PATH}")
    
    # set TensorRT and cuda library and include directory for windows system
    set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_ROOT_DIR}/lib/x64/*.lib")
    file(GLOB TensorRT_LIBS "${TensorRT_ROOT_DIR}/lib/*.lib")
    message("-- TensorRT libs : ${CUDA_LIBS}")
    message("-- Cuda libs : ${TensorRT_LIBS}")
else()
# set TensorRT and cuda library and include directory for linux system
    set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_ROOT_DIR}/lib64/*.so")
    file(GLOB TensorRT_LIBS "${TensorRT_ROOT_DIR}/lib/*.so")
    message("-- TensorRT libs : ${CUDA_LIBS}")
    message("-- Cuda libs : ${TensorRT_LIBS}")
endif()

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

# 为共享库添加BUILD_SHARED定义（用于Windows DLL导出）
if(${LIB_TYPE} STREQUAL "SHARED")
    message("-- BUILD SHARED LIBS")
    target_compile_definitions(trtemplate PRIVATE BUILD_SHARED)
endif()

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



add_executable(segformer Segformer.cc)
target_link_libraries(segformer
    trtemplate
)
