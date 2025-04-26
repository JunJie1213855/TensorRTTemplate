cmake_minimum_required(VERSION 3.15)

project(trttemplate)


# set
set(CMAKE_CXX_STANDARD 20)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(LIB_TYPE SHARED)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_TYPE STATIC)
    message("Window operating system build static library")
endif()
set(CUDA_HOME "/usr/local/cuda")
set(TensorRT_HOME "/root/TensorRT")


# set TensorRT and cuda library and include directory
set(CUDA_INCLUDE_DIR ${CUDA_HOME}/include)
set(TensorRT_INCLUDE_DIR ${TensorRT_HOME}/include)
file(GLOB CUDA_LIBS ${CUDA_HOME}/lib64/*.so)
file(GLOB TensorRT_LIBS ${TensorRT_HOME}/lib/*.so)
message("TensorRT libs : ${TensorRT_LIBS} ")
message("CUDA libs : ${CUDA_LIBS} ")

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


add_executable(main main.cc)
target_link_libraries(main
    trtemplate
)

add_executable(example example.cc)
target_link_libraries(example
    trtemplate
)


add_executable(fastacv fastacv.cc)
target_link_libraries(fastacv
    trtemplate
)