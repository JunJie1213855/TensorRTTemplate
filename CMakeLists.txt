cmake_minimum_required(VERSION 3.15)

project(trttemplate 
    VERSION 2.0.0
    DESCRIPTION "High-performance TensorRT inference library with async support"
)

set(CMAKE_CXX_STANDARD 17)

# ============================================================================
# Options
# ============================================================================
set(LIB_TYPE SHARED)
option(ENABLE_ASYNC "Enable async inference support" ON)
option(BUILD_EXAMPLES "Build example executables" ON)
option(BUILD_TESTS "Build test suite" OFF)
option(ENABLE_WARNINGS "Enable compiler warnings" ON)
option(ENABLE_SANITIZE "Enable address sanitizer" OFF)

# ============================================================================
# Platform-specific Settings
# ============================================================================
if(WIN32)
    message("-- Window operating system")
    set(OpenCV_ROOT_DIR "E:/lib/opencv/opencv-4.8.0/build/x64/vc16/lib")
    set(CMAKE_PREFIX_PATH "${OpenCV_ROOT_DIR};${CMAKE_PREFIX_PATH}")
    set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_ROOT_DIR}/lib/x64/*.lib")
    file(GLOB TensorRT_LIBS "${TensorRT_ROOT_DIR}/lib/*.lib")
    message("-- TensorRT libs : ${CUDA_LIBS}")
    message("-- Cuda libs : ${TensorRT_LIBS}")
else()
    message("-- Linux/Unix operating system")
    set(CUDA_ROOT_DIR "/usr/local/cuda")
    set(TensorRT_ROOT_DIR "/root/TensorRT")
    file(GLOB CUDA_LIBS "${CUDA_ROOT_DIR}/lib64/*.so")
    file(GLOB TensorRT_LIBS "${TensorRT_ROOT_DIR}/lib/*.so")
    message("-- TensorRT libs : ${CUDA_LIBS}")
    message("-- Cuda libs : ${TensorRT_LIBS}")
endif()

# Set include dirs
set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")

# ============================================================================
# Compiler Settings
# ============================================================================
if(ENABLE_WARNINGS)
    if(MSVC)
        add_compile_options(/W4)
    else()
        add_compile_options(-Wall -Wextra -Wpedantic)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            add_compile_options(-Wno-unused-parameter)
        endif()
    endif()
endif()

if(ENABLE_SANITIZE AND NOT CMAKE_BUILD_TYPE MATCHES "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
endif()

# ============================================================================
# Find Dependencies
# ============================================================================
find_package(OpenCV REQUIRED)

# ============================================================================
# Include Directories
# ============================================================================
include_directories(
    ${PROJECT_SOURCE_DIR}/TRTInfer
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# ============================================================================
# Generate Library
# ============================================================================
add_library(trtemplate ${LIB_TYPE}
    TRTInfer/TRTinfer.cc
    TRTInfer/utility.cc
    TRTInfer/stream_pool.cc
    TRTInfer/memory_pool.cc
    TRTInfer/async_infer.cc
)

# Library version
set_target_properties(trtemplate PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Library defines
if(${LIB_TYPE} STREQUAL "SHARED")
    message("-- BUILD SHARED LIBS")
    target_compile_definitions(trtemplate PRIVATE BUILD_SHARED)
endif()

if(ENABLE_ASYNC)
    target_compile_definitions(trtemplate PRIVATE ENABLE_ASYNC_INFERENCE)
    message("-- Async inference enabled")
endif()

# Library dependencies
target_link_libraries(trtemplate 
    PRIVATE
        ${CUDA_LIBS}
        ${TensorRT_LIBS}
    PUBLIC
        ${OpenCV_LIBS}
)

# ============================================================================
# Examples
# ============================================================================
if(BUILD_EXAMPLES)
    message("-- Building examples")
    
    add_executable(igev IGEV.cc)
    target_link_libraries(igev trtemplate)
    
    add_executable(yolo YOLO.cc)
    target_link_libraries(yolo trtemplate)
    
    add_executable(yolo_async YOLO_async.cc)
    target_link_libraries(yolo_async trtemplate)
    
    add_executable(fcn FCN.cc)
    target_link_libraries(fcn trtemplate)
    
    add_executable(segformer Segformer.cc)
    target_link_libraries(segformer trtemplate)
    
    add_executable(s2m2 s2m2.cc)
    target_link_libraries(s2m2 trtemplate)
    
    add_executable(s2m2_async s2m2_async.cc)
    target_link_libraries(s2m2_async trtemplate)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "===========================================")
message(STATUS "  ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "===========================================")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "Library type:       ${LIB_TYPE}")
message(STATUS "C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "Async support:      ${ENABLE_ASYNC}")
message(STATUS "Build examples:    ${BUILD_EXAMPLES}")
message(STATUS "Build tests:       ${BUILD_TESTS}")
message(STATUS "Enable warnings:   ${ENABLE_WARNINGS}")
message(STATUS "Enable sanitize:   ${ENABLE_SANITIZE}")
message(STATUS "CUDA root:         ${CUDA_ROOT_DIR}")
message(STATUS "TensorRT root:     ${TensorRT_ROOT_DIR}")
message(STATUS "OpenCV version:    ${OpenCV_VERSION}")
message(STATUS "===========================================")
message(STATUS "")
