
cmake_minimum_required(VERSION 3.12)
project(trt_opencv_demo)

# 1. 配置CUDA
find_package(CUDA REQUIRED)  # 自动搜索系统CUDA路径‌:ml-citation{ref="2" data="citationList"}
include_directories(${CUDA_INCLUDE_DIRS})  # 包含CUDA头文件‌:ml-citation{ref="2,4" data="citationList"}
link_directories(${CUDA_LIBRARY_DIRS})     # 链接CUDA库路径‌:ml-citation{ref="2,6" data="citationList"}

# 2. 配置TensorRT
set(TENSORRT_ROOT "/home/leonpu/Downloads/TensorRT-10.6.0.26")  # 根据实际路径修改‌:ml-citation{ref="2,4" data="citationList"}
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer HINTS ${TENSORRT_ROOT} PATH_SUFFIXES lib lib64)
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_ROOT}/lib)

# 3. 配置OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 4. 添加可执行文件
add_executable(trt_opencv_demo src/yolov5.cpp)  # 支持CUDA源码编译‌:ml-citation{ref="1,2" data="citationList"}
target_link_libraries(trt_opencv_demo
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}        # 优先链接CUDA库（避免TensorRT版本冲突）‌:ml-citation{ref="6" data="citationList"}
    ${TENSORRT_LIBRARY_INFER}
    cudart cublas            # CUDA运行时库‌:ml-citation{ref="2,4" data="citationList"}
)
