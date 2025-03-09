// #include <stdio.h>
// #include <vector>




// // 反序列化模型 plan(engine) disk--> memory
// std::vector<char> engineData(fsize);
// engineFile.read(engineData.data(), fsize);

// std::unique_ptr<nvinfer1::IRuntime> mRuntime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};

// std::unique_ptr<nvinfer1::ICudaEngine> mEngine(runtime->deserializeCudaEngine(engineData.data(), fsize));

// // TensorRT执行上下文封装了执行状态，例如用于在推理期间保存中间激活张量的持久设备内存。
// // 由于分割模型是在启用动态形状的情况下构建的，因此必须为推理执行指定输入的形状。可以查询网络输出形状以确定输出缓冲器的相应尺寸。

// char const* input_name = "input";
// assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);   
// auto input_dims = nvinfer1::Dims4{1, /* channels */ 3, height, width};  
// context->setInputShape(input_name, input_dims);   //动态显式指定
// auto input_size = util::getMemorySize(input_dims, sizeof(float));

// char const* output_name = "output";
// assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kINT64);
// auto output_dims = context->getTensorShape(output_name);  // 根据输入变化     
// auto output_size = util::getMemorySize(output_dims, sizeof(int64_t));


// // 在准备推理时，为所有输入和输出分配CUDA设备内存，处理图像数据并将其复制到输入内存中，并生成引擎绑定列表。
// void* input_mem{nullptr};
// cudaMalloc(&input_mem, input_size);
// void* output_mem{nullptr};
// cudaMalloc(&output_mem, output_size);


// const std::vector<float> mean{0.485f, 0.456f, 0.406f};
// const std::vector<float> stddev{0.229f, 0.224f, 0.225f};
// auto input_image{util::RGBImageReader(input_filename, input_dims, mean, stddev)};
// input_image.read();

// cudaStream_t stream;
// auto input_buffer = input_image.process();
// cudaMemcpyAsync(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream);

// // 使用上下文的executeV2或enqueueV3方法启动推理执行。执行后，我们将结果复制到主机缓冲区并释放所有设备内存分配。

// context->setTensorAddress(input_name, input_mem);    // 绑定
// context->setTensorAddress(output_name, output_mem);  

// bool status = context->enqueueV3(stream);    // 推理
// auto output_buffer = std::unique_ptr<int64_t>{new int64_t[output_size]};
// cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream);
// cudaStreamSynchronize(stream);

// cudaFree(input_mem);
// cudaFree(output_mem);


// // py脚本
// /*
// def infer(engine, input_file, output_file):
//     print("Reading input image from file {}".format(input_file))
//     with Image.open(input_file) as img:
//         input_image = preprocess(img)
//         image_width = img.width
//         image_height = img.height

//     with engine.create_execution_context() as context:
//         # Set input shape based on image dimensions for inference
//         context.set_binding_shape(engine.get_binding_index("input"), (1, 3, image_height, image_width))
//         # Allocate host and device buffers
//         bindings = []
//         for binding in engine:
//             binding_idx = engine.get_binding_index(binding)
//             size = trt.volume(context.get_binding_shape(binding_idx))
//             dtype = trt.nptype(engine.get_binding_dtype(binding))
//             if engine.binding_is_input(binding):
//                 input_buffer = np.ascontiguousarray(input_image)
//                 input_memory = cuda.mem_alloc(input_image.nbytes)
//                 bindings.append(int(input_memory))
//             else:
//                 output_buffer = cuda.pagelocked_empty(size, dtype)
//                 output_memory = cuda.mem_alloc(output_buffer.nbytes)
//                 bindings.append(int(output_memory))

//         stream = cuda.Stream()
//         # Transfer input data to the GPU.
//         cuda.memcpy_htod_async(input_memory, input_buffer, stream)
//         # Run inference
//         context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
//         # Transfer prediction output from the GPU.
//         cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
//         # Synchronize the stream
//         stream.synchronize()

//     with postprocess(np.reshape(output_buffer, (image_height, image_width))) as img:
//         print("Writing output image to file {}".format(output_file))
//         img.convert('RGB').save(output_file, "PPM")


// */



#include <iostream>
#include <fstream>
#include <stdio.h>
#include "cuda_runtime.h"
#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <vector>
using namespace nvinfer1;
using namespace std;

class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;


std::vector<char> readModelFromFile(const std::string& filename) {

    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};  // 错误处理
    std::streamsize size = file.tellg(); // 直接获取文件大小
    file.seekg(0, std::ios::beg); // 重置指针至起始位置
    std::vector<char> buffer(size); 
    if (file.read(buffer.data(), size)) return buffer; // 一次性读取全部内容
    return {};  // 读取失败
}



int main(){

    // IBuilder* builder = createInferBuilder(logger);  // 创建builder 
IRuntime* runtime = createInferRuntime(logger);
std::vector<char> modelData = readModelFromFile("/home/leonpu/Downloads/inference.engine");


ICudaEngine* engine = runtime->deserializeCudaEngine(modelData.data(), modelData.size());
IExecutionContext *context = engine->createExecutionContext();


// 申请内存
size_t output_size = 1 * 25200 * 85;
float *host_output;
float *device_output;
cudaMalloc(&host_output, output_size * sizeof(float));
cudaMalloc(&device_output, output_size * sizeof(float));

float *device_input;
size_t input_size = 1 * 3 * 640 * 640;
cudaMalloc(&device_input, input_size * sizeof(float));








// 1. 读取图像（BGR+HWC格式）
cv::Mat bgr_image = cv::imread("/home/leonpu/Desktop/2D_Detection/yolov5/yolov5deploy/zidane.jpg", cv::IMREAD_COLOR);  // ‌:ml-citation{ref="1,2" data="citationList"}
if (bgr_image.empty()) {
    std::cerr << "Failed to load image." << std::endl;
    return -1;
}


// 2. BGR转RGB
cv::Mat rgb_image;
cv::cvtColor(bgr_image, rgb_image, cv::COLOR_BGR2RGB);  // ‌:ml-citation{ref="1,4" data="citationList"}

// 3. 调整尺寸至640x640
cv::Mat resized_image;
cv::resize(rgb_image, resized_image, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);  // ‌:ml-citation{ref="4,5" data="citationList"}


// / 255
cv::Mat normalized_image;
resized_image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);
// type = (数据类型深度) + (通道数 - 1) * 8
// 验证输出形状（CHW: 3×640×640）
std::cout << "CHW shape: (" << normalized_image.channels() << ", " 
            << normalized_image.rows << ", " 
            << normalized_image.cols << ")" << *(normalized_image.data)<<std::endl;

double minVal, maxVal;
cv::minMaxLoc(normalized_image, &minVal, &maxVal);
std::cout << "像素范围: [" << minVal << ", " << maxVal << "] "  << normalized_image.type()<< std::endl;  // 若maxVal<=1，则可能已除以255‌:ml-citation{ref="2,3" data="citationList"}


// 4. HWC转CHW（OpenCV默认HWC，需手动转换）
std::vector<cv::Mat> channels;
cv::split(normalized_image, channels);  // 拆分通道（HWC → 3个独立的H×W矩阵）‌:ml-citation{ref="5,7" data="citationList"}
std::cout << channels.size() << " channels " << std::endl;
// 合并为CHW格式（连续内存）













// context->setTensorAddress(INPUT_NAME, inputBuffer);
// context->setTensorAddress(OUTPUT_NAME, outputBuffer);
// // 如果引擎是用动态形状构建的，你还必须指定输入形状：
// context->setInputShape(INPUT_NAME, inputDims);

// context->enqueueV3(stream);



    return 0;
}




