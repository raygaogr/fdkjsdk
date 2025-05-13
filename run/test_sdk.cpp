#include "flabsdk.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <onnxruntime_cxx_api.h>
#include <limits.h>
#include <codecvt>
#include <Windows.h>
#include <time.h>

//#include <cuda_runtime.h>
//#include <cudnn.h>
typedef int (*cudaRuntimeGetVersion_t)(int*);
typedef unsigned int (*cudnnGetVersion_t)();

void drawResult(const cv::Mat& image, const flabsdk::flabio::DetInferRes& infer_res) {
	for (const auto& box : (infer_res.bboxes_vec)[0]) {
		cv::rectangle(image, cv::Point(box.x - int(box.width / 2), box.y - int(box.height / 2)), 
			                 cv::Point(box.x + int(box.width / 2), box.y + int(box.height / 2)), cv::Scalar(0, 255, 0), 2);
		std::string output_text = box.uid + " " + std::to_string(box.score);
		cv::putText(image, output_text, cv::Point(box.x - int(box.width / 2), box.y - int(box.height / 2)), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
	}
	cv::imwrite("res.jpg", image);
}

int calculate_product(const std::vector<std::int64_t>& v) {
	int total = 1;
	for (auto& i : v) total *= i;
	return total;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
	Ort::MemoryInfo mem_info =
		Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
	return tensor;
}

int test_cuda() {
	//int cudaVersion;
	//cudaError_t err = cudaRuntimeGetVersion(&cudaVersion);
	//if (err == cudaSuccess && cudaVersion > 0) {
	//	std::cout << "CUDA is installed. Version: " << (cudaVersion / 1000) << "." << (cudaVersion % 100) << std::endl;
	//}
	//else {
	//	std::cout << "CUDA is not installed or cannot be accessed." << std::endl;
	//}
	HMODULE handle = LoadLibrary(TEXT("cudart64_12.dll"));
	if (handle == NULL) {
		std::cerr << "Cannot open cudart library." << std::endl;
		return 1;
	}
	cudaRuntimeGetVersion_t cudaRuntimeGetVersion = (cudaRuntimeGetVersion_t)GetProcAddress(handle, "cudaRuntimeGetVersion");
	if (cudaRuntimeGetVersion == NULL) {
		std::cerr << "Cannot load function 'cudaRuntimeGetVersion'." << std::endl;
		FreeLibrary(handle);
		return 1;
	}
	int version;
	int result = cudaRuntimeGetVersion(&version);
	if (result == 0) {
		std::cout << "CUDA Runtime Version: " << (version / 1000) << "." <<((version % 1000) / 10) << std::endl;
	}
	else {
		std::cout << "Failed to get CUDA runtime version." << std::endl;
	}

	FreeLibrary(handle);
	return 0;
}

int test_cudnn() {
	// 尝试打开cuDNN共享库
	HMODULE handle = LoadLibrary(TEXT("cudnn64_9.dll")); // 确保这里是你的cuDNN DLL的实际名称
	if (handle == NULL) {
		std::cerr << "Cannot open cudnn library. Error: " << GetLastError() << std::endl;
		return 1;
	}

	// 加载cudnnGetVersion函数
	cudnnGetVersion_t cudnnGetVersion = (cudnnGetVersion_t)GetProcAddress(handle, "cudnnGetVersion");
	if (cudnnGetVersion == NULL) {
		std::cerr << "Cannot load function 'cudnnGetVersion'. Error: " << GetLastError() << std::endl;
		FreeLibrary(handle);
		return 1;
	}

	// 使用加载的函数获取cuDNN版本号
	unsigned int version = cudnnGetVersion();
	std::cout << "cuDNN Version: " << (version / 10000) << "." << ((version % 10000) / 100) << std::endl;

	// 关闭库句柄
	FreeLibrary(handle);

	return 0;
}


int main(int argc, char* argv[]) {
	//test_cuda();
	//test_cudnn();
	flabsdk::flabio::PlatformInfo platform_info;
	auto a = flabsdk::GetPlatformInfo(&platform_info);
	std::cout << platform_info.platform << ": " << platform_info.version << ", "  << std::endl;

	flabsdk::InferEngine* engine = nullptr;
	std::cout << "start create engine" << std::endl;

	int d = INT_MAX;
	int e = INT_MIN;

	auto status = flabsdk::CreateInferEngine("0002", &engine);
	if (status != flabsdk::Status::kSuccess) {
		std::cerr << "Failed to create inference engine: " << static_cast<int>(status) << std::endl;
		return -1;
	}
	std::cout << "sucessfully create engine" << std::endl;


	//getchar();
	//status = engine->InitLog("log.txt");
	//if (status != flabsdk::Status::kSuccess) {
	//	std::cerr << "Failed to initialize log: " << static_cast<int>(status) << std::endl;
	//	delete engine;
	//	return -1;
	//}
	//std::cout << "sucessfully init log" << std::endl;

	std::string str = "config/config.json";
	status = engine->LoadResources(str);
	if (status != flabsdk::Status::kSuccess) {
		std::cerr << "Failed to load resources: " << static_cast<int>(status) << std::endl;
		engine->ClearResources();
		delete engine;
		return -1;
	}

	flabsdk::flabio::DetInferCfg infer_cfg;
	flabsdk::flabio::DetInferRes infer_res;
	cv::Mat input_mat = cv::imread("data/image/test_glassbracket.jpg");
	auto input_h = input_mat.rows;
	auto input_w = input_mat.cols;

	flabsdk::flabio::ROI roi;
	roi.x = int(input_w / 2);
	roi.y = int(input_h / 2);
	roi.width = input_w;
	roi.height = input_h;

	infer_cfg.infer_rois.push_back(roi);

	time_t start_time = time(NULL);
	int infer_times = 10;
	for (int i = 0; i < infer_times; i++) {
		status = engine->InferSync(input_mat, &infer_cfg, &infer_res);
		if (status != flabsdk::Status::kSuccess) {
			std::cerr << "Failed to run inference: " << static_cast<int>(status) << std::endl;
			engine->ClearResources();
			delete engine;
			return -1;
		}
	}
	time_t end_time = time(NULL);
	std::cout << "Inference time: " << difftime(end_time, start_time) / infer_times << " seconds" << std::endl;

	drawResult(input_mat.clone(), infer_res);

	std::cout << "Inference completed successfully." << std::endl;
	engine->ClearResources();

	flabsdk::DestroyInferEngine(engine);

	std::cout << "Engine destroyed successfully." << std::endl;

	getchar();

	//flabsdk::CreateInferEngine("0001", &engine);

	//std::cout << "start create engine" << std::endl;

	return 0;
}