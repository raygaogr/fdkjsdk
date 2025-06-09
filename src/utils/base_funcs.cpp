#include "base_funcs.h"
#include "spdlog/spdlog.h"



typedef int (*cudaRuntimeGetVersion_t)(int*);
typedef unsigned int (*cudnnGetVersion_t)();

namespace flabsdk {

	void letterbox(cv::Mat& img, int input_w, int input_h, cv::Mat& out_img) {
		const int ori_w = img.cols;
		const int ori_h = img.rows;
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

		float r = std::min(input_w / (ori_w * 1.0), input_h / (ori_h * 1.0));
		int resize_w = round(r * static_cast<float>(ori_w));
		int resize_h = round(r * static_cast<float>(ori_h));
		float dw = (input_w - resize_w) / 2.;
		float dh = (input_h - resize_h) / 2.;
		int top = round(dh - 0.1); int bottom = round(dh + 0.1);
		int left = round(dw - 0.1); int right = round(dw + 0.1);

		cv::resize(img, img, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
		cv::copyMakeBorder(img, out_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	}


	bool is_cuda_available() {
		HMODULE handle = LoadLibrary(TEXT("cudart64_12.dll"));
		if (handle == NULL) {
			spdlog::info("Cannot open the cudart library, please check the cuda is available, 12.X is required.");
			return false;
		}
		cudaRuntimeGetVersion_t cudaRuntimeGetVersion = (cudaRuntimeGetVersion_t)GetProcAddress(handle, "cudaRuntimeGetVersion");
		if (cudaRuntimeGetVersion == NULL) {
			spdlog::info("Cannot load function 'cudaRuntimeGetVersion'.");
			FreeLibrary(handle);
			return false;
		}
		int version;
		int result = cudaRuntimeGetVersion(&version);
		if (result == 0) {
			spdlog::info("Get the CUDA Runtime Version: {}.{}", std::to_string(version / 1000), std::to_string((version % 1000) / 10));
		}
		else {
			spdlog::info("Failed to get CUDA runtime version.");
			FreeLibrary(handle);
			return false;
		}

		FreeLibrary(handle);
		return true;
	}

	bool is_cudnn_available() {
		// 尝试打开cuDNN共享库
		HMODULE handle = LoadLibrary(TEXT("cudnn64_9.dll")); // 确保这里是你的cuDNN DLL的实际名称
		if (handle == NULL) {
			spdlog::info("Cannot open cudnn library, please check the cudnn is available, 9.X is required.");
			return false;
		}

		// 加载cudnnGetVersion函数
		cudnnGetVersion_t cudnnGetVersion = (cudnnGetVersion_t)GetProcAddress(handle, "cudnnGetVersion");
		if (cudnnGetVersion == NULL) {
			spdlog::info("Cannot load function 'cudnnGetVersion'.");
			FreeLibrary(handle);
			return false;
		}

		// 使用加载的函数获取cuDNN版本号
		unsigned int version = cudnnGetVersion();
		spdlog::info("cuDNN Version: {}.{}", std::to_string(version / 10000), std::to_string((version % 10000) / 100));

		// 关闭库句柄
		FreeLibrary(handle);
		return true;
	}

}