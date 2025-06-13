#include "base_funcs.hpp"
#include "spdlog/spdlog.h"
#include "utils/io_utils.h"
#include "utils/ops_utils.h"
#include "opencv2/dnn.hpp"

typedef int (*cudaRuntimeGetVersion_t)(int*);
typedef unsigned int (*cudnnGetVersion_t)();

namespace flabsdk {
	void letterbox(cv::Mat& img, int input_w, int input_h, cv::Mat& out_img) {
		const int ori_w = img.cols;
		const int ori_h = img.rows;
		float r = std::min(input_w / (ori_w * 1.0), input_h / (ori_h * 1.0));
		int resize_w = round(r * static_cast<float>(ori_w));
		int resize_h = round(r * static_cast<float>(ori_h));
		float dw = (input_w - resize_w) / 2.;
		float dh = (input_h - resize_h) / 2.;
		int top = round(dh - 0.1); 
		int bottom = round(dh + 0.1);
		int left = round(dw - 0.1); 
		int right = round(dw + 0.1);

		cv::Mat img_rgb;
		cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
		cv::resize(img_rgb, img_rgb, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
		cv::copyMakeBorder(img_rgb, out_img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	}

	void convertHWC2CHW(const cv::Mat& input_img, int input_h, int input_w, float normalized_factor, std::vector<float>& input_vec) {
		//auto blob = cv::dnn::blobFromImage(input_img, normalized_factor, cv::Size(input_w, input_h), cv::Scalar(0, 0, 0), true, false);
		//int totalElements = blob.total();  // 总元素数 = 1 * C * H * W
		//input_vec = std::vector<float>(blob.ptr<float>(), blob.ptr<float>() + totalElements);
		cv::Mat float_img;
		input_img.convertTo(float_img, CV_32FC3, normalized_factor);
		
		std::vector<cv::Mat> channels;
		cv::split(float_img, channels);

		for (size_t i = 0; i < channels.size(); ++i) {
			channels[i] = channels[i].reshape(1, 1);  // 展平为 1 行
			const float* channel_data = channels[i].ptr<float>(0);
			memcpy(input_vec.data() + i * input_h * input_w, channel_data, input_h * input_w * sizeof(float));
		}
	}

	float single_box_iou(const float* a, const float* b) {
		float inter_xmin = std::max(a[0], b[0]);
		float inter_ymin = std::max(a[1], b[1]);
		float inter_xmax = std::min(a[2], b[2]);
		float inter_ymax = std::min(a[3], b[3]);
		float inter_area = std::max(0.f, inter_xmax - inter_xmin) *
			std::max(0.f, inter_ymax - inter_ymin);
		float a_area = (a[2] - a[0]) * (a[3] - a[1]);
		float b_area = (b[2] - b[0]) * (b[3] - b[1]);
		float iou = inter_area / (a_area + b_area - inter_area);
		return iou;
	}

	std::vector<float> get_center_infor(std::vector<float>& rbox) {
		float x1 = rbox[0], y1 = rbox[1], x2 = rbox[2], y2 = rbox[3];
		float center_x = (x1 + x2) / 2, center_y = (y1 + y2) / 2;
		float w = x2 - x1, h = y2 - y1;
		std::vector<float> info = { center_x, center_y, w, h };
		return info;
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

	std::string GetFileExtension(const std::string& filename) {
		size_t pos = filename.find_last_of('.');
		if (pos == std::string::npos || pos == filename.length() - 1) {
			return ""; // No extension found
		}
		return filename.substr(pos + 1);
	}

	Status createInferAssets(const std::string& cfg_path, modules::InferAssets& assets, std::string& task) {
		Status status;
		std::vector<char> cfgs_str;
		status = readFileStream(cfg_path, cfgs_str, false);
		if (status != Status::kSuccess) {
			spdlog::error("Load the config file failed.");
			return status;
		}
		auto file_format = GetFileExtension(cfg_path);
		bool is_json_format;
		if (file_format == "json") {
			is_json_format = true;
		}
		else if (file_format == "yaml" || file_format == "yml") {
			is_json_format = false;
		}
		else {
			spdlog::error("The config file format is not supported: {}", file_format);
			return Status::kInputInvalid;
		}

		status = LoadModels(cfgs_str, assets, is_json_format, task);
		if (status != Status::kSuccess) {
			spdlog::error("Load models failed {}", static_cast<int>(status));
			return status;
		}
		spdlog::info("Create infer assets success");
		return Status::kSuccess;
	}
}