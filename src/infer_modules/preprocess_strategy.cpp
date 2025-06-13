#include "infer_modules/preprocess_strategy.h"
#include "utils/base_funcs.hpp"
#include "spdlog/spdlog.h"

namespace flabsdk {
	namespace modules {
		Status DirectResizePreProcess::preprocess(cv::Mat& img, std::vector<std::vector<std::int64_t>>& input_shapes, std::vector<std::vector<float>>& input_vec) {
			int input_h = input_shapes[0][2];
			int input_w = input_shapes[0][3];
			cv::Mat img_bgr;
			cv::cvtColor(img, img_bgr, cv::COLOR_BGR2RGB);
			cv::resize(img_bgr, img_bgr, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
			auto input_temp = std::vector<float>(input_h * input_w * 3, 0.0f);
			convertHWC2CHW(img_bgr, input_h, input_w, 1.0f / 255.0f, input_temp);
			input_vec.emplace_back(std::move(input_temp));
			return Status::kSuccess;
		}

		Status LetterboxPreProcess::preprocess(cv::Mat& img, std::vector<std::vector<std::int64_t>>& input_shapes, std::vector<std::vector<float>>& input_vec) {
			int input_h = input_shapes[0][2];
			int input_w = input_shapes[0][3];
			cv::Mat resized_img;
			letterbox(img, input_w, input_h, resized_img);

			auto input_temp = std::vector<float>(input_h * input_w * 3, 0.0f);
			std::chrono::milliseconds duration;
			auto start = std::chrono::high_resolution_clock::now();
			convertHWC2CHW(resized_img, input_h, input_w, 1.0f / 255.0f, input_temp);
			auto end = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			spdlog::info("Change the channel overlaps {} ms", duration.count());
			
			input_vec.emplace_back(std::move(input_temp));
			return Status::kSuccess;
		}
	}
}