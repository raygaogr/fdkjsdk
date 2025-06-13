#pragma once

#include "status.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace flabsdk {
	namespace modules {

		class PreProcessStrategy {
		public:
			PreProcessStrategy() = default;
			virtual ~PreProcessStrategy() = default;
			virtual Status preprocess(cv::Mat& img, std::vector<std::vector<std::int64_t>>& input_shapes, std::vector<std::vector<float>>& input_vec) = 0;
		};

		class DirectResizePreProcess : public PreProcessStrategy {
		public:
			DirectResizePreProcess() = default;
			virtual ~DirectResizePreProcess() = default;
			Status preprocess(cv::Mat& img, std::vector<std::vector<std::int64_t>>& input_shapes, std::vector<std::vector<float>>& input_vec);
		};

		class LetterboxPreProcess : public PreProcessStrategy {
		public:
			LetterboxPreProcess() = default;
			virtual ~LetterboxPreProcess() = default;
			Status preprocess(cv::Mat& img, std::vector<std::vector<std::int64_t>>& input_shapes, std::vector<std::vector<float>>& input_vec) override;
		};
	} // namespace modules
} // namespace flabsdk