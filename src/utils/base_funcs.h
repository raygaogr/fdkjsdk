#pragma once

#include <opencv2/opencv.hpp>

namespace flabsdk {

	void letterbox(cv::Mat& img, int input_w, int input_h, cv::Mat& out_img);
	bool is_cuda_available();
	bool is_cudnn_available();

}