#include "base_funcs.h"

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


}