#pragma once

#include <opencv2/opencv.hpp>
#include "status.h"
#include "infer_modules/base_module.h"

namespace flabsdk {

	void letterbox(cv::Mat& img, int input_w, int input_h, cv::Mat& out_img);
	void convertHWC2CHW(const cv::Mat& input_img, int input_h, int input_w, float normalized_factor, std::vector<float>& input_vec);

	template <typename T>
	void qsort_descent_inplace(std::vector<T>& objects, int left, int right) {
		int i = left;
		int j = right;
		float p = objects[(left + right) / 2].class_conf;

		while (i <= j) {
			while (objects[i].class_conf > p)
				i++;

			while (objects[j].class_conf < p)
				j--;

			if (i <= j) {
				// swap
				std::swap(objects[i], objects[j]);
				i++;
				j--;
			}
		}
#pragma omp parallel sections
		{
#pragma omp section
			{
				if (left < j)
					qsort_descent_inplace(objects, left, j);
			}
#pragma omp section
			{
				if (i < right)
					qsort_descent_inplace(objects, i, right);
			}
		}
	}

	template <typename T>
	void nms(const std::vector<T>& objects,
		std::vector<int>& picked, float nms_threshold) {
		picked.clear();
		size_t n = objects.size();
		for (int i = 0; i < n; i++) {
			const T& obja = objects[i];
			int keep = 1;
			for (int j = 0; j < (int)picked.size(); j++) {
				const T& objb = objects[picked[j]];

				// intersection over union
				if (objb.label == obja.label) {
					float ovr = single_box_iou(obja.rbox.data(), objb.rbox.data());
					if (ovr > nms_threshold)
						keep = 0;
				}
			}
			if (keep)
				picked.push_back(i);
		}
	}
	
	float single_box_iou(const float* a, const float* b);
	std::vector<float> get_center_infor(std::vector<float>& rbox);

	bool is_cuda_available();
	bool is_cudnn_available();
	std::string GetFileExtension(const std::string& filename);
	Status createInferAssets(const std::string& cfg_path, modules::InferAssets& assets, std::string& task);
}