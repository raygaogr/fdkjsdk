#pragma once

#include "tasks/io_structures.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace flabsdk {
	namespace det_infer {

		struct DetResult {
			std::string id = "000000";
			std::string name = "Null";
			float score = 0;
			std::vector<float> box;
			float angle = 0;

			std::vector<float> get_center_infor() {
				float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
				float center_x = (x1 + x2) / 2, center_y = (y1 + y2) / 2;
				float w = x2 - x1, h = y2 - y1;
				std::vector<float> info = { center_x, center_y, w, h };
				return info;
			}
		};

		struct RecordInfo {
			cv::Mat img;
			
			flabio::DetInferCfg infer_cfg;

			std::vector<DetResult> det_infors;

		};

	}  // namespace det_infer
}  // namespace flabsdk
