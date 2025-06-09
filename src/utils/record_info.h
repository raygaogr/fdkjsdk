#pragma once

#include "tasks/io_structures.h"
#include "infer_modules/base_module.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace flabsdk {
	struct RecordInfo {
		cv::Mat img;
	};

	struct DetRecordInfo : public RecordInfo {
		flabio::DetInferCfg det_infer_cfg;
		std::vector<flabio::RotateBox> bbox_vec;
	};

	struct SegRecordInfo : public RecordInfo {
		flabio::SegInferCfg seg_infer_cfg;
		std::vector<flabio::Mask> mask_vec;
	};


}  // namespace flabsdk
