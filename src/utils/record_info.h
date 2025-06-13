#pragma once

#include "tasks/io_structures.h"
#include <opencv2/opencv.hpp>
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
