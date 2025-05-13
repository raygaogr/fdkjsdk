#pragma once
#include <nlohmann/json.hpp>
#include <unordered_map>
#include "infer_modules/base_module.h"
#include "status.h"
#include "opencv2/core/types.hpp"
#include "tasks/det_record_info.h"
#include "tasks/det_engine.h"

namespace flabsdk {
	namespace det_infer {

		Status LoadModels(std::vector<char>& cfgs_vec, InferAssets& assets);

		Status RunModels(InferAssets& assets, const cv::Mat& img, RecordInfo& record_info, const flabio::DetInferCfg* rt_cfg, bool verbose);

	} // namespace det_infer
} // namespace flabsdk
