#include "infer_modules/seg_module.h"
#include "infer_modules/postprocess_strategy.h"
#include "infer_modules/preprocess_strategy.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <unordered_map>
#include "tasks/io_structures.h"
#include "utils/io_utils.h"
#include "utils/base_funcs.hpp"
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>

using json = nlohmann::json;

namespace flabsdk {
	namespace modules {

		Status WeldSegInferModule::init(const nlohmann::json& init_params) {
			auto status = init_model_and_cfgs(init_params);
			if (status != Status::kSuccess) {
				spdlog::error("Init model and cfgs failed");
				return status;
			}
			preprocess_strategy_ = std::make_shared<LetterboxPreProcess>();
			postprocess_strategy_ = std::make_shared<YOLOSegPostProcess>();
			spdlog::info("WeldSegInferModule initialized successfully");
			return Status::kSuccess;
		}

	} // namespace modules
} // namespace flabsdk