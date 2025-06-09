#pragma once
#include <nlohmann/json.hpp>
#include <unordered_map>
#include "infer_modules/base_module.h"
#include "status.h"
#include "opencv2/core/types.hpp"
#include "utils/record_info.h"
#include "tasks/det_engine.h"

namespace flabsdk {

		Status LoadModels(std::vector<char>& cfgs_vec, modules::InferAssets& assets, bool is_json_format);

		Status RunModels(modules::InferAssets& assets, RecordInfo* record_info, bool verbose);

} // namespace flabsdk
