#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "infer_modules/base_module.h"
#include "infer_modules/det_module.h"
#include "infer_model/infer_model_ort.h"
#include "opencv2/core/types.hpp"
#include "utils/registry.h"

namespace flabsdk {
	namespace modules {

		class WeldSegInferModule : public DetInferModule {
		public:
			WeldSegInferModule() = default;
			virtual ~WeldSegInferModule() = default;

			Status init(const nlohmann::json& init_params);

		};
		REGISTER_MODULE("WeldSegInferModule", WeldSegInferModule);

	}  // namespace modules
}  // namespace flabsdk