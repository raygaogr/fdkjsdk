#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "infer_modules/base_module.h"
#include "infer_model/infer_model_ort.h"
#include "opencv2/core/types.hpp"
#include "utils/registry.h"
#include "infer_modules/preprocess_strategy.h"
#include "infer_modules/postprocess_strategy.h"

namespace flabsdk {
	namespace modules {

		class DetInferModule : public BaseModule {
		public:
			DetInferModule() = default;
			virtual ~DetInferModule() = default;
			virtual Status init(const nlohmann::json& init_params) override;
			Status run(RecordInfo* record_info) override;

			Status init_model_and_cfgs(const nlohmann::json& init_params);
		
		protected:
			nlohmann::json cfgs_;
			nlohmann::json id_cfgs_;
			std::unordered_map<std::string, std::vector<std::shared_ptr<infer_env::InferEnv>>> model_;
			std::shared_ptr<PreProcessStrategy> preprocess_strategy_;
			std::shared_ptr<PostProcessStrategy> postprocess_strategy_;
		};
		REGISTER_MODULE("DetInferModule", DetInferModule);

		class GlassBracketModule : public DetInferModule {
		public:
			GlassBracketModule() = default;
			~GlassBracketModule() = default;
			Status init(const nlohmann::json& init_params) override;

		};
		REGISTER_MODULE("GlassBracketModule", GlassBracketModule);

	}  // namespace modules
}  // namespace flabsdk