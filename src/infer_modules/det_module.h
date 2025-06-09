#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include "infer_modules/base_module.h"
#include "infer_model/infer_model_ort.h"
#include "opencv2/core/types.hpp"

namespace flabsdk {
	namespace modules {

		class DetInferModule : public BaseModule {
		public:
			DetInferModule() = default;
			virtual ~DetInferModule() = default;
			Status init(const nlohmann::json& init_params) override;
			Status run(RecordInfo* record_info) override;

		private:
			Status preprocess(cv::Mat img, int input_h, int input_w, std::vector<float>& input_vec);
			Status postprocess(std::vector<float>& input_vec, std::vector<int64_t>& output_shape, int img_h, int img_w, int input_h, int input_w, DetRecordInfo* record_info);
			nlohmann::json cfgs_;
			nlohmann::json id_cfgs_;
			std::unordered_map<std::string, std::vector<std::shared_ptr<infer_env::InferEnv>>> model_;
		};

		class GlassBracketModule : public BaseModule {
		public:
			GlassBracketModule() = default;
			~GlassBracketModule() = default;
			Status init(const nlohmann::json& init_params) override;
			Status run(RecordInfo* record_info) override;

		private:
			Status preprocess(cv::Mat img, int input_h, int input_w, std::vector<float>& input_vec);
			Status postprocess(std::vector<float>& input_vec, std::vector<int64_t>& output_shape, int img_h, int img_w, int input_h, int input_w, DetRecordInfo* record_info);
			nlohmann::json cfgs_;
			nlohmann::json id_cfgs_;
			std::unordered_map<std::string, std::vector<std::shared_ptr<infer_env::InferEnv>>> model_;
		};


	}  // namespace modules
}  // namespace flabsdk