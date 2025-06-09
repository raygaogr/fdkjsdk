#pragma once

#include "flabsdk.h"
#include "infer_modules/base_module.h"
#include "utils/record_info.h"


namespace flabsdk {
	namespace det_infer {
		class DetInferEngine : public InferEngine {
		public:
			DetInferEngine() = default;
			~DetInferEngine() override = default;

			Status LoadResources(const std::string& cfg_path) override;
			Status ClearResources() override;
			Status InferSync(const cv::Mat& input_data, const flabio::BaseInferCfg* infer_config, flabio::BaseInferRes* infer_result) override;
			Status CreateInferAssets(const std::string& cfg_path, modules::InferAssets& assets);

		private:
			std::shared_ptr<modules::InferAssets> assets_ = nullptr;
		};

	} // namespace det_infer
} // namespace rsinfer