#pragma once

#include <nlohmann/json.hpp>
#include "tasks/det_record_info.h"
#include "status.h"
//#include "utils/registry.h"

namespace flabsdk {
	namespace modules {

		class BaseModule {
		public:
			BaseModule() = default;
			virtual ~BaseModule() = default;

			virtual Status init(const nlohmann::json& init_params) = 0;
			virtual Status run(det_infer::RecordInfo& record_info) = 0;
		};

		class EmptyModule : public BaseModule {
		public:
			EmptyModule() = default;
			~EmptyModule() = default;

			Status init(const nlohmann::json& init_params) override { cfgs_ = init_params["cfgs"]; return Status::kSuccess; };
			Status run(det_infer::RecordInfo& record_info) override { return Status::kSuccess; };

		private:
			nlohmann::json cfgs_;
		};

		//REGISTER_MODULE("EmptyModule", EmptyModule);
	}
} // namespace flabsdk