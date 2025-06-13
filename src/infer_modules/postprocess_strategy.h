#pragma once

#include "utils/record_info.h"
#include "status.h"
#include <vector>
#include <nlohmann/json.hpp>

namespace flabsdk {
	namespace modules {
		class PostProcessStrategy {
		public:
			PostProcessStrategy() = default;
			virtual ~PostProcessStrategy() = default;
			virtual Status postprocess(std::vector<std::vector<float>>& output_vec, std::vector<std::vector<int64_t>>& output_shapes, int img_h, int img_w,
				std::vector<std::vector<int64_t>>& input_shapes, nlohmann::json id_cfgs_, RecordInfo* record_info) = 0;
		};

		class YOLODetPostProcess : public PostProcessStrategy {
		public:
			YOLODetPostProcess() = default;
			virtual ~YOLODetPostProcess() = default;
			virtual Status postprocess(std::vector<std::vector<float>>& output_vec, std::vector<std::vector<int64_t>>& output_shapes, int img_h, int img_w,
				std::vector<std::vector<int64_t>>& input_shapes, nlohmann::json id_cfgs_, RecordInfo* record_info) override;
			Status do_process(int img_h, int img_w, int input_h, int input_w,
				int num_proposals, nlohmann::json id_cfgs_, bool use_letterbox,
				std::vector<float>& output_arr, RecordInfo* record_info);
		};

		class YOLOLetterboxPostProcess : public YOLODetPostProcess {
		public:
			YOLOLetterboxPostProcess() = default;
			~YOLOLetterboxPostProcess() = default;
			Status postprocess(std::vector<std::vector<float>>& output_vec, std::vector<std::vector<int64_t>>& output_shapes, int img_h, int img_w,
				std::vector<std::vector<int64_t>>& input_shapes, nlohmann::json id_cfgs_, RecordInfo* record_info) override;
		};

		class YOLOSegPostProcess : public PostProcessStrategy{
		public:
			YOLOSegPostProcess() = default;
			virtual ~YOLOSegPostProcess() = default;
			Status postprocess(std::vector<std::vector<float>>& output_vec, std::vector<std::vector<int64_t>>& output_shapes, int img_h, int img_w,
				std::vector<std::vector<int64_t>>& input_shapes, nlohmann::json id_cfgs_, RecordInfo* record_info) override;
		};


	} // namespace modules
} // namespace flabsdk