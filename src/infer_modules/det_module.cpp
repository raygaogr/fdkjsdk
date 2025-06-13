#include "infer_modules/det_module.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <unordered_map>
#include "tasks/io_structures.h"
#include "utils/io_utils.h"
#include "utils/base_funcs.hpp"
#include <spdlog/spdlog.h>

using json = nlohmann::json;

namespace flabsdk {
	namespace modules {

		struct Object {
			std::vector<float> rbox;
			int label = -1;
			float class_conf = 0.;
		};

		Status DetInferModule::init_model_and_cfgs(const nlohmann::json& init_params) {
			spdlog::info("Start init model and cfgs");
			std::string cache_dir = "cache_dir";
			auto model_path_vec = init_params["model_path"].get<std::vector<std::string>>();
			if (model_path_vec.empty()) {
				spdlog::error("model_path is empty");
				return Status::kInputInvalid;
			}
			std::vector<std::shared_ptr<infer_env::InferEnv>> models;
			cfgs_ = init_params["cfgs"];
			id_cfgs_ = init_params["id_cfgs"];
			auto device = cfgs_["device"].get<std::string>();
			for (const auto& model_path : model_path_vec) {
				std::vector<char> model_str;
				auto status = readFileStream(model_path, model_str);
				if (status != Status::kSuccess) {
					spdlog::error("read model file failed");
					return status;
				}
				auto model = std::make_shared<infer_env::InferEnv>();
				status = infer_env::CreateInferEnv(model_str.data(), model_str.size(), cache_dir, device, model);
				if (status != Status::kSuccess) {
					spdlog::error("Create infer env failed");
					return status;
				}
				models.emplace_back(model);
			}
			model_["models"] = models;
			spdlog::info("Create infer model success");
			return Status::kSuccess;
		}

		Status DetInferModule::init(const nlohmann::json& init_params) {
			auto status = init_model_and_cfgs(init_params);
			if (status != Status::kSuccess) {
				spdlog::error("Init model and cfgs failed");
				return status;
			}
			preprocess_strategy_ = std::make_shared<DirectResizePreProcess>();
			postprocess_strategy_ = std::make_shared<YOLODetPostProcess>();
			return Status::kSuccess;
		}


		Status DetInferModule::run(RecordInfo* record_info) {
			const int img_w = record_info->img.cols;
			const int img_h = record_info->img.rows;
			if (img_w == 0 || img_h == 0) {
				spdlog::info("The input image is invalid, skip the infer process.");
				return Status::kInputInvalid;
			}

			std::shared_ptr<infer_env::InferEnv> infer_model = model_["models"][0];
			std::vector<std::vector<std::int64_t>> input_shapes = infer_model->input_shapes;
			
			
			std::chrono::milliseconds duration;
			auto start = std::chrono::high_resolution_clock::now();
			std::vector<std::vector<float>> input_vec;
			auto status = preprocess_strategy_->preprocess(record_info->img, input_shapes, input_vec);
			if (status != Status::kSuccess) {
				spdlog::error("Preprocess failed");
				return status;
			}
			auto end = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			spdlog::info("Do the preprocess overlaps {} ms", duration.count());

			std::vector<std::vector<float>> output_vec;
			status = infer_env::RunInfer(infer_model, input_vec, output_vec);
			if (status != Status::kSuccess) {
				spdlog::error("Run infer failed");
				return status;
			}

			start = std::chrono::high_resolution_clock::now();
			std::vector<std::vector<std::int64_t>> output_shapes = infer_model->output_shapes;
			status = postprocess_strategy_->postprocess(output_vec, output_shapes, img_h, img_w, input_shapes, id_cfgs_, record_info);
			end = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			spdlog::info("Do the postprocess overlaps {} ms", duration.count());
			return status;
		}


		Status GlassBracketModule::init(const nlohmann::json& init_params) {
			auto status = init_model_and_cfgs(init_params);
			if (status != Status::kSuccess) {
				spdlog::error("Init model and cfgs failed");
				return status;
			}
			preprocess_strategy_ = std::make_shared<LetterboxPreProcess>();
			postprocess_strategy_ = std::make_shared<YOLOLetterboxPostProcess>();
			return Status::kSuccess;
		}


	} // namespace modules
} // namespace flabsdk