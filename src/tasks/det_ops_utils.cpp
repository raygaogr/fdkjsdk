#include "tasks/det_ops_utils.h"
#include "tasks/det_engine.h"
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "utils/registry.h"
#include <spdlog/spdlog.h>
#include "tasks/det_record_info.h"


using json = nlohmann::json;

namespace flabsdk {
	namespace det_infer {

		Status run_graph(const nlohmann::json& graph, const std::string start_point, const std::unordered_map<std::string, std::shared_ptr<modules::BaseModule>>& modules, RecordInfo& record_info, bool verbose = false) {
			if (verbose)
				spdlog::info("run_graph {}", start_point);

			for (const auto& node : graph[start_point]) {
				if (node["type"].get<std::string>() == "Run") {
					std::string model_id = node["model_id"].get<std::string>();
					spdlog::info("Running model id {}", model_id);
					try {
						auto module = modules.at(model_id);
						auto status = module->run(record_info);
						if (status != Status::kSuccess) {
							spdlog::info("Run module {} failed", model_id);
							return status;
						}
					}
					catch (const std::exception& e) {
						spdlog::info("Run module error {}", e.what());
						return Status::kRunInferFailed;
					}
				}
				else if (node["type"].get<std::string>() == "Switch") {
					std::string switch_expr;
					if (node["expr"].get<std::string>() == "sp_cls") {
					}
					else if (node["expr"].get<std::string>() == "mode") {
					}
					if (verbose)
						spdlog::info("Switch ", switch_expr);
					for (const auto& it : node["blocks"].items()) {
						std::string blk_key = it.key();
						std::string blk = it.value();
						if (switch_expr == blk_key) {
							auto status = run_graph(graph, blk, modules, record_info, verbose);
							if (status != Status::kSuccess) {
								spdlog::info("Run graph {} failed", blk);
								return status;
							}
						}
					}
				}
				else {
					spdlog::info("Unknown node type {}", node["type"].get<std::string>());
					return Status::kInputInvalid;
				}
			}
			if (verbose)
				spdlog::info("run_graph end");
			return Status::kSuccess;
		}

		void init_record_info(RecordInfo& record_info, const cv::Mat& img, const flabio::DetInferCfg* rt_cfg) {
			record_info.img = img.clone();
			record_info.det_infors.clear();
			record_info.infer_cfg = *rt_cfg;
		}


		Status LoadModels(std::vector<char>& cfgs_vec, InferAssets& assets) {
			assets.cfgs_ = json::parse(cfgs_vec.data(), cfgs_vec.data() + cfgs_vec.size());
			for (auto& node : assets.cfgs_["graph"]["main"]) {
				std::cout << "3333333333" << std::endl;
				//spdlog::info("Load Module: {}", node["model_id"].get<std::string>());
				std::string model_id = node["model_id"].get<std::string>();
				std::cout << "model id" << model_id << std::endl;
				if (!assets.cfgs_["modules"].contains(model_id)) {
					spdlog::info("Module {} not found in cfgs", model_id);
					return Status::kNotFound;
				}
				std::string module_class = assets.cfgs_["modules"][model_id]["class"];
				//spdlog::info("Module class: {}", module_class);
				std::cout << "model name" << module_class << std::endl;
				assets.modules_[model_id] = std::shared_ptr<modules::BaseModule>(get_module(module_class));
				std::cout << "121211212" << std::endl;
				auto status = assets.modules_[model_id]->init(assets.cfgs_["modules"][model_id]["init_params"]);
				if (status != Status::kSuccess) return status;
			}
			return Status::kSuccess;
		}


		Status RunModels(InferAssets& assets, const cv::Mat& img, RecordInfo& record_info, const flabio::DetInferCfg* rt_cfg, bool verbose) {
			init_record_info(record_info, img, rt_cfg);
			auto status = run_graph(assets.cfgs_["graph"], "main", assets.modules_, record_info, verbose);
			return status;
		}

	} // namespace det_infer
} // namespace flabsdk