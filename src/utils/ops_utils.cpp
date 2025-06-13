#include "utils/ops_utils.h"
//#include "tasks/det_engine.h"
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "utils/registry.h"
#include <spdlog/spdlog.h>
#include "utils/record_info.h"
#include "yaml-cpp/yaml.h"

using json = nlohmann::json;

namespace flabsdk {
		json convertYamlToJson(const YAML::Node& node) {
			if (node.IsMap()) {
				json obj;
				for (auto it = node.begin(); it != node.end(); ++it) {
					obj[it->first.as<std::string>()] = convertYamlToJson(it->second);
				}
				return obj;
			}
			else if (node.IsSequence()) {
				json arr;
				for (const auto& item : node) {
					arr.push_back(convertYamlToJson(item));
				}
				return arr;
			}
			else if (node.IsScalar()) {
				return node.as<std::string>(); // 自动推断类型
			}
			return json(); // 默认返回空对象
		}

		Status LoadModels(std::vector<char>& cfgs_vec, modules::InferAssets& assets, bool is_json_format, std::string& task) {
			if (is_json_format) {
				assets.cfgs_ = json::parse(cfgs_vec.data(), cfgs_vec.data() + cfgs_vec.size());
			}
			else {
				auto Node = YAML::Load(cfgs_vec.data());
				assets.cfgs_ = convertYamlToJson(Node);
			}
			std::unordered_map<std::string, std::string> id2task = {
				{"0001", "detect"},
				{"0002", "detect"},
				{"0003", "segment"}
			};
			for (auto& node : assets.cfgs_["graph"]["main"]) {
				std::string model_id = node["model_id"].get<std::string>();
				spdlog::info("Load Module: {}", model_id);
				std::string module_task = id2task[model_id];
				if (!assets.cfgs_["modules"].contains(model_id)) {
					spdlog::info("Module {} not found in cfgs", model_id);
					return Status::kNotFound;
				}
				if (module_task != task) {
					spdlog::info("Module {} task {} does not match expected engine task {}", model_id, module_task, task);
					return Status::kInputInvalid;
				}
				std::string module_class = assets.cfgs_["modules"][model_id]["class"];
				spdlog::info("Module class: {}", module_class);
				assets.modules_[model_id] = ModuleRegistry::instance().get_module(module_class);
				auto status = assets.modules_[model_id]->init(assets.cfgs_["modules"][model_id]["init_params"]);
				if (status != Status::kSuccess) { 
					assets.modules_.clear();
					return status; 
				}
			}
			return Status::kSuccess;
		}

		Status run_graph(const nlohmann::json& graph, const std::string start_point, const std::unordered_map<std::string, std::shared_ptr<modules::BaseModule>>& modules, RecordInfo* record_info, bool verbose = false) {
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

		Status RunModels(modules::InferAssets& assets, RecordInfo* record_info, bool verbose) {
			auto status = run_graph(assets.cfgs_["graph"], "main", assets.modules_, record_info, verbose);
			return status;
		}

} // namespace flabsdk