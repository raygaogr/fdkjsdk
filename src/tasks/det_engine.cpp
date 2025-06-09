#include "tasks/det_engine.h"
#include "utils/ops_utils.h"
#include "opencv2/core/types.hpp"
#include "tasks/io_structures.h"
#include "status.h"
#include "utils/io_utils.h"
#include <iostream>
#include <spdlog/spdlog.h>

namespace flabsdk {
	namespace det_infer {
		std::string GetFileExtension(const std::string& filename) {
			size_t pos = filename.find_last_of('.');
			if (pos == std::string::npos || pos == filename.length() - 1) {
				return ""; // No extension found
			}
			return filename.substr(pos + 1);
		}

		Status DetInferEngine::CreateInferAssets(const std::string& cfg_path, modules::InferAssets& assets) {
			Status status;
			std::vector<char> cfgs_str;
			status = readFileStream(cfg_path, cfgs_str);
			if (status != Status::kSuccess) {
				spdlog::error("Load the config file failed.");
				return status;
			}
			auto file_format = GetFileExtension(cfg_path);
			bool is_json_format;
			if (file_format == "json") {
				is_json_format = true;
			}
			else if (file_format == "yaml" || file_format == "yml") {
				is_json_format = false;
			}
			else {
				spdlog::error("The config file format is not supported: {}", file_format);
				return Status::kInputInvalid;
			}

			status = LoadModels(cfgs_str, assets, is_json_format);
			if (status != Status::kSuccess) {
				spdlog::error("Load models failed {}", static_cast<int>(status));
				return status;
			}
			spdlog::info("Create infer assets success");
			return Status::kSuccess;
		}

		Status DetInferEngine::LoadResources(const std::string& cfg_path) {
			assets_ = std::make_shared<modules::InferAssets>();
			Status status = CreateInferAssets(cfg_path, *assets_);
			if (status != Status::kSuccess) {
				spdlog::error("LoadResources failed {}", static_cast<int>(status));
				return status;
			}
			return Status::kSuccess;
		}

		Status DetInferEngine::ClearResources() {
			assets_.reset();
			return Status::kSuccess;
		}

		void init_record_info(DetRecordInfo* record_info, const cv::Mat& img, const flabio::DetInferCfg* rt_cfg) {
			record_info->bbox_vec.clear();
			record_info->img = img.clone();
			record_info->det_infer_cfg = *rt_cfg;
		}

		Status DetInferEngine::InferSync(const cv::Mat& input_data, const flabio::BaseInferCfg* infer_config, flabio::BaseInferRes* infer_result) {
			spdlog::info("Start run infer sync.");
			if (assets_ == nullptr || assets_->modules_.empty()) {
				spdlog::error("You should load the resources first.");
				return Status::kWrongState;
			}
			
			cv::Mat input = input_data.clone();
			if (input.empty()) {
				spdlog::error("The input image is empty.");
				return Status::kInputInvalid;
			}

			auto rt_cfg = static_cast<const flabio::DetInferCfg*>(infer_config);
			auto infer_rois = rt_cfg->infer_rois;
			if (infer_rois.empty()) {
				spdlog::error("The input roi is empty. Please give a valid roi.");
				return Status::kInputInvalid;
			}
			static_cast<flabio::DetInferRes*>(infer_result)->bboxes_vec.resize(infer_rois.size());
			bool valid_flag = false;

			for (size_t i = 0; i < infer_rois.size(); ++i) {
				cv::Rect roi;
				cv::Mat infer_img;
				roi = cv::Rect(infer_rois[i].x - int(infer_rois[i].width / 2),
					infer_rois[i].y - int(infer_rois[i].height / 2),
					infer_rois[i].width,
					infer_rois[i].height);
				if (roi.x < 0 || roi.width < 0 || roi.x > input.cols || roi.width > input.cols ||
					roi.y < 0 || roi.height < 0 || roi.y > input.rows || roi.height > input.rows ||
					roi.x + int(roi.width / 2) > input.cols || roi.y + int(roi.height / 2) > input.rows) {
					spdlog::error("The roi {} is out of the image, please check out the input.", i);
					continue;
				}
				spdlog::info("Run a single roi.");
				infer_img = input(roi).clone();

				DetRecordInfo record_info;
				init_record_info(&record_info, infer_img, rt_cfg);
				
				auto status = RunModels(*assets_, &record_info, false);
				if (status != Status::kSuccess) {
					spdlog::error("Run single roi {} failed {}", i, static_cast<int>(status));
					continue;
				}

				if (rt_cfg->sort_method == 1) {
					std::sort(record_info.bbox_vec.begin(), record_info.bbox_vec.end(),
						[](const flabio::RotateBox& a, const flabio::RotateBox& b) {
							return a.x > b.x;
						});
				}
				else if (rt_cfg->sort_method == 2) {
					std::sort(record_info.bbox_vec.begin(), record_info.bbox_vec.end(),
						[](const flabio::RotateBox& a, const flabio::RotateBox& b) {
							return a.y > b.y;
						});
				}

				std::vector<flabio::RotateBox> boxes;
				size_t cur_idx = 0;
				size_t max_idx = rt_cfg->max_num > 0 ? rt_cfg->max_num : record_info.bbox_vec.size();
				for (auto& box : record_info.bbox_vec) {
					box.x += roi.x;
					box.y += roi.y;
					boxes.push_back(box);
					cur_idx++;
					if (cur_idx >= max_idx) {
						break; // Stop if we reach the max number of detections
					}
				}
				static_cast<flabio::DetInferRes*>(infer_result)->bboxes_vec[i] = boxes;
				valid_flag = valid_flag || true;
			}
			if (!valid_flag) {
				spdlog::info("No valid result found in the input image, please check out the input.");
				return Status::kOutputInvalid;
			}
			return Status::kSuccess;
		}

	} // namespace det_infer
} // namespace flabsdk