#include "tasks/seg_engine.h"
#include "utils/ops_utils.h"
#include "utils/base_funcs.hpp"
#include "opencv2/core/types.hpp"
#include "tasks/io_structures.h"
#include "status.h"
#include "utils/io_utils.h"
#include <iostream>
#include <spdlog/spdlog.h>

namespace flabsdk {
	namespace seg_infer {
		Status SegInferEngine::CreateInferAssets(const std::string& cfg_path, modules::InferAssets& assets) {
			std::string task = "segment";
			auto status = createInferAssets(cfg_path, assets, task);
			return status;
		}

		Status SegInferEngine::LoadResources(const std::string& cfg_path) {
			assets_ = std::make_shared<modules::InferAssets>();
			Status status = CreateInferAssets(cfg_path, *assets_);
			if (status != Status::kSuccess) {
				spdlog::error("LoadResources failed {}", static_cast<int>(status));
				return status;
			}
			return Status::kSuccess;
		}

		Status SegInferEngine::ClearResources() {
			assets_.reset();
			return Status::kSuccess;
		}

		void init_record_info(SegRecordInfo* record_info, const cv::Mat& img, const flabio::SegInferCfg* rt_cfg) {
			record_info->mask_vec.clear();
			record_info->img = img.clone();
			record_info->seg_infer_cfg = *rt_cfg;
		}

		Status SegInferEngine::InferSync(const cv::Mat& input_data, const flabio::BaseInferCfg* infer_config, flabio::BaseInferRes* infer_result) {
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

			auto rt_cfg = static_cast<const flabio::SegInferCfg*>(infer_config);
			auto infer_rois = rt_cfg->infer_rois;
			if (infer_rois.empty()) {
				spdlog::error("The input roi is empty. Please give a valid roi.");
				return Status::kInputInvalid;
			}
			static_cast<flabio::SegInferRes*>(infer_result)->masks_vec.resize(infer_rois.size());
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

				SegRecordInfo record_info;
				init_record_info(&record_info, infer_img, rt_cfg);
				
				auto status = RunModels(*assets_, &record_info, false);
				if (status != Status::kSuccess) {
					spdlog::error("Run single roi {} failed {}", i, static_cast<int>(status));
					continue;
				}

				//if (rt_cfg->sort_method == 1) {
				//	std::sort(record_info.bbox_vec.begin(), record_info.bbox_vec.end(),
				//		[](const flabio::RotateBox& a, const flabio::RotateBox& b) {
				//			return a.x > b.x;
				//		});
				//}
				//else if (rt_cfg->sort_method == 2) {
				//	std::sort(record_info.bbox_vec.begin(), record_info.bbox_vec.end(),
				//		[](const flabio::RotateBox& a, const flabio::RotateBox& b) {
				//			return a.y > b.y;
				//		});
				//}

				std::vector<flabio::Mask> masks;
				size_t cur_idx = 0;
				size_t max_idx = rt_cfg->max_num > 0 ? rt_cfg->max_num : record_info.mask_vec.size();
				for (auto& mask : record_info.mask_vec) {
					for (auto& point : mask.points) {
						point.x += roi.x;
						point.y += roi.y;
					}
					masks.push_back(mask);
					cur_idx++;
					if (cur_idx >= max_idx) {
						break; // Stop if we reach the max number of Segections
					}
				}
				static_cast<flabio::SegInferRes*>(infer_result)->masks_vec[i] = masks;
				valid_flag = valid_flag || true;
			}
			if (!valid_flag) {
				spdlog::info("No valid result found in the input image, please check out the input.");
				return Status::kOutputInvalid;
			}
			return Status::kSuccess;
		}

	} // namespace seg_infer
} // namespace flabsdk