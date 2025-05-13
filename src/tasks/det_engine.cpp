#include "tasks/det_engine.h"
#include "tasks/det_ops_utils.h"
#include "opencv2/core/types.hpp"
#include "tasks/io_structures.h"
#include "status.h"
#include "utils/io_utils.h"
#include <iostream>
#include <spdlog/spdlog.h>

namespace flabsdk {
	namespace det_infer {

		Status DetInferEngine::CreateInferAssets(const std::string& cfg_path, InferAssets& assets) {
			Status status;
			std::vector<char> cfgs_str;
			status = readFileStream(cfg_path, cfgs_str);
			if (status != Status::kSuccess) {
				spdlog::error("Load the config file failed.");
				return status;
			}
			std::cout << "22222222222" << std::endl;
			status = LoadModels(cfgs_str, assets);
			if (status != Status::kSuccess) {
				spdlog::error("Load models failed {}", static_cast<int>(status));
				return status;
			}
			spdlog::info("Create infer assets success");
			return Status::kSuccess;
		}

		Status DetInferEngine::LoadResources(const std::string& cfg_path) {
			assets_ = std::make_shared<InferAssets>();
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

		Status DetInferEngine::InferSync(const cv::Mat& input_data, const flabio::BaseInferCfg* infer_config, flabio::BaseInferRes* infer_result) {
			spdlog::info("Start run infer sync.");
			if (assets_ == nullptr) {
				spdlog::error("You should load the resources first.");
				return Status::kWrongState;
			}
			
			
			cv::Mat input = input_data.clone();

			RecordInfo record_info;

			auto rt_cfg = static_cast<const flabio::DetInferCfg*>(infer_config);
			auto infer_rois = rt_cfg->infer_rois;

			
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
					spdlog::error("The roi is out of the image, please check out the input.");
					return Status::kInputInvalid;
				}
				spdlog::info("Run a single roi.");
				infer_img = input(roi).clone();

				spdlog::info("get input data success");
				RunModels(*assets_, infer_img, record_info, rt_cfg, false);

				std::vector<flabio::RotateBox> boxes;
				for (auto& x : record_info.det_infors) {
					flabio::RotateBox rbox;
					auto box = x.get_center_infor();
					rbox.x = box[0] + roi.x;
					rbox.y = box[1] + roi.y;
					rbox.width = box[2];
					rbox.height = box[3];
					rbox.angle = x.angle;
					rbox.uid = x.id;
					rbox.name = x.name;
					rbox.score = x.score;
					boxes.push_back(rbox);
				}
				static_cast<flabio::DetInferRes*>(infer_result)->bboxes_vec.emplace_back(boxes);
			}
			return Status::kSuccess;
		}

	} // namespace det_infer
} // namespace flabsdk