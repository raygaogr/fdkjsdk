#include "infer_modules/postprocess_strategy.h"
#include "utils/base_funcs.hpp"
#include "spdlog/spdlog.h"

namespace flabsdk {
	namespace modules {
		struct Object {
			std::vector<float> rbox;
			int label = -1;
			float class_conf = 0.;
		};

		struct SegObject : Object {
			cv::Mat mask_in;
		};

		Status YOLODetPostProcess::do_process(int img_h, int img_w, int input_h, int input_w,
			int num_proposals, nlohmann::json id_cfgs_, bool use_letterbox,
			std::vector<float>& output_arr, RecordInfo* record_info) {

			if (id_cfgs_.empty()) {
				spdlog::info("Class id cfgs is empty, skip the infer process.");
				return Status::kInputInvalid;
			}
			int class_num = id_cfgs_.size();

			float scale_h, scale_w;
			if (use_letterbox) {
				scale_h = scale_w = std::min(input_w / (img_w * 1.0f), input_h / (img_h * 1.0f));
			}
			else {
				scale_h = float(input_h) / float(img_h);
				scale_w = float(input_w) / float(img_w);
			}
			int resize_w = round(scale_w * static_cast<float>(img_w));
			int resize_h = round(scale_h * static_cast<float>(img_h));
			float dw = (input_w - resize_w) / 2.;
			float dh = (input_h - resize_h) / 2.;
			int top = round((dh - 0.1) / scale_h);
			int left = round((dw - 0.1) / scale_w);

			std::vector<Object> proposals;
			for (int64_t j = 0; j < num_proposals; j++) {
				float max_score = output_arr[4 * num_proposals + j];
				int class_id = 0;
				for (int i = 1; i < class_num; i++) {
					if (output_arr[(4 + i) * num_proposals + j] > max_score) {
						max_score = output_arr[(4 + i) * num_proposals + j];
						class_id = i;
					}
				}
				if (max_score >= static_cast<DetRecordInfo*>(record_info)->det_infer_cfg.conf) {
					Object temp_obj;
					float cx = output_arr[0 * num_proposals + j];
					float cy = output_arr[1 * num_proposals + j];
					float bw = output_arr[2 * num_proposals + j];
					float bh = output_arr[3 * num_proposals + j];

					float xmin = (cx - bw / 2) / scale_w - left; // rescale to ori img size
					float ymin = (cy - bh / 2) / scale_h - top;
					float xmax = (cx + bw / 2) / scale_w - left;
					float ymax = (cy + bh / 2) / scale_h - top;

					xmin = std::max(0.f, xmin);
					ymin = std::max(0.f, ymin);
					xmax = std::min(float(img_w), xmax);
					ymax = std::min(float(img_h), ymax);

					temp_obj.rbox = { xmin, ymin, xmax, ymax };
					temp_obj.label = class_id;
					temp_obj.class_conf = max_score;
					proposals.emplace_back(temp_obj);
				}
			}
			if (proposals.empty()) {
				spdlog::info("No object detected.");
				return Status::kOutputInvalid;
			}
			qsort_descent_inplace<Object>(proposals, 0, proposals.size() - 1);
			std::vector<int> picked;
			nms<Object>(proposals, picked, static_cast<DetRecordInfo*>(record_info)->det_infer_cfg.iou);

			size_t count = picked.size();
			if (count == 0) {
				spdlog::info("No object detected.");
				return Status::kOutputInvalid;
			}
			for (size_t i = 0; i < count; i++) {
				Object obj = proposals[picked[i]];
				std::vector<float> rbox = {
					obj.rbox[0], obj.rbox[1], obj.rbox[2], obj.rbox[3]
				};
				auto bbox = get_center_infor(rbox);
				flabio::RotateBox single_box;
				single_box.angle = 0;
				single_box.score = obj.class_conf;
				single_box.uid = std::to_string(obj.label);
				single_box.name = id_cfgs_[single_box.uid].get<std::string>();
				single_box.x = bbox[0];
				single_box.y = bbox[1];
				single_box.width = bbox[2];
				single_box.height = bbox[3];
				static_cast<DetRecordInfo*>(record_info)->bbox_vec.emplace_back(single_box);
			}
			return Status::kSuccess;
		}

		Status YOLODetPostProcess::postprocess(std::vector<std::vector<float>>& output_vec, std::vector<std::vector<int64_t>>& output_shapes,
			int img_h, int img_w, std::vector<std::vector<int64_t>>& input_shapes, nlohmann::json id_cfgs_,
			RecordInfo* record_info) {
			auto status = do_process(img_h, img_w, input_shapes[0][2], input_shapes[0][3],
				output_shapes[0][2], id_cfgs_, false, output_vec[0], record_info);
			return status;
		}

		Status YOLOLetterboxPostProcess::postprocess(std::vector<std::vector<float>>& output_vec, std::vector<std::vector<int64_t>>& output_shapes,
			int img_h, int img_w, std::vector<std::vector<int64_t>>& input_shapes, nlohmann::json id_cfgs_,
			RecordInfo* record_info) {
			auto status = do_process(img_h, img_w, input_shapes[0][2], input_shapes[0][3],
				output_shapes[0][2], id_cfgs_, true, output_vec[0], record_info);
			return status;
		}

		Status YOLOSegPostProcess::postprocess(std::vector<std::vector<float>>& output_vec, std::vector<std::vector<int64_t>>& output_shapes,
			int img_h, int img_w, std::vector<std::vector<int64_t>>& input_shapes, nlohmann::json id_cfgs_,
			RecordInfo* record_info) {
			if (id_cfgs_.empty()) {
				spdlog::info("Class id cfgs is empty, skip the infer process.");
				return Status::kInputInvalid;
			}

			int class_num = id_cfgs_.size();
			spdlog::info("Class num: {}", class_num);

			int input_h = input_shapes[0][2];
			int input_w = input_shapes[0][3];
			float r = std::min(input_w / (img_w * 1.0f), input_h / (img_h * 1.0f));
			int resize_w = round(r * static_cast<float>(img_w));
			int resize_h = round(r * static_cast<float>(img_h));
			float dw = (input_w - resize_w) / 2.f;
			float dh = (input_h - resize_h) / 2.f;
			int top = round((dh - 0.1f) / r);
			int left = round((dw - 0.1f) / r);
			int bottom = round((dh + 0.1f) / r);
			int right = round((dw + 0.1f) / r);

			int mask_w = output_shapes[1][3];
			int mask_h = output_shapes[1][2];
			float r_m = std::min(mask_w / (img_w * 1.0), mask_h / (img_h * 1.0));
			int resize_w_m = round(r_m * static_cast<float>(img_w));
			int resize_h_m = round(r_m * static_cast<float>(img_h));
			float dw_m = (mask_w - resize_w_m) / 2.f;
			float dh_m = (mask_h - resize_h_m) / 2.f;
			int top_m = round(dh_m - 0.1f);
			int left_m = round(dw_m - 0.1f);
			int bottom_m = round(mask_h - dh_m + 0.1f);
			int right_m = round(mask_w - dw_m + 0.1f);

			auto box_vec = output_vec[0];
			auto mask_vec = output_vec[1];

			cv::Mat box_mat(output_shapes[0][1], output_shapes[0][2], CV_32F, box_vec.data());
			cv::Mat mask_mat(output_shapes[1][1], mask_w * mask_h, CV_32F, mask_vec.data());
			cv::Mat trans_mat;
			cv::transpose(box_mat, trans_mat);

			std::vector<SegObject> proposals;
			for (int i = 0; i < trans_mat.rows; i++) {
				cv::Mat classes_scores = (trans_mat.row(i).colRange(4, 4 + class_num)).clone();
				cv::Point class_id;
				double max_score;
				cv::minMaxLoc(classes_scores, nullptr, &max_score, nullptr, &class_id);
				if (max_score > static_cast<SegRecordInfo*>(record_info)->seg_infer_cfg.conf) {
					SegObject temp_obj;
					cv::Mat mask_in = (trans_mat.row(i).colRange(4 + class_num, output_shapes[0][1])).clone();
					float cx = trans_mat.at<float>(i, 0);
					float cy = trans_mat.at<float>(i, 1);
					float bw = trans_mat.at<float>(i, 2);
					float bh = trans_mat.at<float>(i, 3);

					float xmin = (cx - bw / 2) / r - left; // rescale to ori img size
					float ymin = (cy - bh / 2) / r - top;
					float xmax = (cx + bw / 2) / r - left;
					float ymax = (cy + bh / 2) / r - top;

					temp_obj.mask_in = mask_in;
					temp_obj.rbox = { xmin, ymin, xmax, ymax };
					temp_obj.class_conf = max_score;
					temp_obj.label = class_id.x;
					proposals.push_back(temp_obj);
				}
			}
			if (proposals.empty()) {
				spdlog::info("No object detected.");
				return Status::kOutputInvalid;
			}
			qsort_descent_inplace<SegObject>(proposals, 0, proposals.size() - 1);
			std::vector<int> picked;
			nms<SegObject>(proposals, picked, static_cast<SegRecordInfo*>(record_info)->seg_infer_cfg.iou);

			size_t count = picked.size();
			if (count == 0) {
				spdlog::info("No object detected.");
				return Status::kOutputInvalid;
			}

			for (size_t i = 0; i < count; i++) {
				SegObject obj = proposals[picked[i]];
				cv::Mat m = obj.mask_in * mask_mat;
				cv::Mat m1 = m.reshape(1, mask_h);
				cv::Mat mask_roi = m1(cv::Range(top_m, bottom_m), cv::Range(left_m, right_m));
				cv::Mat rm;
				cv::resize(mask_roi, rm, cv::Size(img_w, img_h), 0, 0, cv::INTER_LINEAR);
				cv::Mat mask_out = cv::Mat::zeros(img_h, img_w, CV_8U);
				int bias = 20;
				for (int r = 0; r < rm.rows; r++) {
					if (r > obj.rbox[1] - bias && r < obj.rbox[3] + bias) {
						for (int c = 0; c < rm.cols; c++) {
							if (c > obj.rbox[0] - bias && c < obj.rbox[2] + bias) {
								float pv = rm.at<float>(r, c);
								if (pv > 0.0f) {
									mask_out.at<unsigned char>(r, c) = 1;
								}
							}
						}
					}
				}
				std::vector<std::vector<cv::Point>> contours;
				cv::findContours(mask_out, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				if (contours.empty()) {
					spdlog::warn("No contours found for object with box: {}", i);
					continue;
				}

				size_t max_id = 0;
				size_t max_val = contours[0].size();
				for (size_t idx = 1; idx < contours.size(); idx++) {
					if (contours[idx].size() > max_val) {
						max_val = contours[idx].size();
						max_id = idx;
					}
				}
				flabio::Mask temp_mask;
				temp_mask.uid = std::to_string(obj.label);
				temp_mask.name = id_cfgs_[temp_mask.uid];
				temp_mask.score = obj.class_conf;
				temp_mask.points = contours[max_id];
				static_cast<SegRecordInfo*>(record_info)->mask_vec.emplace_back(temp_mask);
			}
			if (static_cast<SegRecordInfo*>(record_info)->mask_vec.empty()) {
				spdlog::info("No mask detected.");
				return Status::kOutputInvalid;
			}
			return Status::kSuccess;
		}

	} // namespace modules
} // namespace flabsdk