#include "infer_modules/det_module.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <unordered_map>
#include "tasks/io_structures.h"
#include "utils/io_utils.h"
#include "utils/base_funcs.h"
#include <spdlog/spdlog.h>

using json = nlohmann::json;

namespace flabsdk {
	namespace modules {

		struct Object {
			std::vector<float> rbox;
			int label = -1;
			float class_conf = 0.;
		};

		static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right) {
			int i = left;
			int j = right;
			float p = objects[(left + right) / 2].class_conf;

			while (i <= j) {
				while (objects[i].class_conf > p)
					i++;

				while (objects[j].class_conf < p)
					j--;

				if (i <= j) {
					// swap
					std::swap(objects[i], objects[j]);
					i++;
					j--;
				}
			}
#pragma omp parallel sections
			{
#pragma omp section
				{
					if (left < j)
						qsort_descent_inplace(objects, left, j);
				}
#pragma omp section
				{
					if (i < right)
						qsort_descent_inplace(objects, i, right);
				}
			}
		}


		static float single_box_iou(const float* a, const float* b) {
			float inter_xmin = std::max(a[0], b[0]);
			float inter_ymin = std::max(a[1], b[1]);
			float inter_xmax = std::min(a[2], b[2]);
			float inter_ymax = std::min(a[3], b[3]);
			float inter_area = std::max(0.f, inter_xmax - inter_xmin) *
				std::max(0.f, inter_ymax - inter_ymin);
			float a_area = (a[2] - a[0]) * (a[3] - a[1]);
			float b_area = (b[2] - b[0]) * (b[3] - b[1]);
			float iou = inter_area / (a_area + b_area - inter_area);
			return iou;
		}


		static void qsort_descent(std::vector<Object>& objects) {
			if (objects.empty())
				return;
			qsort_descent_inplace(objects, 0, objects.size() - 1);
		}

		
		static void nms(const std::vector<Object>& objects,
			std::vector<int>& picked, float nms_threshold) {
			picked.clear();
			size_t n = objects.size();
			for (int i = 0; i < n; i++) {
				const Object& obja = objects[i];
				int keep = 1;
				for (int j = 0; j < (int)picked.size(); j++) {
					const Object& objb = objects[picked[j]];

					// intersection over union
					if (objb.label == obja.label) {
						float ovr = single_box_iou(obja.rbox.data(), objb.rbox.data());
						if (ovr > nms_threshold)
							keep = 0;
					}
				}
				if (keep)
					picked.push_back(i);
			}
		}


		Status DetInferModule::preprocess(cv::Mat img, int input_h, int input_w, std::vector<float>& input_vec) {
			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_32FC3);
			cv::resize(img, img, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);

			input_vec.resize(input_h * input_w * 3);
			float* img_data_ptr = reinterpret_cast<float*>(img.data);
			// hwc to chw
			for (int c = 0; c < 3; ++c) {
				for (int h = 0; h < input_h; ++h) {
					for (int w = 0; w < input_w; ++w) {
						input_vec[c * input_h * input_w + h * input_w + w] =
							img_data_ptr[h * input_w * 3 + w * 3 + c] / 255.;
					}
				}
			}
			return Status::kSuccess;
		}


		Status DetInferModule::postprocess(std::vector<float>& input_vec, std::vector<int64_t>& output_shape, int img_h, int img_w, int input_h, int input_w, det_infer::RecordInfo& record_info) {
			if (id_cfgs_.empty()) {
				spdlog::info("Class id cfgs is empty, skip the infer process.");
				return Status::kInputInvalid;
			}

			int class_num = id_cfgs_.size();
			spdlog::info("Class num: {}", class_num);

			float scale_h = float(img_h) / float(input_h);
			float scale_w = float(img_w) / float(input_w);

			std::vector<Object> proposals;
			for (int64_t j = 0; j < output_shape[2]; j++) {
				float max_score = input_vec[4 * output_shape[2] + j];
				int class_id = 0;
				for (int i = 1; i < class_num; i++) {
					if (input_vec[(4 + i) * output_shape[2] + j] > max_score) {
						max_score = input_vec[(4 + i) * output_shape[2] + j];
						class_id = i;
					}
				}
				if (max_score >= record_info.infer_cfg.conf) {
					Object temp_obj;
					float cx = input_vec[0 * output_shape[2] + j];
					float cy = input_vec[1 * output_shape[2] + j];
					float bw = input_vec[2 * output_shape[2] + j];
					float bh = input_vec[3 * output_shape[2] + j];

					float xmin = (cx - bw / 2) * scale_w; // rescale to ori img size
					float ymin = (cy - bh / 2) * scale_h;
					float xmax = (cx + bw / 2) * scale_w;
					float ymax = (cy + bh / 2) * scale_h;

					xmin = std::max(0.f, xmin);
					ymin = std::max(0.f, ymin);
					xmax = std::min(float(img_w), xmax);
					ymax = std::min(float(img_h), ymax);

					temp_obj.rbox = {xmin, ymin, xmax, ymax};
					temp_obj.label = class_id;
					temp_obj.class_conf = max_score;
					proposals.emplace_back(temp_obj);
				}
			}
			qsort_descent(proposals);
			std::vector<int> picked;
			nms(proposals, picked, record_info.infer_cfg.iou);

			size_t count = picked.size();
			if (count == 0) {
				spdlog::info("No object detected.");
				return Status::kOutputInvalid;
			}
			for (size_t i = 0; i < count; i++) {
				Object obj = proposals[picked[i]];
				det_infer::DetResult single_box;
				single_box.angle = 0;
				single_box.box.push_back(obj.rbox[0]);
				single_box.box.push_back(obj.rbox[1]);
				single_box.box.push_back(obj.rbox[2]);
				single_box.box.push_back(obj.rbox[3]);
				single_box.score = obj.class_conf;
				single_box.id = std::to_string(obj.label);
				single_box.name = id_cfgs_[single_box.id].get<std::string>();
				record_info.det_infors.emplace_back(single_box);
			}
			return Status::kSuccess;
			//std::cout << "det_infors size: " << record_info.det_infors.size() << std::endl;
			//for (const auto& det : record_info.det_infors) {
			//	std::cout << "det_infors: " << det.id << " " << det.name << " " << det.score << std::endl;
			//	for (const auto& box : det.box) {
			//		std::cout << box << " ";
			//	}
			//	std::cout << std::endl;
			//}
		}


		Status DetInferModule::init(const nlohmann::json& init_params) {
			spdlog::info("DetInferModule init");
			std::string cache_dir = "cache_dir";
			auto model_path_vec = init_params["model_path"].get<std::vector<std::string>>();
			if (model_path_vec.empty()) {
				spdlog::error("model_path is empty");
				return Status::kInputInvalid;
			}
			std::cout << "444444444444" << std::endl;
			std::vector<std::shared_ptr<infer_env::InferEnv>> det_models;
			cfgs_ = init_params["cfgs"];
			id_cfgs_ = init_params["id_cfgs"];
			auto device = cfgs_["device"].get<std::string>();
			for (const auto& model_path : model_path_vec) {
				std::cout << model_path << std::endl;
				std::vector<char> model_str;
				auto status = readFileStream(model_path, model_str);
				if (status != Status::kSuccess) {
					spdlog::error("read model file failed");
					return status;
				}
				auto det_model = std::make_shared<infer_env::InferEnv>();
				status = infer_env::CreateInferEnv(model_str.data(), model_str.size(), cache_dir, device, det_model);
				det_models.emplace_back(det_model);
			}
			spdlog::info("Create infer model success");
			model_["models"] = det_models;
			return Status::kSuccess;
		}


		Status DetInferModule::run(det_infer::RecordInfo& record_info) {
			const int img_w = record_info.img.cols;
			const int img_h = record_info.img.rows;
			if (img_w == 0 || img_h == 0) {
				spdlog::info("The input image is invalid, skip the infer process.");
				return Status::kInputInvalid;
			}

			std::shared_ptr<infer_env::InferEnv> infer_model = model_["models"][0];
			auto input_shape = (infer_model->input_shapes)[0];

			std::vector<std::vector<float>> input_vec(1);
			auto status = preprocess(record_info.img, input_shape[2], input_shape[3], input_vec[0]);
			if (status != Status::kSuccess) {
				spdlog::error("Preprocess failed");
				return status;
			}

			std::vector<std::vector<float>> output_vec;
			status = infer_env::RunInfer(infer_model, input_vec, output_vec);
			if (status != Status::kSuccess) {
				spdlog::error("Run infer failed");
				return status;
			}

			auto output_shape = (infer_model->output_shapes)[0];
			std::vector<float> output_arr = std::move(output_vec[0]);

			status = postprocess(output_arr, output_shape, img_h, img_w, input_shape[2], input_shape[3], record_info);
			return status;
		}



		Status GlassBracketModule::preprocess(cv::Mat img, int input_h, int input_w, std::vector<float>& input_vec) {

			cv::Mat resized_img;
			letterbox(img, input_w, input_h, resized_img);
			
			resized_img.convertTo(resized_img, CV_32FC3);
			input_vec.resize(input_h * input_w * 3);
			float* img_data_ptr = reinterpret_cast<float*>(resized_img.data);
			// hwc to chw
			for (int c = 0; c < 3; ++c) {
				for (int h = 0; h < input_h; ++h) {
					for (int w = 0; w < input_w; ++w) {
						input_vec[c * input_h * input_w + h * input_w + w] =
							img_data_ptr[h * input_w * 3 + w * 3 + c] / 255.;
					}
				}
			}
			return Status::kSuccess;
		}


		Status GlassBracketModule::postprocess(std::vector<float>& input_vec, std::vector<int64_t>& output_shape, int img_h, int img_w, int input_h, int input_w, det_infer::RecordInfo& record_info) {
			if (id_cfgs_.empty()) {
				spdlog::info("Class id cfgs is empty, skip the infer process.");
				return Status::kInputInvalid;
			}

			int class_num = id_cfgs_.size();
			spdlog::info("Class num: {}", class_num);

			float r = std::min(input_w / (img_w * 1.0), input_h / (img_h * 1.0));
			int resize_w = round(r * static_cast<float>(img_w));
			int resize_h = round(r * static_cast<float>(img_h));
			float dw = (input_w - resize_w) / 2.;
			float dh = (input_h - resize_h) / 2.;
			int top = round((dh - 0.1) / r);
			int left = round((dw - 0.1) / r);

			std::vector<Object> proposals;
			for (int64_t j = 0; j < output_shape[2]; j++) {
				float max_score = input_vec[4 * output_shape[2] + j];
				int class_id = 0;
				for (int i = 1; i < class_num; i++) {
					if (input_vec[(4 + i) * output_shape[2] + j] > max_score) {
						max_score = input_vec[(4 + i) * output_shape[2] + j];
						class_id = i;
					}
				}
				if (max_score >= record_info.infer_cfg.conf) {
					Object temp_obj;
					float cx = input_vec[0 * output_shape[2] + j];
					float cy = input_vec[1 * output_shape[2] + j];
					float bw = input_vec[2 * output_shape[2] + j];
					float bh = input_vec[3 * output_shape[2] + j];

					float xmin = (cx - bw / 2) / r - left; // rescale to ori img size
					float ymin = (cy - bh / 2) / r - top;
					float xmax = (cx + bw / 2) / r - left;
					float ymax = (cy + bh / 2) / r - top;

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
			qsort_descent(proposals);
			std::vector<int> picked;
			nms(proposals, picked, record_info.infer_cfg.iou);

			size_t count = picked.size();
			if (count == 0) {
				spdlog::info("No object detected.");
				return Status::kOutputInvalid;
			}
			for (size_t i = 0; i < count; i++) {
				Object obj = proposals[picked[i]];
				det_infer::DetResult single_box;
				single_box.angle = 0;
				single_box.box.push_back(obj.rbox[0]);
				single_box.box.push_back(obj.rbox[1]);
				single_box.box.push_back(obj.rbox[2]);
				single_box.box.push_back(obj.rbox[3]);
				single_box.score = obj.class_conf;
				single_box.id = std::to_string(obj.label);
				single_box.name = id_cfgs_[single_box.id].get<std::string>();
				record_info.det_infors.emplace_back(single_box);
			}
			return Status::kSuccess;
		}


		Status GlassBracketModule::init(const nlohmann::json& init_params) {
			spdlog::info("GlassBracketModule init");
			std::string cache_dir = "cache_dir";
			auto model_path_vec = init_params["model_path"].get<std::vector<std::string>>();
			if (model_path_vec.empty()) {
				spdlog::error("model_path is empty");
				return Status::kInputInvalid;
			}
			std::vector<std::shared_ptr<infer_env::InferEnv>> det_models;
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
				auto det_model = std::make_shared<infer_env::InferEnv>();
				status = infer_env::CreateInferEnv(model_str.data(), model_str.size(), cache_dir, device, det_model);
				det_models.emplace_back(det_model);
			}
			spdlog::info("Create infer model success");
			model_["models"] = det_models;
			return Status::kSuccess;
		}


		Status GlassBracketModule::run(det_infer::RecordInfo& record_info) {
			const int img_w = record_info.img.cols;
			const int img_h = record_info.img.rows;
			if (img_w == 0 || img_h == 0) {
				spdlog::info("The input image is invalid, skip the infer process.");
				return Status::kInputInvalid;
			}

			std::shared_ptr<infer_env::InferEnv> infer_model = model_["models"][0];
			auto input_shape = (infer_model->input_shapes)[0];

			std::vector<std::vector<float>> input_vec(1);
			auto status = preprocess(record_info.img, input_shape[2], input_shape[3], input_vec[0]);
			if (status != Status::kSuccess) {
				spdlog::error("Preprocess failed");
				return status;
			}

			std::vector<std::vector<float>> output_vec;
			status = infer_env::RunInfer(infer_model, input_vec, output_vec);
			if (status != Status::kSuccess) {
				spdlog::error("Run infer failed");
				return status;
			}

			auto output_shape = (infer_model->output_shapes)[0];
			std::vector<float> output_arr = std::move(output_vec[0]);

			status = postprocess(output_arr, output_shape, img_h, img_w, input_shape[2], input_shape[3], record_info);
			return status;
		}



	} // namespace modules
} // namespace flabsdk