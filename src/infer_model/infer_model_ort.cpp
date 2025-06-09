#include "infer_model/infer_model_ort.h"
#include <chrono>
#include "spdlog/spdlog.h"
#include <iostream>
#include <sstream>
#include "utils/io_utils.h"
#include "utils/base_funcs.h"


namespace flabsdk {
	namespace infer_env {

		std::string print_shape(const std::vector<std::int64_t>& v) {
			std::stringstream ss("");
			for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
			ss << v[v.size() - 1];
			return ss.str();
		}

		size_t calculate_product(std::vector<std::int64_t>& shape) {
			size_t product = 1;
			for (size_t i = 0; i < shape.size(); i++) {
				product *= shape[i];
			}
			return product;
		}

		template <typename T>
		Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
			Ort::MemoryInfo mem_info =
				Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
			auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
			return tensor;
		}

		Status CreateInferEnv(const char* model_data, const size_t model_data_length,
			const std::string& cache_dir, std::string& device,
			std::shared_ptr<InferEnv> env) {

			env->env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "infer_model");
			Ort::SessionOptions session_options;
			session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

			std::transform(device.begin(), device.end(), device.begin(), ::tolower);
			if (device == "gpu") {
				if (is_cuda_available() && is_cudnn_available()) {
					Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
				}
				else {
					spdlog::info("CUDA or cuDNN is not available.");
					return Status::kNotFound;
				}
			}
			env->session = Ort::Session(env->env, model_data, model_data_length, session_options);
			spdlog::info("Create infer model success");

			Ort::AllocatorWithDefaultOptions allocator;
			for (std::size_t i = 0; i < env->session.GetInputCount(); i++) {
				env->input_names.emplace_back(env->session.GetInputNameAllocated(i, allocator).get());
				auto input_shape = env->session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
				spdlog::info( "\t{} : {}.", env->input_names.at(i), print_shape(input_shape));
				env->input_shapes.emplace_back(input_shape);
			}

			for (std::size_t i = 0; i < env->session.GetOutputCount(); i++) {
				env->output_names.emplace_back(env->session.GetOutputNameAllocated(i, allocator).get());
				auto output_shape = env->session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
				spdlog::info("\t{} : {}", env->output_names.at(i), print_shape(output_shape));
				env->output_shapes.emplace_back(output_shape);
			}
			return Status::kSuccess;
		}


		Status DestroyInferEnv(std::shared_ptr<InferEnv> env) {
			env.reset();
			return Status::kSuccess;
		}


		Status RunInfer(const std::shared_ptr<InferEnv> env,
			std::vector<std::vector<float>>& input_data, std::vector<std::vector<float>>& output_data) {
			spdlog::info("Start RunInferModel");
			std::chrono::milliseconds duration;
			try {
				std::vector<Ort::Value> input_tensors;
				for (size_t i = 0; i < env->input_shapes.size(); i++) {
					input_tensors.emplace_back(vec_to_tensor<float>(input_data[i], env->input_shapes[i]));
				}

				std::vector<const char*> input_names_char(env->input_names.size(), nullptr);
				for (size_t i = 0; i < env->input_names.size(); i++) {
					spdlog::info("input name: {}", env->input_names[i]);
					input_names_char[i] = env->input_names[i].c_str();
				}
				
				std::vector<const char*> output_names_char(env->output_names.size(), nullptr);
				for (size_t i = 0; i < env->output_names.size(); i++) {
					output_names_char[i] = env->output_names[i].c_str();
					spdlog::info("output name: {}", env->output_names[i]);
				}

				auto start = std::chrono::high_resolution_clock::now();
				spdlog::info("Running model ...");
				auto output_tensors = env->session.Run(Ort::RunOptions{ nullptr },
					input_names_char.data(), input_tensors.data(), input_names_char.size(),
					output_names_char.data(), env->output_names.size());
				spdlog::info("Run model success");
				
				for (size_t i = 0; i < env->output_shapes.size(); i++) {
					auto output_elem = calculate_product(env->output_shapes[i]);
					float* output_arr = output_tensors[i].GetTensorMutableData<float>();
					std::vector<float> out{ output_arr, output_arr + output_elem };
					output_data.emplace_back(out);
				}
				auto end = std::chrono::high_resolution_clock::now();
				duration =
					std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			}
			catch (const Ort::Exception& exception) {
				spdlog::error("RunInferModel failed: {}", exception.what());
				return Status::kRunInferFailed;
			}
			spdlog::info("RunInferModel overlaps {} ms", duration.count());
			return Status::kSuccess;
		}

	} // namespace infer_env
} // namespace flabsdk





