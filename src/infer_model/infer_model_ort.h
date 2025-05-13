#pragma once

#include "status.h"
#include <memory>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

namespace flabsdk {
	namespace infer_env {

		struct InferEnv {
			Ort::Env env{ nullptr };
			Ort::Session session{ nullptr };
			std::vector<std::string> input_names;
			std::vector<std::vector<std::int64_t>> input_shapes;
			std::vector<std::string> output_names;
			std::vector<std::vector<std::int64_t>> output_shapes;
		};

		Status CreateInferEnv(const char* model_data, const size_t model_data_length,
			const std::string& cache_dir, std::string& device,
			std::shared_ptr<InferEnv> env);

		Status DestroyInferEnv(std::shared_ptr<InferEnv> env);

		Status RunInfer(const std::shared_ptr<InferEnv> env,
			std::vector<std::vector<float>>& input_data, std::vector<std::vector<float>>& output_data);

	} // namespace infer_env
} // namespace flabsdk
