#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>
#include "flabsdk.h"
#include "tasks/det_engine.h"
#include "utils/base_funcs.h"
#include <iostream>

namespace flabsdk {

	Status InferEngine::InitLog(const std::string& log_path) {
		if (log_path.empty()) {
			return Status::kInputInvalid;
		}
		try {
			auto logger = spdlog::get(log_path);
			if (logger == nullptr) {
				logger =
					spdlog::rotating_logger_mt(log_path, log_path, 1024 * 1024 * 10, 3);
			}
			logger->flush_on(spdlog::level::info);
			spdlog::set_default_logger(logger);
		}
		catch (const std::exception& e) {
			spdlog::info("Init log failed, err: {}", e.what());
			return Status::kInitLogFailed;
		}
		spdlog::set_pattern("[%Y-%m-%d %H:%M:%S] [%l] %v");
		return Status::kSuccess;
	}


	Status FLABINFER_EXPORT CreateInferEngine(const std::string& model_id, InferEngine** engine) {
		if (model_id == "0001" || model_id == "0002") {
			*engine = new det_infer::DetInferEngine();
			if (*engine == nullptr) {
				spdlog::info("Create the engine failed.");
				return Status::kOutputInvalid;
			}
			return Status::kSuccess;
		}
		else {
			spdlog::info("Invalid model id.");
			return Status::kInputInvalid;
		}
	}

	Status FLABINFER_EXPORT DestroyInferEngine(InferEngine* engine) {
		if (engine == nullptr) {
			spdlog::info("The engine is null, please check out the input.");
			return Status::kInputInvalid;
		}
		delete engine;
		engine = nullptr;
		spdlog::info("Destroy the engine success.");
		return Status::kSuccess;
	}

	Status FLABINFER_EXPORT GetPlatformInfo(flabio::PlatformInfo* platform_info) {
		platform_info->platform = "CUDA";
		platform_info->version = "12.6";
		platform_info->cudnn_version = "9.10";
		if (is_cuda_available())
			platform_info->is_cuda_matched = true;
		if (is_cudnn_available())
			platform_info->is_cudnn_matched = true;
		return Status::kSuccess;
	}

}  // namespace rsinfer
