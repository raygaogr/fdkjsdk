#pragma once

#include <memory>
#include <string>

#include "export.h"
#include "status.h"
#include "tasks/io_structures.h"
#include "opencv2/core/types.hpp"

namespace flabsdk {

/**
 * @brief The InferEngine class is the base class for all inference engines.
 */
class FLABINFER_EXPORT InferEngine {
 public:
  InferEngine() = default;
  InferEngine(const InferEngine&) = delete;
  InferEngine& operator=(const InferEngine&) = delete;
  virtual ~InferEngine() = default;

  /**
   * @brief Initializes the log file.
   * 
   * @param log_path The path to the log file.
   * @return Status The status of the operation.
   */
  Status InitLog(const std::string& log_path);

  /**
   * @brief Loads the resources required for inference.
   * 
   * @param cfg_path The path to the config file for the resources.
   * @return Status The status of the operation.
   */
  virtual Status LoadResources(const std::string& cfg_path) = 0;

  /**
   * @brief Clears the resources used for inference.
   * 
   * @return Status The status of the operation.
   */
  virtual Status ClearResources() = 0;

  /**
   * @brief Performs synchronous inference.
   * 
   * @param input_data The input data to perform inference on.
   * @param infer_config The configuration for the inference.
   * @param infer_result[out] The result of the inference.
   * @return Status The status of the operation.
   */
  virtual Status InferSync(const cv::Mat& input_data, const flabio::BaseInferCfg* infer_config, flabio::BaseInferRes* infer_result) = 0;

};


/**
 * @brief Creates an inference engine for the specified model.
 *
 * @param model_id The uid of the model is used to create the engine.
 * @param engine[out] Pointer to the created InferEngine object.
 * @return Status The status of the operation.
 */
Status FLABINFER_EXPORT CreateInferEngine(const std::string& model_id, InferEngine** engine);


/**
 * @brief Destroys the specified InferEngine object.
 * 
 * @param engine The InferEngine object to destroy.
 * @return Status The status of the operation.
 */
Status FLABINFER_EXPORT DestroyInferEngine(InferEngine* engine);



Status FLABINFER_EXPORT GetPlatformInfo(flabio::PlatformInfo* platform_info);

}  // namespace flabsdk

