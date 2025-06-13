#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

namespace flabsdk {
	namespace flabio {
		struct PlatformInfo {
			std::string platform = "";
			std::string version = "";
			std::string cudnn_version = "";
			bool is_cuda_matched = false;
			bool is_cudnn_matched = false;
		};


		////////////////////////////////////////////////////////////////////////////////
		//                            Basic struct, task agnostic                      //
		////////////////////////////////////////////////////////////////////////////////
		struct ROI {
			int x = 0;          // center point x coordinate
			int y = 0;          // center point y coordinate  
			int width = 0;      // width of the ROI
			int height = 0;     // height of the ROI
			float angle = 0;    // the angle of the ROI
		};

		struct RotateBox {
			int x = 0;   // center point x coordinate
			int y = 0;	 // center point y coordinate
			int width = 0;		// width of the bbox
			int height = 0;		// height of the bbox
			float angle = 0.f; // the angle of the bbox
			float score = 0.f; // the score of the bbox
			std::string uid; // unique id of the bbox
			std::string name; // the name of the bbox
		};

		struct Mask {
			std::vector<cv::Point> points; // points of the contour
			std::string uid; // unique id of the mask
			std::string name; // type of the mask
			float score = 0.f; // the score of the mask
		};

		////////////////////////////////////////////////////////////////////////////////
		//                     Input struct, task specific                            //
		////////////////////////////////////////////////////////////////////////////////
		// 通用输入基类
		struct BaseInferCfg {
			virtual ~BaseInferCfg() = default;
			std::vector<ROI> infer_rois;       // ROI region of interest
		};

		struct DetInferCfg : public BaseInferCfg {
			float conf = 0.3f;    // The confidence threshold
			float iou = 0.3f;     // The IOU threshold
			int max_num = -1;    // The max number of objects to detect
			int sort_method = 0;   // The mothod of sorting
		};

		struct SegInferCfg : public BaseInferCfg {
			float conf = 0.3f;    // The confidence threshold
			float iou = 0.3f;     // The IOU threshold
			int max_num = -1;    // The max number of objects to detect
		};

		struct Task3InferCfg : public BaseInferCfg {
			std::vector<ROI> infer_rois;
			std::string label;   // Label to classify
		};


		////////////////////////////////////////////////////////////////////////////////
		//                    Output struct, task specific                            //
		////////////////////////////////////////////////////////////////////////////////
		struct BaseInferRes {
			virtual ~BaseInferRes() = default;
		};

		struct DetInferRes : public BaseInferRes {
			std::vector<std::vector<RotateBox>> bboxes_vec; // bounding boxes result
		};

		struct SegInferRes : public BaseInferRes {
			std::vector<std::vector<Mask>> masks_vec;       // segmentation masks
		};

		struct Task3InferRes : public BaseInferRes {
			std::unordered_map<std::string, float> scores; // class scores
		};
	} // namespace flabio
} // namespace flabsdk