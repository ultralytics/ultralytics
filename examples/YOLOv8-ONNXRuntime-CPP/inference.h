#pragma once

#define _CRT_SECURE_NO_WARNINGS
#define	RET_OK nullptr

#include <string>
#include <vector>
#include <stdio.h>
#include "io.h"
#include "direct.h"
#include "opencv.hpp"
#include <Windows.h>
#include "onnxruntime_cxx_api.h"


enum MODEL_TYPE
{
	//FLOAT32 MODEL
	YOLO_ORIGIN_V5 = 0,
	YOLO_ORIGIN_V8 = 1,//only support v8 detector currently
	YOLO_POSE_V8 = 2,
	YOLO_CLS_V8 = 3
};


typedef struct _DCSP_INIT_PARAM
{
	std::string								ModelPath;
	MODEL_TYPE								ModelType = YOLO_ORIGIN_V8;
	std::vector<int>						imgSize={640, 640};

	int										classesNum=80;
	float									RectConfidenceThreshold = 0.6;
	float									iouThreshold = 0.5;
	bool									CudaEnable = false;
	int										LogSeverityLevel = 3;
	int										IntraOpNumThreads = 1;
}DCSP_INIT_PARAM;


typedef struct _DCSP_RESULT
{
	int classId;
	float confidence;
	cv::Rect box;
}DCSP_RESULT;


class DCSP_CORE
{
public:
	DCSP_CORE();
	~DCSP_CORE();

public:
	char* CreateSession(DCSP_INIT_PARAM &iParams);


	char* RunSession(cv::Mat &iImg, std::vector<DCSP_RESULT>& oResult);


	char* WarmUpSession();


	template<typename N>
	char* TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DCSP_RESULT>& oResult);


private:
	Ort::Env				env;
	Ort::Session*			session;
	bool					cudaEnable;
	Ort::RunOptions			options;
	std::vector<const char*> inputNodeNames;
	std::vector<const char*> outputNodeNames;


    int						classesNum;
	MODEL_TYPE				modelType;
	std::vector<int>		imgSize;
	float					rectConfidenceThreshold;
	float					iouThreshold;
};
