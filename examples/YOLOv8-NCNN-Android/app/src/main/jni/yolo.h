// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>

#include <net.h>
#include "opencv-mobile-4.6.0-android/sdk/native/jni/include/opencv2/core/matx.hpp"
#include "opencv-mobile-4.6.0-android/sdk/native/jni/include/opencv2/core/types.hpp"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_feat;
};

class Yolo
{
public:
    Yolo();

    int load(AAssetManager* mgr, const char* modeltype, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object>& objects);

    void generate_proposals2(const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

    const std::vector<std::string> _class_names = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"};

private:
    ncnn::Net yolo;
    float _prob_threshold = 0.5;
    float _nms_threshold = 0.45;
    const int _target_size = 640;   //network input size
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

};

#endif // NANODET_H
