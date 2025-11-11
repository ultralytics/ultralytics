//
// Created by wangh on 2025/11/4.
//

#ifndef ONNX_RUNNER_STRUCT_DEF_H
#define ONNX_RUNNER_STRUCT_DEF_H
#include <string>
typedef struct  {
    double x;
    double y;
}Point ;

typedef struct  {
    Point startPoint;
    Point endPoint;
    std::string boxName;
    double score;
}OttCheckAns;
#endif //ONNX_RUNNER_STRUCT_DEF_H
