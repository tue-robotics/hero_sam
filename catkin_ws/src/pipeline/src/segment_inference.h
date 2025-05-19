#ifndef SEGMENT_INFERENCE_H
#define SEGMENT_INFERENCE_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <filesystem>

// Forward declarations of YOLO and SAM classes
class YOLO_V8;
namespace SEG {
    struct DL_RESULT;
    struct DL_INIT_PARAM;
    enum MODEL_TYPE;
}
struct DL_RESULT;

// Function declarations from segment_inference.cpp
void Detector(YOLO_V8*& p, std::vector<DL_RESULT>& res, std::filesystem::directory_entry& i);
void Classifier(YOLO_V8*& p, std::filesystem::directory_entry& i);
int ReadCocoYaml(YOLO_V8*& p);
void DetectTest();
void ClsTest();

#endif  // SEGMENT_INFERENCE_H
