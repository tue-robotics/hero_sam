#pragma once

#include <vector>
#include <string>
#include "yolo_inference.h"
#include "sam_inference.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <random>
#include <ros/ros.h>


// Forward declarations
class YOLO_V8;

// Declare the functions from main.cpp
void Detector(YOLO_V8*& p, std::vector<DL_RESULT>& res, const cv::Mat& img);
void Classifier(YOLO_V8*& p, const cv::Mat& img);
int ReadCocoYaml(YOLO_V8*& p);
//void DetectTest(cv::Mat& img);
std::vector<cv::Mat> DetectTest(const cv::Mat& img);
void ClsTest();
void ClassificationInference(const cv::Mat& img);