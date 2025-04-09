#pragma once

#include <vector>
#include <string>
#include "inference.h" // Include the YOLO_V8 class and related definitions
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <random>
#include <ros/ros.h>
// Declare the functions from main.cpp
void Detector(YOLO_V8*& p, const cv::Mat& img);
void Classifier(cv::Mat& img);
int ReadCocoYaml(YOLO_V8*& p);
void DetectTest(cv::Mat& img);