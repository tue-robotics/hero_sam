#include <ros/ros.h>
#include "segment_inference.h"  // Header for DetectTest()

int main(int argc, char** argv)
{
    // Initialize ROS node
    ros::init(argc, argv, "segment_inference_node");
    ros::NodeHandle nh;

    ROS_INFO("Running segment inference using YOLO + SAM...");

    // Call the existing function from segment_inference.cpp
    DetectTest();

    ROS_INFO("Inference completed. Shutting down.");

    return 0;
}
