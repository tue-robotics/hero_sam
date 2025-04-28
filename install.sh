#!/bin/bash
# This script installs the required packages for the project.

# Define ONNX Runtime version and folder name
ONNX_VERSION="1.21.1"
ONNX_NAME="onnxruntime-linux-x64-gpu-${ONNX_VERSION}"


# Clone the deep learning repository
git clone git@github.com:utkutahan/hero_sam.git hero_sam_test
cd hero_sam_test
git checkout master
git pull

# INstall ONNX Runtime
wget "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_NAME}.tgz"
tar -xvzf ${ONNX_NAME}.tgz
rm -rf ${ONNX_NAME}.tgz


# Replace ONNXRUNTIME_ROOT in CMakeLists.txt
sed -i "s|set(ONNXRUNTIME_ROOT .*|set(ONNXRUNTIME_ROOT \"\${CMAKE_CURRENT_SOURCE_DIR}/../${ONNX_NAME}\")|" sam_inference/CMakeLists.txt
sed -i "s|set(ONNXRUNTIME_ROOT .*|set(ONNXRUNTIME_ROOT \"\${CMAKE_CURRENT_SOURCE_DIR}/../${ONNX_NAME}\")|" yolo_inference/CMakeLists.txt

# Build SAM Inference
cd sam_inference && mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=${PWD}/../../${ONNX_NAME} && make -j8
cd ../..
# Build YOLO Inference
cd yolo_inference && mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$PWD/../../onnxruntime-linux-x64-gpu-1.21.1 && make -j8

./Yolov8OnnxRuntimeCPPInference
