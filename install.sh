#!/bin/bash
# This script installs the required packages for the project.

# Set the installation directory
INSTALL_DIR="$PWD/hero_sam"
if [ ! -d "$INSTALL_DIR" ] ; then
    echo "$INSTALL_DIR does not exist."
    # Create a new directory for the project
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    echo "Installing to $INSTALL_DIR"
else
    echo "$INSTALL_DIR already exists."
    cd "$INSTALL_DIR"
fi


# Define ONNX Runtime version and folder name
ONNX_VERSION="1.21.1"
ONNX_NAME="onnxruntime-linux-x64-gpu-${ONNX_VERSION}"

# Clone the deep learning repository
SAM_DIRECTORY="$PWD/sam_inference"
ONNX_DIRECTORY="$PWD/$ONNX_NAME"


if [ ! -d "$SAM_DIRECTORY" ] ; then
    echo "$SAM_DIRECTORY does not exist."
    git clone git@github.com:utkutahan/hero_sam.git ./
    git checkout feature/LocalImplementation
    git pull

fi


# INstall ONNX Runtime
if [ ! -d "$ONNX_DIRECTORY" ] ; then
    echo "$ONNX_DIRECTORY does not exist."
    wget "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_NAME}.tgz"
    tar -xvzf ${ONNX_NAME}.tgz
    rm -rf ${ONNX_NAME}.tgz
fi


# Replace ONNXRUNTIME_ROOT in CMakeLists.txt
sed -i "s|set(ONNXRUNTIME_ROOT .*|set(ONNXRUNTIME_ROOT \"\${CMAKE_CURRENT_SOURCE_DIR}/../${ONNX_NAME}\")|" sam_inference/CMakeLists.txt
sed -i "s|set(ONNXRUNTIME_ROOT .*|set(ONNXRUNTIME_ROOT \"\${CMAKE_CURRENT_SOURCE_DIR}/../${ONNX_NAME}\")|" yolo_inference/CMakeLists.txt
sed -i "s|set(ONNXRUNTIME_ROOT .*|set(ONNXRUNTIME_ROOT \"\${CMAKE_CURRENT_SOURCE_DIR}/../${ONNX_NAME}\")|" pipeline/CMakeLists.txt

# Build SAM Inference
cd sam_inference && mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=${PWD}/../../${ONNX_NAME} && make -j8
cd ../..
# Build YOLO Inference
cd yolo_inference && mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$PWD/../../onnxruntime-linux-x64-gpu-1.21.1 && make -j8
cd ../..
# Build Pipeline Inference
cd pipeline && mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$PWD/../../${ONNX_NAME} && make -j8

./PipelineCPPInference
