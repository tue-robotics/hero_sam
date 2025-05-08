cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME PipelineCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- ONNXRuntime  ------------------#
set(ONNXRUNTIME_VERSION 1.21.0)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-gpu-1.21.1")
include_directories(${ONNXRUNTIME_ROOT}/include)

# -------------- Cuda ------------------#
add_definitions(-DUSE_CUDA=1)
include_directories(/usr/local/cuda/include)

# Include directories for YOLO and SAM
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/inc
    ${ONNXRUNTIME_ROOT}/include
)

# Add YOLO source files directly if needed
set(PROJECT_SOURCES
    src/segment_inference.cpp
    # Include YOLO files if needed for linking
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/src/yolo_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/src/sam_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/src/utils.cpp
)

# Add executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Link OpenCV libraries along with ONNX Runtime
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)

# For Windows system, copy onnxruntime.dll to the same folder of the executable file
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()

# Copy needed files to the same folder of the executable file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/model/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_mask_decoder.onnx  COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/model/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_encoder.onnx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/data/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/model/yolo11m.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolo11m.onnx COPYONLY)

# Create folder name images in the same folder of the executable file
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
)

#If you want to debug
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
