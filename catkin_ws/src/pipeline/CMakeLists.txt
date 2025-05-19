cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME PipelineCPPInference)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -------------- OpenCV ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- ONNXRuntime ------------------#
set(ONNXRUNTIME_VERSION 1.21.0)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-gpu-1.21.1")
include_directories(${ONNXRUNTIME_ROOT}/include)

# -------------- Cuda ------------------#
add_definitions(-DUSE_CUDA=1)
include_directories(/usr/local/cuda/include)

# -------------- ROS Dependencies ------------------#
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  image_transport
  cv_bridge
)
include_directories(${catkin_INCLUDE_DIRS})

# -------------- Include Directories ------------------#
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/inc
)

# -------------- Source Files ------------------#
set(PROJECT_SOURCES
    src/segment_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/src/yolo_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/src/sam_inference.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/src/utils.cpp
)

# -------------- Executable 1: Your original program (non-ROS) ------------------#
add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# -------------- Executable 2: ROS node using the same core logic ------------------#
add_executable(segment_inference_node
    src/main.cpp
    ${PROJECT_SOURCES}
)

target_link_libraries(segment_inference_node
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)

# -------------- Windows only: Copy DLL ------------------#
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
        $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif ()

# -------------- Copy model files ------------------#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/model/SAM_mask_decoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_mask_decoder.onnx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../sam_inference/model/SAM_encoder.onnx ${CMAKE_CURRENT_BINARY_DIR}/SAM_encoder.onnx COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/data/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../yolo_inference/model/yolo11m.onnx ${CMAKE_CURRENT_BINARY_DIR}/yolo11m.onnx COPYONLY)

# -------------- Create images folder ------------------#
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/images
)

# -------------- Debug flags ------------------#
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")

