# Fast GPU Segment Anything Pipeline with ONNXRuntime in C++ 🦾



## Benefits ✨

- General Purpose implementation (in the future abstraction for different NN plague and play pipelines will be implemented)
- Friendly for deployment in the industrial sector.
- Faster than OpenCV's DNN inference on both CPU and GPU.
- Supports FP32 and FP16 CUDA acceleration.



## Dependencies ⚙️

| Dependency                       | Version       |
| -------------------------------- | ------------- |
| Onnxruntime(linux,windows,macos) | >=1.14.1      |
| OpenCV                           | >=4.0.0       |
| C++ Standard                     | >=17          |
| Cmake                            | >=3.5         |
| Cuda (Optional)                  | >=12.8        |
| cuDNN (Cuda required)            | =9            |

Note: The dependency on C++17 is due to the usage of the C++17 filesystem feature.

Note (2): Due to ONNX Runtime, we need to use CUDA 12(.8) and cuDNN 9. Keep in mind that this requirement might change in the future.


## Build 🛠️

0. You can just use run install ```console ./install.sh```. For manual installation.

1. Clone the repository to your local machine.

2. Navigate to the root directory of the repository.

3. Create a build directory and navigate to it SAM or YOLO:

   ```console
   mkdir build && cd build
   ```

4. Run CMake to generate the build files:

   ```console
   cmake ..
   ```

   **Notice**:

   If you encounter an error indicating that the `ONNXRUNTIME_ROOT` variable is not set correctly, you can resolve this by building the project using the appropriate command tailored to your system.

   ```console
   # compiled in a win32 system
   cmake -D WIN32=TRUE ..
   # compiled in a linux system
   cmake -D LINUX=TRUE ..
   # compiled in an apple system
   cmake -D APPLE=TRUE ..
   ```

5. Build the project:

   ```console
   make
   ```

6. The built executable should now be located in the `build` directory.

## Usage of e.g. YOLO 🚀
   ```console
   ./Yolov8OnnxRuntimeCPPInference
   ```

   **Notice**:
   Make sure you have an image to on the build/image folder.
