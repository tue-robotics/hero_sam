#pragma once

#define    RET_OK nullptr

#ifdef _WIN32
#include <Windows.h>
#include <direct.h>
#include <io.h>
#endif

#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"
#include "dl_types.h"
#ifdef USE_CUDA
#include <cuda_fp16.h>
#endif

class Utils
{
    public:
        Utils();
        ~Utils();

        void overlay(std::vector<Ort::Value>& output_tensors, cv::Mat& iImg, std::vector<int> iImgSize, std::vector<DL_RESULT>& oResult);
        char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg);
        void ScaleBboxPoints(cv::Mat& iImg, std::vector<int> iImgSize, std::vector<float>& pointCoords, std::vector<float>& PointsCoordsScaled);

        std::vector<Ort::Value> PrepareInputTensor(Ort::Value& decoderInputTensor, std::vector<float>& pointCoordsScaled, std::vector<int64_t> pointCoordsDims,
                                                    std::vector<float>& pointLabels, std::vector<int64_t> pointLabelsDims, std::vector<float>& maskInput,
                                                    std::vector<int64_t> maskInputDims, std::vector<float>& hasMaskInput, std::vector<int64_t> hasMaskInputDims);

        // Definition: Flattened image to blob (and normalizaed) for deep learning inference. Also reorganize from HWC to CHW.
        // Note: Code in header file since it is used outside of this utils src code.
        template<typename T>
        char* BlobFromImage(cv::Mat& iImg, T& iBlob) {
            int channels = iImg.channels();
            int imgHeight = iImg.rows;
            int imgWidth = iImg.cols;

            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < imgHeight; h++)
                {
                    for (int w = 0; w < imgWidth; w++)
                    {
                        iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = typename std::remove_pointer<T>::type(
                            (iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
                    }
                }
            }
            return RET_OK;
        }
        private:
            float resizeScales;
            float resizeScalesBbox; //letterbox scale


    };
