#include "ros_segment_inference.h"
void Detector(YOLO_V8*& p, std::vector<DL_RESULT>& res, const cv::Mat& img) {


            p->RunSession(img, res);

            for (auto& re : res)
            {
                cv::RNG rng(cv::getTickCount());
                cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

                cv::rectangle(img, re.box, color, 3);

                float confidence = floor(100 * re.confidence) / 100;
                std::cout << std::fixed << std::setprecision(2);
                std::string label = p->classes[re.classId] + " " +
                    std::to_string(confidence).substr(0, std::to_string(confidence).size() - 4);

                cv::rectangle(
                    img,
                    cv::Point(re.box.x, re.box.y - 25),
                    cv::Point(re.box.x + label.length() * 15, re.box.y),
                    color,
                    cv::FILLED
                );

                cv::putText(
                    img,
                    label,
                    cv::Point(re.box.x, re.box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.75,
                    cv::Scalar(0, 0, 0),
                    2
                );


            }
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();

}


void Classifier(YOLO_V8*& p, const cv::Mat& img)
{

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

            std::vector<DL_RESULT> res;
            const char* ret = p->RunSession(img, res);

            float positionY = 50;
            for (int i = 0; i < res.size(); i++)
            {
                int r = dis(gen);
                int g = dis(gen);
                int b = dis(gen);
                cv::putText(img, std::to_string(i) + ":", cv::Point(10, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                cv::putText(img, std::to_string(res.at(i).confidence), cv::Point(70, positionY), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(b, g, r), 2);
                positionY += 50;
            }

            cv::imshow("TEST_CLS", img);
            cv::waitKey(0);
            cv::destroyAllWindows();
            //cv::imwrite("E:\\output\\" + std::to_string(k) + ".png", img);
        //}

    //}
}



int ReadCocoYaml(YOLO_V8*& p) {
    // Open the YAML file
    std::ifstream file("coco.yaml");
    if (!file.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    // Read the file line by line
    std::string line;
    std::vector<std::string> lines;
    while (std::getline(file, line))
    {
        lines.push_back(line);
    }

    // Find the start and end of the names section
    std::size_t start = 0;
    std::size_t end = 0;
    for (std::size_t i = 0; i < lines.size(); i++)
    {
        if (lines[i].find("names:") != std::string::npos)
        {
            start = i + 1;
        }
        else if (start > 0 && lines[i].find(':') == std::string::npos)
        {
            end = i;
            break;
        }
    }

    // Extract the names
    std::vector<std::string> names;
    for (std::size_t i = start; i < end; i++)
    {
        std::stringstream ss(lines[i]);
        std::string name;
        std::getline(ss, name, ':'); // Extract the number before the delimiter
        std::getline(ss, name); // Extract the string after the delimiter
        names.push_back(name);
    }

    p->classes = names;
    return 0;
}


void DetectTest(const cv::Mat& img)
{

    ////////////////////////// YOLO //////////////////////////////////////
    YOLO_V8* yoloDetector = new YOLO_V8;
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params;
    params.rectConfidenceThreshold = 0.1;
    params.iouThreshold = 0.5;
    params.modelPath = "yolo11m.onnx";
    params.imgSize = { 640, 640 };
#ifdef USE_CUDA
    params.cudaEnable = true;

    // GPU FP32 inference
    params.modelType = YOLO_DETECT_V8;
    // GPU FP16 inference
    //Note: change fp16 onnx model
    //params.modelType = YOLO_DETECT_V8_HALF;

#else
    // CPU inference
    params.modelType = YOLO_DETECT_V8;
    params.cudaEnable = false;

#endif

    yoloDetector->CreateSession(params);
    ////////////////////////// SAM //////////////////////////////////////
    SAM* samSegmentorEncoder = new SAM;
    SAM* samSegmentorDecoder = new SAM;
    SEG::DL_INIT_PARAM params1;
    SEG::DL_INIT_PARAM params2;

    params1.rectConfidenceThreshold = 0.1;
    params1.modelPath = "SAM_encoder.onnx";
    params1.imgSize = { 1024, 1024 };


#ifdef USE_CUDA
    params1.cudaEnable = true;
#else
    params1.cudaEnable = false;
#endif
    samSegmentorEncoder->CreateSession(params1);
    params2 = params1;
    params2.modelType = SEG::SAM_SEGMENT_DECODER;
    params2.modelPath = "SAM_mask_decoder.onnx";
    samSegmentorDecoder->CreateSession(params2);


    std::vector<SEG::DL_RESULT> resSam;
            ////////////////////////// YOLO //////////////////////////////////////
            std::vector<DL_RESULT> resYolo;
            Detector(yoloDetector, resYolo, img);

            ////////////////////////// SAM //////////////////////////////////////
            SEG::DL_RESULT res;
            SEG::MODEL_TYPE modelTypeRef = params1.modelType;


            samSegmentorEncoder->RunSession(img, resSam, modelTypeRef, res);

            // Make sure we have at least one result

            for (const auto& result : resYolo) {
                res.boxes.push_back(result.box);
            }

            modelTypeRef = params2.modelType;
            samSegmentorDecoder->RunSession(img, resSam, modelTypeRef, res);
            std::cout << "Press any key to exit" << std::endl;
            cv::imshow("Result of Detection", img);
            cv::waitKey(0);
            cv::destroyAllWindows();

            }



void ClsTest()
{
    YOLO_V8* yoloDetector = new YOLO_V8;
    std::string model_path = "cls.onnx";
    ReadCocoYaml(yoloDetector);
    DL_INIT_PARAM params{ model_path, YOLO_CLS, {224, 224} };
    yoloDetector->CreateSession(params);
    //Classifier(yoloDetector);
}