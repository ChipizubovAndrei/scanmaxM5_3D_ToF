#include <iostream>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <string>

#include <OpenNI.h>
#include <opencv2/opencv.hpp>

#include "OpenNI2OpenCV.h"
#include "FaceKeyPointDetector.h"
#include "FaceDetectors.h"

#include "Utils.h"


int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    OpenNIOpenCV::OpenNI2OpenCV oni;

    FaceBBDetector::YuNetDetector BBdetector;
    FaceKPDetector::DlibDetector KPdetector;

    std::vector<cv::Rect2i> boxes;
    std::vector <std::vector <cv::Point2i>> landmarks;

    if (oni.init() != openni::STATUS_OK){
        printf("Initializatuion failed");
        return 1;
    }
    std::string textFPS;
    int currFPS = 0;
    cv::Mat colorFrame, depthFrame, irFrame;
    auto t1 = high_resolution_clock::now();
    for (;;) {
//        oni.getDepthFrame(depthFrame);
//        oni.getIrFrame(irFrame);
        oni.getColorFrame(colorFrame);
        boxes = BBdetector.predict(colorFrame);
        landmarks = KPdetector.predict(colorFrame, boxes);

        drawLandmarks(colorFrame, landmarks);

        // Вычисление количество FPS
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = (t2 - t1);
        if (ms_double.count() > 1000.0){
            textFPS = "FPS = " + std::to_string(currFPS);
            currFPS = 0;
            t1 = high_resolution_clock::now();
        }
        else{
            currFPS++;
        }
        if(!colorFrame.empty() || !depthFrame.empty() || !irFrame.empty()){
            cv::putText(colorFrame, textFPS, cv::Point(10, 450), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

//           cv::imshow("Depth", depthFrame);
//           cv::imshow("IR", irFrame);
            cv::imshow("Color", colorFrame);
        }
        if (cv::waitKey(10) == 27) break;

    }

    openni::OpenNI::shutdown();
}



