#include <iostream>
#include <stdio.h>

#include <OpenNI.h>
#include <opencv2/opencv.hpp>

#include "OpenNI2OpenCV.h"


int main() {
    OpenNIOpenCV::OpenNI2OpenCV oni;
    if (oni.init() != openni::STATUS_OK){
        printf("Initializatuion failed");
        return 1;
    }
    cv::Mat colorFrame, depthFrame, irFrame;
    for (;;) {
        oni.getDepthFrame(depthFrame);
        oni.getColorFrame(colorFrame);
        oni.getIrFrame(irFrame);
        if(!colorFrame.empty() || !depthFrame.empty() || !irFrame.empty()){
            cv::imshow("Color", colorFrame);
           cv::imshow("Depth", depthFrame);
           cv::imshow("IR", irFrame);
        }
        if (cv::waitKey(10) == 27)   break;
    }
    openni::OpenNI::shutdown();
}
