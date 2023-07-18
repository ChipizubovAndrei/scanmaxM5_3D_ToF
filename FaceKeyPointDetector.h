#ifndef FACEKEYPOINTDETECTOR_H
#define FACEKEYPOINTDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/opencv.h>

#include "Utils.h"


// Face Key Point Detector
namespace FaceKPDetector{

/*
    LBF модель делектора ключевых точек на лице
*/
class FaceKeyPointDetectorLBF
{
private:
    cv::Ptr<cv::face::Facemark> m_facemark;
    // Путь к параметрам LBF модели
    cv::String m_lbf_model_path = "lbfmodel.yaml";

public:

    FaceKeyPointDetectorLBF()
    {
        m_facemark = cv::face::FacemarkLBF::create();
        m_facemark->loadModel(m_lbf_model_path);


    };
    ~FaceKeyPointDetectorLBF(){};

    /*
        Функция определения ключевых точек на лице
        Аргументы:
            - image - изображение
            - boxes - массив с координатами bounding boxes полученных
            с детектора лиц
    */
    std::vector< std::vector<cv::Point2f> > predict(cv::Mat& image, std::vector<cv::Rect2i> boxes)
    {
        // Variable for landmarks.
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we
        // use a vector of vector of points.
        std::vector< std::vector<cv::Point2f> > landmarks;

        // Run landmark detector
        bool success = m_facemark->fit(image, boxes, landmarks);

//        if(success){
//            // If successful, render the landmarks on the face
//            for(int i = 0; i < landmarks.size(); i++){
////                cv::rectangle(image, boxes[i], cv::Scalar(255, 0, 0));
//                cv::face::drawFacemarks(image, landmarks[i], cv::Scalar(0, 255, 0));
//            }
//        }
        return landmarks;
    }
};

/*
    Модель на основе сети YuNet для детекции ключевых точек на изображении
    Ссылка на модель: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
*/
class YuNetDetector
{
private:
    // Путь к параметрам модели
    cv::String m_fd_modelPath = "face_detection_yunet_2023mar.onnx";
    // Filter out faces of score < m_scoreThreshold
    float m_scoreThreshold = 0.9;
    // Параметр отсеивающий bounding boxes с IoU < m_nmsThreshold
    float m_nmsThreshold = 0.3;
    int m_topK = 5000;
    // Параметр определяющий степень увеличения изображения
    float m_scale = 1.0;

    cv::Ptr<cv::FaceDetectorYN> m_detector;

public:
    YuNetDetector()
    {
        m_detector = cv::FaceDetectorYN::create(m_fd_modelPath, "", cv::Size(320, 320),
                                                m_scoreThreshold, m_nmsThreshold, m_topK);
    };
    ~YuNetDetector(){};

    std::vector <std::vector<cv::Point2i>> predict(cv::Mat& image)
    {

        int imageWidth = int(image.cols * m_scale);
        int imageHeight = int(image.rows * m_scale);
        resize(image, image, cv::Size(imageWidth, imageHeight));

        // Установка значений размера входного изображения
        m_detector->setInputSize(image.size());
        cv::Mat faces;

        m_detector->detect(image, faces);
        std::vector <std::vector<cv::Point2i>> landmarks(faces.rows);
        if (faces.rows >= 1){
            for (int i = 0; i < faces.rows; i++){
                // Формирование массива с координатами ключевых точек
                std::vector<cv::Point2i> currLandmark;
                currLandmark.push_back(cv::Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))));
                currLandmark.push_back(cv::Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))));
                currLandmark.push_back(cv::Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))));
                currLandmark.push_back(cv::Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))));
                currLandmark.push_back(cv::Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))));
                landmarks[i] = currLandmark;
            }
        }
        return landmarks;
    }

};

class DlibDetector
{
private:
    dlib::shape_predictor m_sp;

public:

    DlibDetector(){
        dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> m_sp;
    }

    ~DlibDetector(){}

    std::vector< std::vector <cv::Point2i>> predict(cv::Mat& image, std::vector<cv::Rect2i> boxes)
    {
        dlib::cv_image<dlib::rgb_pixel> dImage(image);
        std::vector<dlib::rectangle> dBoxes(boxes.size());
        for (int i = 0; i < boxes.size(); i++){
            dBoxes[i] = openCVRectToDlib(boxes[i]);
        }
        std::vector <std::vector <cv::Point2i>> landmarks(dBoxes.size());
        for (unsigned int i = 0; i < dBoxes.size(); i++){
            dlib::full_object_detection shape = m_sp(dImage, dBoxes[i]);
            std::vector<cv::Point2i> currLandmarks;
            for (int j = 0; j < shape.num_parts(); j++){
                currLandmarks.push_back(cv::Point(int(shape.part(j).x()),
                                                  int(shape.part(j).y())));
            }
            landmarks[i] = currLandmarks;
        }
        return landmarks;
    }
};
}

#endif // FACEKEYPOINTDETECTOR_H
