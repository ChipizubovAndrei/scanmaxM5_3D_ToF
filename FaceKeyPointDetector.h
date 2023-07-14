#ifndef FACEKEYPOINTDETECTOR_H
#define FACEKEYPOINTDETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>


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
    void predict(cv::Mat& image, std::vector<cv::Rect2i> boxes)
    {
        // Variable for landmarks.
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we
        // use a vector of vector of points.
        std::vector< std::vector<cv::Point2f> > landmarks;

        // Run landmark detector
        bool success = m_facemark->fit(image, boxes, landmarks);

        if(success){
            // If successful, render the landmarks on the face
            for(int i = 0; i < landmarks.size(); i++){
//                cv::rectangle(image, boxes[i], cv::Scalar(255, 0, 0));
                cv::face::drawFacemarks(image, landmarks[i], cv::Scalar(0, 255, 0));
            }
        }
    }
};

/*
    AAM модель делектора ключевых точек на лице
*/
class FaceKeyPointDetectorAAM
{
private:
    cv::Ptr<cv::face::Facemark> m_facemark;
    // Путь к параметрам LBF модели
    cv::String m_lbf_model_path = "lbfmodel.yaml";

public:

    FaceKeyPointDetectorAAM()
    {
        m_facemark = cv::face::FacemarkLBF::create();
        m_facemark->loadModel(m_lbf_model_path);


    };
    ~FaceKeyPointDetectorAAM(){};

    /*
        Функция определения ключевых точек на лице
        Аргументы:
            - image - изображение
            - boxes - массив с координатами bounding boxes полученных
            с детектора лиц
    */
    void predict(cv::Mat& image, std::vector<cv::Rect2i> boxes)
    {
        // Variable for landmarks.
        // Landmarks for one face is a vector of points
        // There can be more than one face in the image. Hence, we
        // use a vector of vector of points.
        std::vector< std::vector<cv::Point2f> > landmarks;

        // Run landmark detector
        bool success = m_facemark->fit(image, boxes, landmarks);

        if(success){
            // If successful, render the landmarks on the face
            for(int i = 0; i < landmarks.size(); i++){
//                cv::rectangle(image, boxes[i], cv::Scalar(255, 0, 0));
                cv::face::drawFacemarks(image, landmarks[i], cv::Scalar(0, 255, 0));
            }
        }
    }
};

#endif // FACEKEYPOINTDETECTOR_H
