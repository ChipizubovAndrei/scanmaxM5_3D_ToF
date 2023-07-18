#ifndef FACEDETECTORS_H
#define FACEDETECTORS_H

#include <iostream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

// Face Bounding Box Detector
namespace FaceBBDetector{
/*
    Модель на основе каскадов Хаара для детекции лица на изображении
*/
class HaarCascaadDetector
{
private:
    // Путь к каскадам Хаара
    cv::String m_face_cascade_path = "haarcascade_frontalface_default.xml";
    cv::CascadeClassifier m_faceDetector;

public:
    HaarCascaadDetector()
    {
        m_faceDetector.load(m_face_cascade_path);
    }
    ~HaarCascaadDetector(){};

    /*
        Функция предсказания координат bounding boxes лица
    */
    std::vector<cv::Rect2i> predict(cv::Mat& image){
        // Переменная для хранения изображения в оттенки серого
        cv::Mat gray;

        // Массив с координатами bounding boxes
        std::vector<cv::Rect2i> boxes;
        // Перевод изображения в градации серого
        // так как faceDetector требует изображение в градациях серого.
        cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // Определение координат
        m_faceDetector.detectMultiScale(gray, boxes);
        return boxes;
    }
};

/*
    Модель на основе сети YuNet для детекции лица на изображении
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

    std::vector<cv::Rect> predict(cv::Mat& image)
    {

        int imageWidth = int(image.cols * m_scale);
        int imageHeight = int(image.rows * m_scale);
        resize(image, image, cv::Size(imageWidth, imageHeight));

        // Установка значений размера входного изображения
        m_detector->setInputSize(image.size());
        cv::Mat faces;
        std::vector<cv::Rect2i> boxes;

        m_detector->detect(image, faces);
        if (faces.rows >= 1){
            for (int i = 0; i < faces.rows; i++){
                // Формирование массива с координатами bounding boxes
                boxes.push_back(cv::Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)),
                                           int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))));
            }
        }
        return boxes;
    }

};

}

#endif // FACEDETECTORS_H
