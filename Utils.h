#ifndef UTILS_H
#define UTILS_H

#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <opencv2/opencv.hpp>

/*
    Функция для оттображения ограничивающих рамок
    Аргументы:
        - image - изображение
        - boxes - массив с координатами bounding boxes
*/
void drawLandmarks(cv::Mat& image, std::vector< std::vector<cv::Point2i>> landmarks){
    for (int i = 0; i < landmarks.size(); i++){
        for (int j = 0; j < landmarks[i].size(); j++){
            // Отображение ключевых точек
            cv::circle(image, landmarks[i][j], 1, cv::Scalar(0, 255, 0),    2);
        }
    }
}

/*
    Функция для оттображения ограничивающих рамок
    Аргументы:
        - image - изображение
        - boxes - массив с координатами bounding boxes
*/
void drawBoundingBox(cv::Mat& image, std::vector<cv::Rect2i> boxes){
    for (int i = 0; i < boxes.size(); i++){
        // Отображение bounding boxes
        cv::rectangle(image, boxes[i], cv::Scalar(0, 255, 0), 1);
    }
}

cv::Rect dlibRectangleToOpenCV(dlib::rectangle r)
{
  return cv::Rect(cv::Point2i(r.left(), r.top()), cv::Point2i(r.right() + 1, r.bottom() + 1));
}

dlib::rectangle openCVRectToDlib(cv::Rect r)
{
  return dlib::rectangle((long)r.tl().x, (long)r.tl().y, (long)r.br().x - 1, (long)r.br().y - 1);
}

#endif // UTILS_H
