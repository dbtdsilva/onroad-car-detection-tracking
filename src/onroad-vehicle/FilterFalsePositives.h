#ifndef ONROAD_VEHICLE_FILTERFPMEANSQUARE_H
#define ONROAD_VEHICLE_FILTERFPMEANSQUARE_H

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <string>

enum class FilterType { MEAN_SQUARE, HSV_ROAD };

class FilterFalsePositives {
public:
    FilterFalsePositives();

    std::vector<cv::Rect> filter(cv::Mat frame, std::vector<cv::Rect> obj, FilterType filter);
private:
    std::vector<cv::Rect> filterMeanSquare(cv::Mat frame, std::vector<cv::Rect> obj);
    std::vector<cv::Rect> filterHSVRoad(cv::Mat frame, std::vector<cv::Rect> obj);
    double mse(const cv::Mat& frame1, const cv::Mat& frame2);
    double diffUpDown(const cv::Mat& in);
    double diffLeftRight(const cv::Mat& in);

    std::vector<cv::Scalar> average_road_hsv;
};

#endif //ONROAD_VEHICLE_FILTERFPMEANSQUARE_H