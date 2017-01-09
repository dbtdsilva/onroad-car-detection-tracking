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
    std::vector<cv::Rect> filterMeanSquare(cv::Mat frame, std::vector<cv::Rect> obj, int lower_x = 80,
                                           int upper_x = 150, int lower_y = 200);
    std::vector<cv::Rect> filterHSVRoad(cv::Mat frame, std::vector<cv::Rect> obj);
private:
    double mse(const cv::Mat& frame1, const cv::Mat& frame2);
    double diffUpDown(const cv::Mat& in);
    double diffLeftRight(const cv::Mat& in);

    std::vector<cv::Scalar> average_road_hsv;
};

#endif //ONROAD_VEHICLE_FILTERFPMEANSQUARE_H