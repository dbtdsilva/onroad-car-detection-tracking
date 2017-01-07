#ifndef ONROAD_VEHICLE_FILTERFPMEANSQUARE_H
#define ONROAD_VEHICLE_FILTERFPMEANSQUARE_H

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <string>

enum class FilterType { MEAN_SQUARE };

class FilterFalsePositives {
public:
    FilterFalsePositives();

    bool filter(cv::Mat frame, FilterType filter);
private:
    bool filterMeanSquare(cv::Mat frame);
    double mse(const cv::Mat& frame1, const cv::Mat& frame2);
    double diffUpDown(const cv::Mat& in);
    double diffLeftRight(const cv::Mat& in);
};

#endif //ONROAD_VEHICLE_FILTERFPMEANSQUARE_H