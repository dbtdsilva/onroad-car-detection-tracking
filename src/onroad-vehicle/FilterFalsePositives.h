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
};

#endif //ONROAD_VEHICLE_FILTERFPMEANSQUARE_H