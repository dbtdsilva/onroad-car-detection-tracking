#ifndef ONROAD_VEHICLE_HELPER_H
#define ONROAD_VEHICLE_HELPER_H

#include <opencv2/core/types.hpp>

class Helper {
public:
    static double overlapPercentage(const cv::Rect& A, const cv::Rect& B);
};

#endif //ONROAD_VEHICLE_HELPER_H
