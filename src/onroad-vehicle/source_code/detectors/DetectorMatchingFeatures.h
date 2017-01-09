#ifndef ONROAD_VEHICLE_DETECTIONMATCHINGFEATURES_H
#define ONROAD_VEHICLE_DETECTIONMATCHINGFEATURES_H

#include <opencv2/imgproc.hpp>
#include <vector>

class DetectorMatchingFeatures {
public:
    DetectorMatchingFeatures();
    DetectorMatchingFeatures(int history);
    DetectorMatchingFeatures(int history, int num_features);

    std::vector<cv::Rect> detect(cv::Mat frame);
private:
    const int history_;
    std::vector<cv::Mat> buffer_;
    int frames_received_, num_features_;
};

#endif //ONROAD_VEHICLE_DETECTIONMATCHINGFEATURES_H
