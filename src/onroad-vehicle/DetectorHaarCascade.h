#ifndef ONROAD_VEHICLE_DETECTORHAARCASCADE_H
#define ONROAD_VEHICLE_DETECTORHAARCASCADE_H

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <vector>
#include <string>

class DetectorHaarCascade {
public:
    DetectorHaarCascade(std::string cascade_location);

    std::vector<cv::Rect> detect(cv::Mat frame);
private:
    cv::CascadeClassifier classifier_;
};


#endif //ONROAD_VEHICLE_DETECTORHAARCASCADE_H
