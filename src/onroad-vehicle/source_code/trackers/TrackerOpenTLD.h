#ifndef ONROAD_VEHICLE_TRACKERTLD_H
#define ONROAD_VEHICLE_TRACKERTLD_H

#include <memory>
#include <opencv2/core/types.hpp>

#include "tld/TLDUtil.h"
#include "tld/TLD.h"

class TrackerOpenTLD {
public:
    TrackerOpenTLD(cv::Mat frame_gray, cv::Rect);
    TrackerOpenTLD(const TrackerOpenTLD& other);
    ~TrackerOpenTLD();

    cv::Rect* detect(cv::Mat frame);
    cv::Rect* get_current_bounding();
private:
    std::unique_ptr<tld::TLD> tld;

    cv::Mat initial_frame_gray;
    cv::Rect initial_window;
    cv::Rect window;
};

#endif //ONROAD_VEHICLE_TRACKERTLD_H
