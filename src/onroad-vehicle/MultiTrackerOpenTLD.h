#ifndef ONROAD_VEHICLE_MULTITRACKEROPENTLD_H
#define ONROAD_VEHICLE_MULTITRACKEROPENTLD_H

#include <vector>
#include <map>

#include "TrackerOpenTLD.h"

class MultiTrackerOpenTLD {
public:
    MultiTrackerOpenTLD();

    void add_tracker(cv::Rect window, cv::Mat frame);
    void update_trackers(cv::Mat frame);
    std::map<int, TrackerOpenTLD>& get_trackers();
    void check_for_dead_trackers();
private:
    int current_index;
    std::map<int, TrackerOpenTLD> trackers;
};


#endif //ONROAD_VEHICLE_MULTITRACKEROPENTLD_H
