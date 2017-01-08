#ifndef ONROAD_VEHICLE_MULTITRACKEROPENCV_H
#define ONROAD_VEHICLE_MULTITRACKEROPENCV_H


#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/tracking.hpp>
#include <map>



typedef struct {
    cv::Ptr<cv::Tracker> tracker;
    cv::Rect2d bounding_box;
    int misses;
} TrackerData;

class MultiTrackerOpenCV {
public:
    MultiTrackerOpenCV();

    void add_tracker(cv::Rect window, cv::Mat frame);
    void update_trackers(cv::Mat frame);
    void replace_bounding_box(int car_key, cv::Rect box, cv::Mat frame);
    std::map<int, TrackerData>& get_trackers();
private:
    int current_index;
    std::map<int, TrackerData> trackers;
};


#endif //ONROAD_VEHICLE_MULTITRACKEROPENCV_H
