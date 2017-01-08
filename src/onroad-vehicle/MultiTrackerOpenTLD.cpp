#include <opencv2/core/mat.hpp>
#include "MultiTrackerOpenTLD.h"

using namespace cv;
using namespace std;

MultiTrackerOpenTLD::MultiTrackerOpenTLD() : current_index(0) {

}

void MultiTrackerOpenTLD::add_tracker(cv::Rect window, cv::Mat frame) {
    current_index += 1;
    trackers.insert(std::pair<int, TrackerOpenTLD>(current_index, TrackerOpenTLD(frame, window)));
}

void MultiTrackerOpenTLD::update_trackers(cv::Mat frame) {
    vector<int> keys_to_delete;
    for (auto& pair_tracker : trackers) {
        Rect* bounding = pair_tracker.second.detect(frame);
        if (bounding == nullptr)
            keys_to_delete.push_back(pair_tracker.first);
    }

    for (auto& key : keys_to_delete)
        trackers.erase(key);
}

std::map<int, TrackerOpenTLD>& MultiTrackerOpenTLD::get_trackers() {
    return trackers;
}

void MultiTrackerOpenTLD::check_for_dead_trackers() {
}