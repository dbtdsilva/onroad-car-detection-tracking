#include "MultiTrackerOpenCV.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

MultiTrackerOpenCV::MultiTrackerOpenCV() : current_index(0) {

}

void MultiTrackerOpenCV::add_tracker(cv::Rect window, cv::Mat frame) {
    current_index += 1;
    TrackerData data;
    data.tracker = cv::Tracker::create("BOOSTING");
    data.bounding_box = Rect2d(window);
    data.misses = 0;
    data.tracker->init(frame, data.bounding_box);
    trackers.insert(pair<int, TrackerData>(current_index, data));
}

void MultiTrackerOpenCV::update_trackers(cv::Mat frame) {
    vector<int> keys_to_remove;
    for (auto& pair_tracker : trackers) {
        pair_tracker.second.misses = pair_tracker.second.tracker->update(frame, pair_tracker.second.bounding_box) ?
                                     0 : pair_tracker.second.misses + 1;

        Rect2d box = pair_tracker.second.bounding_box;
        if (pair_tracker.second.misses >= 5 || box.y + box.height > frame.rows || box.x + box.width > frame.cols) {
            keys_to_remove.push_back(pair_tracker.first);
        }
    }

    for (int& key : keys_to_remove) {
        trackers.erase(key);
    }
}

void MultiTrackerOpenCV::replace_bounding_box(int car_key, cv::Rect box, cv::Mat frame) {
    trackers.erase(trackers.find(car_key));
    TrackerData data;
    data.tracker = cv::Tracker::create("BOOSTING");
    data.bounding_box = Rect2d(box);
    data.misses = 0;

    data.tracker->init(frame, data.bounding_box);
    trackers.insert(pair<int, TrackerData>(car_key, data));
}

std::map<int, TrackerData> &MultiTrackerOpenCV::get_trackers() {
    return trackers;
}
