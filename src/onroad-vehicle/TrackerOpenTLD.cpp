#include "TrackerOpenTLD.h"

using namespace tld;
using namespace std;

TrackerOpenTLD::TrackerOpenTLD(cv::Mat frame_gray, cv::Rect window) :
    tld(make_unique<TLD>()), initial_window(window), initial_frame_gray(frame_gray)
{
    tld->detectorCascade->imgWidth = frame_gray.cols;
    tld->detectorCascade->imgHeight = frame_gray.rows;
    tld->detectorCascade->imgWidthStep = frame_gray.step;

    tld->selectObject(frame_gray, &window);
}

TrackerOpenTLD::TrackerOpenTLD(const TrackerOpenTLD &other) :
        TrackerOpenTLD(other.initial_frame_gray, other.initial_window)
{

}

TrackerOpenTLD::~TrackerOpenTLD() {

}

cv::Rect* TrackerOpenTLD::detect(cv::Mat frame) {
    tld->processImage(frame);
    return tld->currBB;
}


cv::Rect* TrackerOpenTLD::get_current_bounding() {
    return tld->currBB;
}
