#include "TrackerOpenTLD.h"

using namespace tld;
using namespace std;

TrackerOpenTLD::TrackerOpenTLD(cv::Mat frame, cv::Rect window) :
    tld(make_unique<TLD>()), initial_window(window)
{
    tld->detectorCascade->imgWidth = frame.cols;
    tld->detectorCascade->imgHeight = frame.rows;
    tld->detectorCascade->imgWidthStep = frame.step;

    tld->selectObject(frame, &window);
}

cv::Rect* TrackerOpenTLD::detect(cv::Mat frame) {
    tld->processImage(frame);
    return tld->currBB;
}


cv::Rect* TrackerOpenTLD::get_current_bounding() {
    return tld->currBB == nullptr ? tld->currBB : tld->prevBB;
}
