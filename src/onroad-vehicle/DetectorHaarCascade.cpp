#include "DetectorHaarCascade.h"

using namespace std;
using namespace cv;


DetectorHaarCascade::DetectorHaarCascade(string cascade_location) : DetectorHaarCascade(cascade_location, 2) {

}

DetectorHaarCascade::DetectorHaarCascade(string cascade_location, double cascade_scale) :
    cascade_scale_(cascade_scale) {
    if (!classifier_.load(cascade_location))
        throw invalid_argument("Failed to load cascade file");
}

vector<Rect> DetectorHaarCascade::detect(Mat frame) {
    vector<Rect> detections;
    cvtColor(frame, frame, COLOR_BGR2GRAY);
    classifier_.detectMultiScale(frame, detections, 1.2, 2);
    return detections;
}

