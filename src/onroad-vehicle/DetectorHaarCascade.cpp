#include "opencv2/highgui/highgui.hpp"
#include "DetectorHaarCascade.h"

using namespace std;
using namespace cv;

double overlapPercentage2(const Rect& A, const Rect2d& B){
    int x_overlap = max(0, min(A.x+A.width,(int)B.x + (int)B.width) - max(A.x,(int)B.x));
    int y_overlap = max(0, min(A.y+A.height,(int)B.y + (int)B.height) - max(A.y,(int)B.y));
    int overlapArea = x_overlap * y_overlap;
    return (double)overlapArea/(A.width*A.height);
}

DetectorHaarCascade::DetectorHaarCascade(string cascade_location) {
    if (!classifier_.load(cascade_location))
        throw invalid_argument("Failed to load cascade file");
}

vector<Rect> DetectorHaarCascade::detect(Mat frame, Size min_size, int neighbours, double scale, bool equalized) {
    vector<Rect> detections, detections_equalized;
    Mat frame_equalized;
    cvtColor(frame, frame, COLOR_BGR2GRAY);

    if (equalized) {
        equalizeHist(frame, frame_equalized);
        classifier_.detectMultiScale(frame_equalized, detections_equalized, scale, neighbours, 0, min_size);

        /*vector<Rect> final;
        for (auto& dec : detections_equalized) {
            for (auto& dec2 : detections) {
                if (overlapPercentage2(dec, dec2) >= 0.9) {
                    final.push_back(dec);
                    break;
                }
            }
        }*/
    } else {
        classifier_.detectMultiScale(frame, detections, scale, neighbours, 0, min_size);
    }
    return equalized ? detections_equalized : detections;
}

