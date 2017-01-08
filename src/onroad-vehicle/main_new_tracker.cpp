#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <opencv2/tracking.hpp>

#include "DetectorHaarCascade.h"
#include "FilterFalsePositives.h"
#include "TrackerOpenTLD.h"
#include "MultiTrackerOpenTLD.h"
#include "Helper.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

void exit_with_message(string value) {
    cerr << value << endl;
    exit(EXIT_FAILURE);
}

string help_message(string program_name) {
    stringstream ss;
    ss << program_name << " video_file cascade_file.xml";
    return ss.str();
}

int main(int argc, const char** argv)
{
    if (argc < 3)
        exit_with_message(help_message(argv[0]));

    Mat frame;
    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened())
        exit_with_message("Unable to open video file: ");

    int keyboard;
    vector<Rect> cars;
    Mat frame_gray;

    DetectorHaarCascade cascade(argv[2]);
    FilterFalsePositives fp;

    vector<Mat> history;
    MultiTrackerOpenTLD multi_tracker;
    do {
        if (!capture.read(frame))
            exit_with_message("Unable to read next frame.");

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        cars = cascade.detect(frame, Size(45, 45), 1, 1.05, true);

        Mat frame_pre = frame.clone();
        Mat frame_final = frame.clone();
        for (auto& car : cars)
            rectangle(frame_pre, car, Scalar(0, 255, 0), 2);
        imshow("Camera before filtering", frame_pre);

        cars = fp.filter(frame_gray, cars, FilterType::MEAN_SQUARE);
        cars = fp.filter(frame.clone(), cars, FilterType::HSV_ROAD);
        for (auto& car : cars) {
            bool already_in_list = false;
            for (auto& pair_tracker : multi_tracker.get_trackers()) {
                if (pair_tracker.second.get_current_bounding() == nullptr) continue;

                if (Helper::overlapPercentage(car, *(pair_tracker.second.get_current_bounding())) > 0.3) {
                    already_in_list = true;
                    break;
                }
            }
            if (!already_in_list)
                multi_tracker.add_tracker(car, frame_gray);
            rectangle(frame_final, car, Scalar(0, 255, 0), 2);
        }

        multi_tracker.update_trackers(frame);

        for (auto& pair_tracker : multi_tracker.get_trackers()) {
            Rect* bounding = pair_tracker.second.get_current_bounding();
            if (bounding != nullptr) {
                putText(frame_final, "Car "+to_string(pair_tracker.first),
                        cvPoint(bounding->x + 20, bounding->y + 20), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
                rectangle(frame_final, *bounding, Scalar(0, 0, 255), 2);
            }
        }
        imshow("Camera", frame_final);


        keyboard = waitKey(0);
    } while((char)keyboard != 'q' && (char)keyboard != 'Q');

    return EXIT_SUCCESS;
}