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
    vector<Rect> cars, cars_filtered;
    Mat frame_gray;

    /*Mat ycrcb;
    cvtColor(frame,ycrcb,CV_BGR2YCrCb);
    vector<Mat> channels;
    split(ycrcb,channels);
    equalizeHist(channels[0], channels[0]);
    Mat result;
    merge(channels,ycrcb);
    cvtColor(ycrcb,result,CV_YCrCb2BGR);
    cvtColor(result,result,COLOR_BGR2GRAY);
    imshow("Equalized before", result);*/

    DetectorHaarCascade cascade(argv[2]);
    FilterFalsePositives fp;

    vector<Mat> history;
    int frame_counter = 0;

    Rect2d roi, roi2;
    Mat frame_equalized;
    bool selected = false;
    int tracker_frames = 0;
    TrackerOpenTLD* tracker;
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
        //cars = fp.filter(frame.clone(), cars, FilterType::HSV_ROAD);
        for (auto& car : cars) {
            if (!selected) {
                selected = true;
                cout << "Selected.." << endl;
                tracker = new TrackerOpenTLD(frame_gray, car);
            }
            rectangle(frame_final, car, Scalar(0, 255, 0), 2);
        }

        if (selected) {
            //trackerBoosting->update(frame, roi);
            Rect* window = tracker->detect(frame);
            if (window != nullptr) {
            //if (tracker->update(frame, roi)) {
                rectangle(frame_final, *window, Scalar(0, 0, 255), 2);
            } else {
                /*tracker_frames++;
                cout << tracker_frames << endl;
                if (tracker_frames >= 25) {
                    selected = false;
                    tracker_frames = 0;
                }*/
            }
        }
        imshow("Camera", frame_final);

        keyboard = waitKey(20);
    } while((char)keyboard != 'q' && (char)keyboard != 'Q');

    return EXIT_SUCCESS;
}