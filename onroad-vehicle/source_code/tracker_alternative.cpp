#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <cstring>
#include <string>

#include "extra/FilterFalsePositives.h"
#include "detectors/DetectorHaarCascade.h"
#include "trackers/MultiTrackerOpenCV.h"
#include "extra/Helper.h"

using namespace std;
using namespace cv;


double mse(const Mat& frame1, const Mat& frame2) {
    Mat s1;
    Mat f1 = frame1.clone();
    Mat f2 = frame2.clone();
    f1.convertTo(f1, CV_32F);
    f2.convertTo(f2, CV_32F);

    absdiff(frame1, frame2, s1); // |I1 - I2|
    s1 = s1.mul(s1);             // |I1 - I2|^2

    Scalar s = sum(s1);
    double mse  = s[0] / (double)(frame1.size().height * frame1.size().width);
    return mse;
}

double diffUpDown(const Mat& in) {
    Mat frame = in.clone();
    int height = frame.size().height;
    int width = frame.size().width;
    int half = height / 2;

    Rect topCrop(0, 0, width, half);
    Rect bottomCrop(0, half, width, half);
    Mat top = frame(topCrop);
    Mat bottom = frame(bottomCrop);

    flip(top, top, 1);
    resize(bottom, bottom, Size(32, 64));
    resize(top, top, Size(32,64));
    return mse(top, bottom);
}

double diffLeftRight(const Mat& in) {
    Mat frame = in.clone();
    int height = frame.size().height;
    int width = frame.size().width;
    int half = width / 2;

    Rect leftCrop(0, 0, half, height);
    Rect rightCrop(half, 0, half, height);
    Mat left = frame(leftCrop);
    Mat right = frame(rightCrop);

    flip(right, right, 1);
    resize(left, left, Size(32, 64));
    resize(right, right, Size(32,64));

    return mse(left, right);
}

bool isNewRoi(const Rect& rect, vector<Rect> &rects){
    for (vector<Rect>::iterator it = rects.begin(); it != rects.end(); ++it) {
        if(abs((*it).x - rect.x) < 40 && abs((*it).y - rect.y) < 40)
            return false;
    }
    return true;
}

double overlapPercentage(const Rect& A, const Rect2d& B){
    int x_overlap = max(0, min(A.x+A.width,(int)B.x + (int)B.width) - max(A.x,(int)B.x));
    int y_overlap = max(0, min(A.y+A.height,(int)B.y + (int)B.height) - max(A.y,(int)B.y));
    int overlapArea = x_overlap * y_overlap;
    return (double)overlapArea/(A.width*A.height);
}


int main(int argc, const char** argv)
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " video_file cascade_classifier.xml" << endl;
        return -1;
    }

    Mat frame;
    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    int keyboard;
    vector<Rect> cars;

    int actualFrameCount = 0;
    Mat frame_gray;

    FilterFalsePositives fp_filter;
    DetectorHaarCascade detector(argv[2]);
    // road.mp4
    //MultiTrackerOpenCV multi_tracker("KCF");
    // camera/
    MultiTrackerOpenCV multi_tracker("BOOSTING");
    // cacia2
    Mat final_frame;
    do {
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        final_frame = frame.clone();

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        // Road sample video (road.mp4)
        //cars = detector.detect(frame, Size(80, 80), 2, 1.2, false);
        //cars = fp_filter.filterMeanSquare(frame_gray, cars);

        // Camera recoded videos (camera/)
        cars = detector.detect(frame, Size(50, 50), 1, 1.05, true);
        cars = fp_filter.filterMeanSquare(frame_gray, cars, 120, 170, 210);
        //cars = fp.filter(frame.clone(), cars, FilterType::HSV_ROAD);
        for (auto& car : cars) {
            bool replaced = false;
            for (auto& tracker : multi_tracker.get_trackers()) {
                if (Helper::overlapPercentage(tracker.second.bounding_box, car) > 0.2) {
                    multi_tracker.replace_bounding_box(tracker.first, car, frame);
                    replaced = true;
                    break;
                }
            }
            if (!replaced) {
                multi_tracker.add_tracker(car, frame);
            }
            rectangle(final_frame, car, Scalar(255, 0, 0), 2);
        }

        multi_tracker.update_trackers(frame);
        for (auto& trackers : multi_tracker.get_trackers()) {
            TrackerData& data = trackers.second;
            rectangle(final_frame, trackers.second.bounding_box, Scalar(0, 0, 255), 2);
            putText(final_frame, "Car "+to_string(trackers.first),
                    cvPoint((int)data.bounding_box.x + 20, (int)data.bounding_box.y + 20),
                    FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);
        }

        putText(final_frame, "Frame "+to_string(actualFrameCount++), cvPoint(8,20),
                FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,0), 1, CV_AA);

        imshow("Camera", final_frame);
        keyboard = waitKey( 30 );

    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );
    return EXIT_SUCCESS;
}