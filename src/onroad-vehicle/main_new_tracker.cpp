#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "DetectorHaarCascade.h"
#include "FilterFalsePositives.h"
#include <vector>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

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
    do {
        if (!capture.read(frame))
            exit_with_message("Unable to read next frame.");

        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        cars = cascade.detect(frame);

        cars = fp.filter(frame_gray, cars, FilterType::MEAN_SQUARE);
        //cars_filtered = fp.filter(frame, cars, FilterType::HSV_ROAD);
        for (auto& car : cars) {
            rectangle(frame, car, Scalar(0, 255, 0), 2);
        }
        imshow("Camera", frame);

        keyboard = waitKey(30);
    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );

    return EXIT_SUCCESS;
}