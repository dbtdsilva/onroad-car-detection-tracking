#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "DetectorHaarCascade.h"
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
    vector<Rect> cars;
    Mat frame_gray;

    DetectorHaarCascade cascade(argv[2]);

    do {
        if (!capture.read(frame))
            exit_with_message("Unable to read next frame.");


        cars = cascade.detect(frame);

        for (auto& car : cars)
            rectangle(frame, car, Scalar(0, 255, 0), 2);
        imshow("Camera", frame);
        keyboard = waitKey(100);
    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );

    return EXIT_SUCCESS;
}