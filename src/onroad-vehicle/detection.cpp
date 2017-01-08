#include <stdio.h>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "DetectorMatchingFeatures.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    DetectorMatchingFeatures matcher(6);
    int keyboard;
    do {
        Mat img_object;
        if (!capture.read(img_object)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting." << endl;
            exit(EXIT_FAILURE);
        }

        matcher.detect(img_object);
        keyboard = waitKey(1000);
    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27);
    return 0;
}