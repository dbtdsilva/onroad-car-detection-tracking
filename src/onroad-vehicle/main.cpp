#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(int argc, const char** argv)
{
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " video_file cascade_classifier.xml" << endl;
        return -1;
    }

    Mat frame;
    CascadeClassifier car_cascade;
    if(!car_cascade.load(argv[2])) {
        cout << "Failed to load " << argv[2] << endl;
        exit(EXIT_FAILURE);
    }

    VideoCapture capture = VideoCapture(argv[1]);
    if(!capture.isOpened()){
        cerr << "Unable to open video file: " << argv[1] << endl;
        exit(EXIT_FAILURE);
    }

    int keyboard;
    do {
        if (!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        imshow("Camera", frame);
        keyboard = waitKey( 30 );
    } while((char)keyboard != 'q' && (char)keyboard != 'Q' && keyboard != 27 );

    return EXIT_SUCCESS;
}